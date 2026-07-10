from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import fx, nn
from torch.utils import _pytree

import torch_lattice
from torch_lattice import operators as lattice_ops
from torch_lattice.nn import functional as F

from .builder import (
    SUPPORTED_MODULE_TYPES,
    ArtifactValue,
    TorchLatticeArtifactBuilder,
)

__all__ = [
    "LatticeArtifactInterpreter",
    "LatticeTracer",
    "lower_fx_artifact",
]


_CAT_FUNCTIONS = frozenset(
    fn
    for fn in (
        torch_lattice.cat,
        lattice_ops.cat,
    )
    if fn is not None
)

_BINARY_FUNCTIONS = {
    fn: op
    for fn, op in (
        (operator.add, "add"),
        (torch.add, "add"),
        (torch_lattice.generative_add, "add"),
        (lattice_ops.generative_add, "add"),
        (torch_lattice.sparse_add, "add"),
        (lattice_ops.sparse_add, "add"),
        (operator.sub, "sub"),
        (torch.sub, "sub"),
        (torch_lattice.sparse_sub, "sub"),
        (lattice_ops.sparse_sub, "sub"),
        (operator.mul, "mul"),
        (torch.mul, "mul"),
        (torch_lattice.sparse_mul, "mul"),
        (lattice_ops.sparse_mul, "mul"),
        (torch.maximum, "maximum"),
        (torch_lattice.sparse_maximum, "maximum"),
        (lattice_ops.sparse_maximum, "maximum"),
        (torch.minimum, "minimum"),
        (torch_lattice.sparse_minimum, "minimum"),
        (lattice_ops.sparse_minimum, "minimum"),
    )
    if fn is not None
}

_VOXELIZE_FUNCTIONS = frozenset(
    fn for fn in (torch_lattice.voxelize, F.voxelize) if fn is not None
)
_DEVOXELIZE_FUNCTIONS = frozenset(
    fn for fn in (torch_lattice.devoxelize, F.devoxelize) if fn is not None
)
_REINDEX_FUNCTIONS = frozenset(
    fn
    for fn in (torch_lattice.reindex_sparse, lattice_ops.reindex_sparse)
    if fn is not None
)
_STRUCTURAL_FUNCTIONS = frozenset((operator.getitem,))
FxLoweringFn = Callable[
    [
        "LatticeArtifactInterpreter",
        fx.node.Target,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Any,
]


_FX_FUNCTION_LOWERINGS: dict[object, FxLoweringFn] = {}


def fx_function_lowering(
    *targets: object,
) -> Callable[[FxLoweringFn], FxLoweringFn]:
    """Register an FX function lowering."""

    def decorator(fn: FxLoweringFn) -> FxLoweringFn:
        for target in targets:
            if target in _FX_FUNCTION_LOWERINGS:
                raise ValueError(f"duplicate FX artifact lowering: {target}")
            _FX_FUNCTION_LOWERINGS[target] = fn
        return fn

    return decorator


class LatticeTracer(fx.Tracer):
    """FX tracer that preserves supported lattice modules and ops."""

    def __init__(self) -> None:
        super().__init__(
            autowrap_modules=(torch_lattice, lattice_ops, F),
            autowrap_functions=tuple(
                _CAT_FUNCTIONS
                | frozenset(_BINARY_FUNCTIONS)
                | _VOXELIZE_FUNCTIONS
                | _DEVOXELIZE_FUNCTIONS
                | _REINDEX_FUNCTIONS
            ),
        )

    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(module, SUPPORTED_MODULE_TYPES):
            return True
        return super().is_leaf_module(module, module_qualified_name)


class LatticeArtifactInterpreter(fx.Interpreter):
    """Lower an FX graph by interpreting it with symbolic lattice values."""

    def __init__(
        self, module: fx.GraphModule, builder: TorchLatticeArtifactBuilder
    ) -> None:
        super().__init__(module)
        self.builder = builder

    def call_module(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArtifactValue:
        module = self.fetch_attr(str(target))
        values = _artifact_values(args, kwargs)
        return self.builder.lower_module(str(target), module, *values)

    def call_function(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if target in _STRUCTURAL_FUNCTIONS:
            return super().call_function(target, args, kwargs)
        lowering = _FX_FUNCTION_LOWERINGS.get(target)
        if lowering is not None:
            return lowering(self, target, args, kwargs)
        raise ValueError(f"unsupported FX function for lattice artifact: {target}")

    def call_method(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        raise ValueError(f"unsupported FX method for lattice artifact: {target}")

    @fx_function_lowering(*_CAT_FUNCTIONS)
    def _cat(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArtifactValue:
        del target
        values = _artifact_values(args, kwargs)
        if len(values) < 2:
            raise ValueError(
                "lattice cat artifact requires at least two sparse values."
            )
        out = values[0]
        stem = _current_node_name(self, "cat")
        join = str(kwargs.get("join", "inner"))
        for index, value in enumerate(values[1:], start=1):
            out = self.builder.sparse_cat(f"{stem}_{index}", out, value, join=join)
        return out

    @fx_function_lowering(*_BINARY_FUNCTIONS)
    def _binary(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArtifactValue:
        values = _artifact_values(args, kwargs)
        op = _BINARY_FUNCTIONS[target]
        if len(values) != 2:
            raise ValueError(
                f"lattice {op} artifact requires exactly two sparse values."
            )
        return self.builder.sparse_binary(
            _current_node_name(self, op),
            values[0],
            values[1],
            op,
            join=str(kwargs.get("join", _default_join(op))),
            lhs_fill=float(kwargs.get("lhs_fill", 0.0)),
            rhs_fill=float(kwargs.get("rhs_fill", 0.0)),
        )

    @fx_function_lowering(*_VOXELIZE_FUNCTIONS)
    def _voxelize(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArtifactValue:
        del target
        return self.builder.voxelize(
            _current_node_name(self, "voxelize"),
            points=_artifact_arg(args, kwargs, 0, "points", context="voxelize"),
            features=_artifact_arg(args, kwargs, 1, "features", context="voxelize"),
            batch_indices=_artifact_arg(
                args, kwargs, 2, "batch_indices", context="voxelize"
            ),
            active_rows=_artifact_arg(
                args, kwargs, 3, "active_rows", context="voxelize"
            ),
            voxel_size=kwargs.get("voxel_size", 1.0),
            origin=kwargs.get("origin", 0.0),
            reduction=kwargs.get("reduction", "mean"),
            stride=kwargs.get("stride", 1),
        )

    @fx_function_lowering(*_DEVOXELIZE_FUNCTIONS)
    def _devoxelize(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArtifactValue:
        del target
        return self.builder.devoxelize(
            _current_node_name(self, "devoxelize"),
            points=_artifact_arg(args, kwargs, 0, "points", context="devoxelize"),
            voxels=_artifact_arg(args, kwargs, 1, "voxels", context="devoxelize"),
            batch_indices=_artifact_arg(
                args, kwargs, 2, "batch_indices", context="devoxelize"
            ),
            point_active_rows=_artifact_arg(
                args, kwargs, 3, "point_active_rows", context="devoxelize"
            ),
            voxel_size=kwargs.get("voxel_size", 1.0),
            origin=kwargs.get("origin", 0.0),
            interpolation=kwargs.get("interpolation", "nearest"),
        )

    @fx_function_lowering(*_REINDEX_FUNCTIONS)
    def _reindex_sparse(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArtifactValue:
        del target
        return self.builder.sparse_reindex(
            _current_node_name(self, "reindex"),
            _artifact_arg(args, kwargs, 0, "input", context="reindex_sparse"),
            _artifact_arg(args, kwargs, 1, "target", context="reindex_sparse"),
            fill=float(kwargs.get("fill", 0.0)),
        )


def lower_fx_artifact(
    builder: TorchLatticeArtifactBuilder,
    model: nn.Module,
    inputs: Iterable[ArtifactValue] | None = None,
    *,
    output_names: tuple[str, ...] | None = None,
) -> TorchLatticeArtifactBuilder:
    run_inputs = tuple(inputs) if inputs is not None else (builder.current,)
    if isinstance(model, SUPPORTED_MODULE_TYPES):
        result = builder.lower_module(type(model).__name__.lower(), model, *run_inputs)
        builder.output(result, names=output_names)
        return builder

    graph = LatticeTracer().trace(model)
    graph_module = fx.GraphModule(model, graph)
    result = LatticeArtifactInterpreter(graph_module, builder).run(*run_inputs)
    builder.output(*_output_values(result), names=output_names)
    return builder


def _artifact_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    position: int,
    name: str,
    *,
    context: str,
) -> ArtifactValue:
    value = kwargs.get(name, args[position] if position < len(args) else None)
    if not isinstance(value, ArtifactValue):
        raise ValueError(f"{context} artifact requires symbolic argument '{name}'.")
    return value


def _default_join(op: str) -> str:
    if op in {"mul", "maximum", "minimum"}:
        return "inner"
    return "outer"


def _output_values(value: Any) -> tuple[ArtifactValue, ...]:
    leaves, _ = _pytree.tree_flatten(value)
    if not leaves:
        raise ValueError("lattice artifact model produced no tensor outputs")
    invalid = [leaf for leaf in leaves if not isinstance(leaf, ArtifactValue)]
    if invalid:
        names = ", ".join(sorted({type(value).__name__ for value in invalid}))
        raise ValueError(
            "lattice artifact outputs must all be tensors; unsupported: " + names
        )
    return tuple(leaves)


def _artifact_values(*values: Any) -> list[ArtifactValue]:
    leaves: list[Any] = []
    for value in values:
        flat, _ = _pytree.tree_flatten(value)
        leaves.extend(flat)
    return [value for value in leaves if isinstance(value, ArtifactValue)]


def _current_node_name(interpreter: LatticeArtifactInterpreter, fallback: str) -> str:
    node = getattr(interpreter, "current_node", None)
    name = getattr(node, "name", None)
    return str(name or fallback)
