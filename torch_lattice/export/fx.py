from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

import torch
from torch import fx, nn
from torch.utils import _pytree

import torch_lattice
from torch_lattice import nn as spnn
from torch_lattice import operators as lattice_ops

from .builder import ExportValue, TorchLatticeExportBuilder

__all__ = ["LatticeExportInterpreter", "LatticeTracer", "lower_fx_module"]


SUPPORTED_LEAF_MODULES = (
    spnn.Conv3d,
    spnn.SubmConv3d,
    spnn.ConvTranspose3d,
    spnn.GenerativeConvTranspose3d,
    spnn.BatchNorm,
    spnn.InstanceNorm,
    spnn.GroupNorm,
    spnn.ReLU,
    spnn.LeakyReLU,
    spnn.SiLU,
    spnn.GlobalAvgPool,
    spnn.GlobalMaxPool,
    nn.Linear,
    nn.ReLU,
    nn.LeakyReLU,
    nn.SiLU,
    nn.Identity,
)

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

_STRUCTURAL_FUNCTIONS = frozenset((operator.getitem,))


class LatticeTracer(fx.Tracer):
    """FX tracer that preserves supported lattice modules and ops."""

    def __init__(self) -> None:
        super().__init__(
            autowrap_modules=(torch_lattice, lattice_ops),
            autowrap_functions=tuple(
                _CAT_FUNCTIONS | frozenset(_BINARY_FUNCTIONS)
            ),
        )

    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(module, SUPPORTED_LEAF_MODULES):
            return True
        return super().is_leaf_module(module, module_qualified_name)


class LatticeExportInterpreter(fx.Interpreter):
    """Lower an FX graph by interpreting it with symbolic lattice values."""

    def __init__(
        self,
        module: fx.GraphModule,
        builder: TorchLatticeExportBuilder,
    ) -> None:
        super().__init__(module)
        self.builder = builder

    def call_module(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ExportValue:
        module = self.fetch_attr(str(target))
        value = _single_export_value(args, kwargs, context=str(target))
        return self.builder.lower_module(str(target), module, value)

    def call_function(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if target in _CAT_FUNCTIONS:
            return self._cat(args, kwargs)
        if target in _BINARY_FUNCTIONS:
            return self._binary(_BINARY_FUNCTIONS[target], args, kwargs)
        if target in _STRUCTURAL_FUNCTIONS:
            return super().call_function(target, args, kwargs)
        raise ValueError(f"unsupported FX function for lattice export: {target}")

    def call_method(
        self,
        target: fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        raise ValueError(f"unsupported FX method for lattice export: {target}")

    def _cat(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ExportValue:
        values = _export_values(args, kwargs)
        if len(values) < 2:
            raise ValueError("lattice cat export requires at least two sparse values.")
        out = values[0]
        stem = _current_node_name(self, "cat")
        for index, value in enumerate(values[1:], start=1):
            out = self.builder.sparse_cat(f"{stem}_{index}", out, value)
        return out

    def _binary(
        self,
        op: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ExportValue:
        values = _export_values(args, kwargs)
        if len(values) != 2:
            raise ValueError(
                f"lattice {op} export requires exactly two sparse values."
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


def lower_fx_module(
    builder: TorchLatticeExportBuilder,
    model: nn.Module,
) -> TorchLatticeExportBuilder:
    if isinstance(model, SUPPORTED_LEAF_MODULES):
        builder.module(type(model).__name__.lower(), model)
        builder.output()
        return builder

    graph = LatticeTracer().trace(model)
    graph_module = fx.GraphModule(model, graph)
    result = LatticeExportInterpreter(graph_module, builder).run(builder.current)
    builder.output(_single_output_value(result))
    return builder


def _default_join(op: str) -> str:
    if op in {"mul", "maximum", "minimum"}:
        return "inner"
    return "outer"


def _single_export_value(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    context: str,
) -> ExportValue:
    values = _export_values(args, kwargs)
    if len(values) != 1:
        raise ValueError(
            f"module {context} must consume exactly one symbolic lattice value."
        )
    return values[0]


def _single_output_value(value: Any) -> ExportValue:
    values = _export_values(value)
    if len(values) != 1:
        raise ValueError("lattice export currently supports one model output.")
    return values[0]


def _export_values(*values: Any) -> list[ExportValue]:
    leaves: list[Any] = []
    for value in values:
        flat, _ = _pytree.tree_flatten(value)
        leaves.extend(flat)
    return [value for value in leaves if isinstance(value, ExportValue)]


def _current_node_name(interpreter: LatticeExportInterpreter, fallback: str) -> str:
    node = getattr(interpreter, "current_node", None)
    name = getattr(node, "name", None)
    return str(name or fallback)
