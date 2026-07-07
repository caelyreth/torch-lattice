from __future__ import annotations

from torch import fx, nn

from torch_lattice import nn as spnn

from .builder import TorchLatticeExportBuilder

__all__ = ["LatticeTracer", "lower_fx_module"]


SUPPORTED_LEAF_MODULES = (
    spnn.Conv3d,
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


class LatticeTracer(fx.Tracer):
    """FX tracer that preserves supported lattice modules as graph leaves."""

    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(module, SUPPORTED_LEAF_MODULES):
            return True
        return super().is_leaf_module(module, module_qualified_name)


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
    modules = dict(graph_module.named_modules())
    env: dict[fx.Node, object] = {}

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if env:
                raise ValueError("lattice export currently supports one model input.")
            env[node] = builder.current
            continue
        if node.op == "call_module":
            module = modules[str(node.target)]
            _require_single_data_operand(node, env)
            env[node] = builder.module(str(node.target), module)
            continue
        if node.op == "output":
            result = node.args[0]
            if isinstance(result, tuple):
                if len(result) != 1:
                    raise ValueError("lattice export currently supports one model output.")
                result = result[0]
            if not isinstance(result, fx.Node) or result not in env:
                raise ValueError("lattice export output must reference a lowered node.")
            builder.output(env[result])
            continue
        raise ValueError(f"unsupported FX node for lattice export: {node.op} {node.target}")

    return builder


def _require_single_data_operand(node: fx.Node, env: dict[fx.Node, object]) -> None:
    data_args = [arg for arg in node.args if isinstance(arg, fx.Node)]
    if len(data_args) != 1 or data_args[0] not in env:
        raise ValueError(
            f"module {node.target!s} must consume exactly one previously lowered value."
        )
