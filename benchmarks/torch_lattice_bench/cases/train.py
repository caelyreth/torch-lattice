from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torch_lattice import SparseTensor

from torch_lattice_bench.cases.common import (
    F,
    conv_module,
    fresh_sparse,
    set_conv_config,
)
from torch_lattice_bench.datasets import SparseFixture, params_matrix, sparse_fixture
from torch_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class TrainFixture:
    base: SparseFixture
    module: torch.nn.Module
    dataflow: F.Dataflow
    kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainPrepared:
    x: SparseTensor
    module: torch.nn.Module
    dataflow: F.Dataflow
    kwargs: dict[str, Any]


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None,
    channels: tuple[int, ...] | None,
    layouts: tuple[str, ...] | None,
    dtype: str,
    device: torch.device,
) -> tuple[BenchmarkCase, ...]:
    params = params_matrix(
        preset, n_values=n_values, channels=channels, layouts=layouts, dtype=dtype
    )
    specs = (
        (
            "conv3_implicit_unsorted_forward_backward",
            F.Dataflow.ImplicitGEMM,
            {"ifsort": False, "split_mask_num": 1},
        ),
        (
            "conv3_implicit_sorted_forward_backward",
            F.Dataflow.ImplicitGEMM,
            {"ifsort": True, "split_mask_num": 3},
        ),
        (
            "conv3_fetch_on_demand_forward_backward",
            F.Dataflow.FetchOnDemand,
            {"FOD_fusion": False},
        ),
    )
    return tuple(
        BenchmarkCase(
            name=name,
            group="train",
            params=params,
            setup=_setup_factory(device, dataflow, kwargs),
            prepare=_prepare,
            run=lambda prepared: prepared,
            backward=_backward,
            units=("elements",),
            modes=("backward",),
        )
        for name, dataflow, kwargs in specs
    )


def _setup_factory(device: torch.device, dataflow: F.Dataflow, kwargs: dict[str, Any]):
    def setup(params: dict[str, object]) -> TrainFixture:
        base = sparse_fixture(params, device=device)
        module = conv_module(
            base.channels, base.tensor.feats.dtype, device, 3, 1
        ).train()
        return TrainFixture(base, module, dataflow, dict(kwargs))

    return setup


def _prepare(fixture: TrainFixture) -> TrainPrepared:
    return TrainPrepared(
        fresh_sparse(fixture.base.tensor),
        fixture.module,
        fixture.dataflow,
        fixture.kwargs,
    )


def _backward(prepared: TrainPrepared) -> torch.Tensor:
    set_conv_config(prepared.dataflow, **prepared.kwargs)
    prepared.module.zero_grad(set_to_none=True)
    feats = prepared.x.feats.detach().clone().requires_grad_(True)
    inp = SparseTensor(
        feats, prepared.x.coords, prepared.x.stride, prepared.x.spatial_range
    )
    out = prepared.module(inp).feats
    loss = out.float().square().mean()
    loss.backward()
    return loss.detach()
