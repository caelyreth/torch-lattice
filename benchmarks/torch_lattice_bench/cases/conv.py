from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn

from torch_lattice_bench.cases.common import (
    F,
    conv_module,
    fresh_sparse,
    set_conv_config,
)
from torch_lattice_bench.datasets import (
    SparseFixture,
    params_matrix,
    sparse_fixture,
)
from torch_lattice_bench.harness import BenchmarkCase, SkipCase


@dataclass(frozen=True, slots=True)
class ConvFixture:
    base: SparseFixture
    module: torch.nn.Module
    dataflow: F.Dataflow | None
    kwargs: dict[str, Any]
    min_spatial_extent: int


@dataclass(frozen=True, slots=True)
class ConvPrepared:
    x: SparseTensor
    module: torch.nn.Module
    dataflow: F.Dataflow | None
    kwargs: dict[str, Any]
    min_spatial_extent: int


@dataclass(frozen=True, slots=True)
class TargetTransposePrepared:
    source: SparseTensor
    target: SparseTensor
    module: torch.nn.Module


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
        preset,
        n_values=n_values,
        channels=channels,
        layouts=layouts,
        dtype=dtype,
    )
    ordinary = tuple(
        BenchmarkCase(
            name=name,
            group="conv",
            params=params,
            setup=_setup_factory(
                device,
                kernel_size,
                stride,
                dataflow,
                kwargs,
                min_extent,
                subm,
                normalized,
            ),
            prepare=_prepare,
            run=_run_conv,
            metrics=_conv_metrics,
            units=("elements",),
        )
        for name, dataflow, kwargs, kernel_size, stride, min_extent, subm, normalized in _specs()
    )
    target_transpose = tuple(
        BenchmarkCase(
            name=name,
            group="conv",
            params=params,
            setup=lambda p, normalized=normalized: _setup_target_transpose(
                dict(p), device, normalized=normalized
            ),
            prepare=lambda fixture: fixture,
            run=_run_target_transpose,
            metrics=lambda params, fixture, prepared, output: {
                "elements": int(output.feats.numel())
            },
            units=("elements",),
        )
        for name, normalized in (
            ("conv_transpose3_target", False),
            ("normalized_conv_transpose3_target", True),
        )
    )
    return (*ordinary, *target_transpose)


def _specs() -> tuple[
    tuple[str, F.Dataflow | None, dict[str, Any], int, int, int, bool, bool], ...
]:
    return (
        ("conv1x1_matmul", None, {}, 1, 1, 1, False, False),
        (
            "conv3_implicit_gemm_unsorted",
            F.Dataflow.ImplicitGEMM,
            {"ifsort": False, "split_mask_num": 1},
            3,
            1,
            1,
            False,
            False,
        ),
        (
            "conv3_implicit_gemm_sorted",
            F.Dataflow.ImplicitGEMM,
            {"ifsort": True, "split_mask_num": 3},
            3,
            1,
            1,
            False,
            False,
        ),
        (
            "conv3_fetch_on_demand_fused",
            F.Dataflow.FetchOnDemand,
            {"FOD_fusion": True},
            3,
            1,
            1,
            False,
            False,
        ),
        (
            "conv3_fetch_on_demand_no_fusion",
            F.Dataflow.FetchOnDemand,
            {"FOD_fusion": False},
            3,
            1,
            1,
            False,
            False,
        ),
        (
            "conv3_gather_scatter",
            F.Dataflow.GatherScatter,
            {"ifsort": False, "split_mask_num": 1},
            3,
            1,
            1,
            False,
            False,
        ),
        (
            "conv2_stride2_implicit",
            F.Dataflow.ImplicitGEMM,
            {"ifsort": True, "split_mask_num": 2},
            2,
            2,
            2,
            False,
            False,
        ),
        (
            "subm3_implicit_gemm_unsorted",
            F.Dataflow.ImplicitGEMM,
            {"ifsort": False, "split_mask_num": 1},
            3,
            1,
            1,
            True,
            False,
        ),
        (
            "normalized_subm3_gather_scatter",
            F.Dataflow.GatherScatter,
            {"ifsort": False, "split_mask_num": 1},
            3,
            1,
            1,
            True,
            True,
        ),
    )


def _setup_factory(
    device: torch.device,
    kernel_size: int,
    stride: int,
    dataflow: F.Dataflow | None,
    kwargs: dict[str, Any],
    min_extent: int,
    subm: bool,
    normalized: bool,
) -> Callable[[dict[str, object]], ConvFixture]:
    def setup(params: dict[str, object]) -> ConvFixture:
        base = sparse_fixture(params, device=device)
        if subm:
            module_type = spnn.NormalizedSubmConv3d if normalized else spnn.SubmConv3d
            module = module_type(
                base.channels,
                base.channels,
                kernel_size=kernel_size,
                bias=False,
            ).to(device)
            if base.tensor.feats.dtype == torch.float16:
                module = module.half()
            module.eval()
        else:
            module = conv_module(
                base.channels,
                base.tensor.feats.dtype,
                device,
                kernel_size,
                stride,
            )
        return ConvFixture(base, module, dataflow, dict(kwargs), min_extent)

    return setup


def _prepare(fixture: ConvFixture) -> ConvPrepared:
    return ConvPrepared(
        fresh_sparse(fixture.base.tensor),
        fixture.module,
        fixture.dataflow,
        fixture.kwargs,
        fixture.min_spatial_extent,
    )


def _run_conv(prepared: ConvPrepared) -> SparseTensor:
    if min(prepared.x.spatial_range[1:]) < prepared.min_spatial_extent:
        raise SkipCase(
            f"spatial range {prepared.x.spatial_range[1:]} is too thin for "
            f"kernel extent {prepared.min_spatial_extent}"
        )
    if prepared.dataflow is not None:
        set_conv_config(prepared.dataflow, **prepared.kwargs)
    with torch.no_grad():
        return prepared.module(prepared.x)


def _conv_metrics(params, fixture, prepared, output) -> dict[str, int | float]:
    del params, fixture, prepared
    if isinstance(output, SparseTensor):
        return {"elements": int(output.feats.numel())}
    return {}


def _setup_target_transpose(
    params: dict[str, object],
    device: torch.device,
    *,
    normalized: bool,
) -> TargetTransposePrepared:
    base = sparse_fixture(params, device=device)
    target = fresh_sparse(base.tensor)
    source = SparseTensor(
        feats=target.feats,
        coords=target.coords,
        stride=2,
        spatial_range=target.spatial_range,
    )
    module_type = spnn.NormalizedConvTranspose3d if normalized else spnn.ConvTranspose3d
    module = module_type(
        base.channels,
        base.channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    ).to(device)
    if base.tensor.feats.dtype == torch.float16:
        module = module.half()
    return TargetTransposePrepared(source, target, module.eval())


def _run_target_transpose(prepared: TargetTransposePrepared) -> SparseTensor:
    with torch.no_grad():
        return prepared.module(prepared.source, prepared.target)
