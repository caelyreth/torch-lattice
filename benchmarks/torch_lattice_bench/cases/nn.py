from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn

from torch_lattice_bench.datasets import (
    SparseFixture,
    fresh_sparse,
    params_matrix,
    sparse_fixture,
)
from torch_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class ModuleFixture:
    base: SparseFixture
    module: torch.nn.Module


@dataclass(frozen=True, slots=True)
class ModulePrepared:
    x: SparseTensor
    module: torch.nn.Module
    target: SparseTensor | None = None


class SparseClassifier(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.stem = spnn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.norm = spnn.BatchNorm(channels)
        self.act = spnn.ReLU()
        self.pool = spnn.GlobalAvgPool()
        self.head = torch.nn.Linear(channels, max(2, channels // 2))

    def forward(self, x: SparseTensor) -> torch.Tensor:
        return self.head(self.pool(self.act(self.norm(self.stem(x)))))


class SparseResidual(torch.nn.Module):
    def __init__(self, channels: int, *, merge: Literal["add", "cat"]) -> None:
        super().__init__()
        self.left = spnn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.right = spnn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.tail = spnn.Conv3d(
            channels * 2 if merge == "cat" else channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.merge = merge

    def forward(self, x: SparseTensor) -> SparseTensor:
        lhs = self.left(x)
        rhs = self.right(x)
        if self.merge == "cat":
            merged = torch_lattice.cat([lhs, rhs])
        else:
            merged = lhs + rhs
        return self.tail(merged)


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
    return (
        _case(
            "sparse_classifier_module",
            params,
            device,
            lambda c: SparseClassifier(c),
        ),
        _case(
            "sparse_residual_add_module",
            params,
            device,
            lambda c: SparseResidual(c, merge="add"),
        ),
        _case(
            "sparse_residual_cat_module",
            params,
            device,
            lambda c: SparseResidual(c, merge="cat"),
        ),
        _case("activation_chain_module", params, device, _activation_chain),
        _pool_transpose_case(
            "pool_transpose_generated_module", params, device, target=False
        ),
        _pool_transpose_case(
            "pool_transpose_target_module", params, device, target=True
        ),
        _trilinear_upsample_case(
            "trilinear_upsample_generated_module", params, device, target=False
        ),
        _trilinear_upsample_case(
            "trilinear_upsample_target_module", params, device, target=True
        ),
    )


def _case(name, params, device, factory) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group="nn",
        params=params,
        setup=lambda p, device=device, factory=factory: _setup(
            dict(p), device, factory
        ),
        prepare=_prepare,
        run=_run,
        units=("elements",),
    )


def _setup(params: dict[str, object], device: torch.device, factory) -> ModuleFixture:
    base = sparse_fixture(params, device=device)
    module = factory(base.channels).to(device).eval()
    if base.tensor.feats.dtype == torch.float16:
        module = module.half()
    return ModuleFixture(base, module)


def _prepare(fixture: ModuleFixture) -> ModulePrepared:
    return ModulePrepared(fresh_sparse(fixture.base.tensor), fixture.module)


def _run(prepared: ModulePrepared):
    with torch.no_grad():
        if prepared.target is None:
            return prepared.module(prepared.x)
        return prepared.module(prepared.x, prepared.target)


def _pool_transpose_case(
    name: str,
    params,
    device: torch.device,
    *,
    target: bool,
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group="nn",
        params=params,
        setup=lambda p: _setup_pool_transpose(dict(p), device, target=target),
        prepare=lambda fixture: fixture,
        run=_run,
        units=("elements",),
    )


def _setup_pool_transpose(
    params: dict[str, object],
    device: torch.device,
    *,
    target: bool,
) -> ModulePrepared:
    base = sparse_fixture(params, device=device)
    fine = fresh_sparse(base.tensor)
    coarse = spnn.AvgPool3d(kernel_size=2, stride=2)(fine)
    module = spnn.PoolTranspose3d(kernel_size=2, stride=2).to(device).eval()
    return ModulePrepared(coarse, module, fine if target else None)


def _trilinear_upsample_case(
    name: str,
    params,
    device: torch.device,
    *,
    target: bool,
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group="nn",
        params=params,
        setup=lambda p: _setup_trilinear_upsample(dict(p), device, target=target),
        prepare=lambda fixture: fixture,
        run=_run,
        units=("elements",),
    )


def _setup_trilinear_upsample(
    params: dict[str, object],
    device: torch.device,
    *,
    target: bool,
) -> ModulePrepared:
    base = sparse_fixture(params, device=device)
    fine = fresh_sparse(base.tensor)
    coarse = spnn.AvgPool3d(kernel_size=2, stride=2)(fine)
    module = spnn.TrilinearUpsample3d(stride=2).to(device).eval()
    return ModulePrepared(coarse, module, fine if target else None)


def _activation_chain(channels: int) -> torch.nn.Module:
    del channels
    return torch.nn.Sequential(spnn.ReLU(), spnn.SiLU(), spnn.LeakyReLU(), spnn.GELU())
