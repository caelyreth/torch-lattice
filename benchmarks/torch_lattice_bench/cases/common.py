from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F

from torch_lattice_bench.datasets import SparseFixture, clone_sparse, fresh_sparse, params_matrix, sparse_fixture
from torch_lattice_bench.harness import BenchmarkCase


def sparse_cases(
    preset: str,
    *,
    group: str,
    specs: tuple[tuple[str, Callable[[SparseFixture], Any], tuple[str, ...], tuple[str, ...] | None], ...],
    n_values: tuple[int, ...] | None,
    channels: tuple[int, ...] | None,
    layouts: tuple[str, ...] | None,
    dtype: str,
    device: torch.device,
) -> tuple[BenchmarkCase, ...]:
    params = params_matrix(preset, n_values=n_values, channels=channels, layouts=layouts, dtype=dtype)
    return tuple(
        BenchmarkCase(
            name=name,
            group=group,
            params=params,
            setup=lambda p, device=device: sparse_fixture(dict(p), device=device),
            prepare=lambda fixture: fixture,
            run=run,
            units=units,
            modes=modes,
        )
        for name, run, units, modes in specs
    )


def set_conv_config(
    dataflow: F.Dataflow,
    *,
    ifsort: bool = False,
    split_mask_num: int = 1,
    wgrad_split_k: int | str = 'auto',
    FOD_fusion: bool = True,
    IGEMM_center_only: bool = False,
) -> None:
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = dataflow
    config.kmap_mode = 'hashmap_on_the_fly'
    config.ifsort = ifsort
    config.split_mask_num = split_mask_num
    config.split_mask_num_bwd = 3
    config.wgrad_split_k = wgrad_split_k
    config.FOD_fusion = FOD_fusion
    config.IGEMM_center_only = IGEMM_center_only
    F.conv_config.set_global_conv_config(config)


def conv_module(channels: int, dtype: torch.dtype, device: torch.device, kernel_size: int, stride: int = 1) -> spnn.Conv3d:
    module = spnn.Conv3d(channels, channels, kernel_size=kernel_size, stride=stride, bias=False).to(device)
    if dtype == torch.float16:
        module = module.half()
    module.eval()
    return module


def module_dtype(module: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    if dtype == torch.float16:
        return module.half()
    return module


def shifted_sparse(x: SparseTensor) -> SparseTensor:
    shifted = SparseTensor(x.feats * 0.25, x.coords.clone(), x.stride, x.spatial_range)
    shifted.coords[:, 1] += 1
    return shifted


__all__ = [
    'F',
    'SparseFixture',
    'clone_sparse',
    'conv_module',
    'fresh_sparse',
    'module_dtype',
    'set_conv_config',
    'shifted_sparse',
    'sparse_cases',
    'spnn',
    'torch_lattice',
]
