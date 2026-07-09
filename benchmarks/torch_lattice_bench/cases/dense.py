from __future__ import annotations

import math

import torch

from torch_lattice.nn.functional.devoxelize import calc_ti_weights, spdevoxelize
from torch_lattice.nn.functional.voxelize import spvoxelize
from torch_lattice.utils import to_dense

from torch_lattice_bench.cases.common import F, SparseFixture, sparse_cases
from torch_lattice_bench.harness import BenchmarkCase, SkipCase

MAX_DENSE_ELEMENTS = 256_000_000


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None,
    channels: tuple[int, ...] | None,
    layouts: tuple[str, ...] | None,
    dtype: str,
    device,
) -> tuple[BenchmarkCase, ...]:
    return sparse_cases(
        preset,
        group='dense',
        specs=(
            ('to_dense_forward', _to_dense, ('elements',), None),
            ('spvoxelize_forward', _spvoxelize, ('points',), None),
            ('spdevoxelize_forward', _spdevoxelize, ('points',), None),
            ('calc_ti_weights', _calc_ti_weights, ('points',), None),
        ),
        n_values=n_values,
        channels=channels,
        layouts=layouts,
        dtype=dtype,
        device=device,
    )


def _to_dense(fixture: SparseFixture) -> torch.Tensor:
    x = fixture.tensor
    dense_elements = math.prod(x.spatial_range) * x.feats.shape[1]
    if dense_elements > MAX_DENSE_ELEMENTS:
        raise SkipCase(f'dense_elements={dense_elements} exceeds {MAX_DENSE_ELEMENTS}')
    return to_dense(x.feats, x.coords, x.spatial_range)


def _spvoxelize(fixture: SparseFixture) -> torch.Tensor:
    x = fixture.tensor
    bins = max(1, x.feats.shape[0] // 4)
    voxel_idx = torch.arange(x.feats.shape[0], device=x.feats.device, dtype=torch.int32) % bins
    counts = F.spcount(voxel_idx, bins)
    return spvoxelize(x.feats, voxel_idx, counts)


def _spdevoxelize(fixture: SparseFixture) -> torch.Tensor:
    x = fixture.tensor
    rows = x.feats.shape[0]
    idx = torch.randint(0, rows, (rows, 8), dtype=torch.int32, device=x.feats.device)
    weights = torch.rand(rows, 8, dtype=x.feats.dtype, device=x.feats.device)
    weights /= weights.sum(dim=1, keepdim=True)
    return spdevoxelize(x.feats, idx, weights)


def _calc_ti_weights(fixture: SparseFixture) -> torch.Tensor:
    x = fixture.tensor
    rows = x.feats.shape[0]
    idx = torch.randint(0, rows, (rows, 8), dtype=torch.int32, device=x.feats.device)
    return calc_ti_weights(x.coords[:, 1:].float() + 0.25, idx)
