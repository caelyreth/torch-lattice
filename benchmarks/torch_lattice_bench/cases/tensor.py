from __future__ import annotations

import torch

from torch_lattice import SparseTensor
from torch_lattice.operators import (
    cat,
    generative_add,
    prune_mask,
    reindex_sparse,
)

from torch_lattice_bench.cases.common import (
    F,
    SparseFixture,
    clone_sparse,
    shifted_sparse,
    sparse_cases,
    spnn,
)
from torch_lattice_bench.harness import BenchmarkCase


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
        group="tensor",
        specs=(
            ("sparse_tensor_construct", _construct, ("n_in",), None),
            ("sparse_tensor_to_device_noop", _to_device, ("n_in",), None),
            ("sparse_tensor_half", _half, ("elements",), None),
            ("cat_features", _cat_features, ("elements",), None),
            (
                "generative_add_overlap",
                _generative_add_overlap,
                ("elements",),
                None,
            ),
            (
                "generative_add_shifted",
                _generative_add_shifted,
                ("elements",),
                None,
            ),
            ("sparse_reindex", _reindex, ("elements",), None),
            ("prune_mask", _prune_mask, ("n_in",), None),
            (
                "global_avg_pool",
                lambda f: F.global_avg_pool(f.tensor),
                ("n_in",),
                None,
            ),
            (
                "global_max_pool",
                lambda f: F.global_max_pool(f.tensor),
                ("n_in",),
                None,
            ),
            ("crop_center_half", _crop_center_half, ("n_in",), None),
            (
                "relu",
                lambda f: F.relu(clone_sparse(f.tensor), inplace=False),
                ("elements",),
                None,
            ),
            (
                "silu",
                lambda f: F.silu(clone_sparse(f.tensor), inplace=False),
                ("elements",),
                None,
            ),
            (
                "leaky_relu",
                lambda f: F.leaky_relu(clone_sparse(f.tensor), inplace=False),
                ("elements",),
                None,
            ),
            ("batch_norm_module", _batch_norm, ("elements",), None),
            ("group_norm_module", _group_norm, ("elements",), None),
        ),
        n_values=n_values,
        channels=channels,
        layouts=layouts,
        dtype=dtype,
        device=device,
    )


def _construct(fixture: SparseFixture) -> SparseTensor:
    x = fixture.tensor
    return SparseTensor(x.feats, x.coords, x.stride, x.spatial_range)


def _to_device(fixture: SparseFixture) -> SparseTensor:
    return fixture.tensor.to(fixture.tensor.feats.device)


def _half(fixture: SparseFixture) -> SparseTensor:
    return clone_sparse(fixture.tensor).half()


def _cat_features(fixture: SparseFixture) -> SparseTensor:
    x = fixture.tensor
    twin = SparseTensor(x.feats * 0.5, x.coords.clone(), x.stride, x.spatial_range)
    return cat([x, twin])


def _generative_add_overlap(fixture: SparseFixture) -> SparseTensor:
    x = fixture.tensor
    twin = SparseTensor(x.feats * 0.5, x.coords.clone(), x.stride, x.spatial_range)
    return generative_add(x, twin)


def _generative_add_shifted(fixture: SparseFixture) -> SparseTensor:
    return generative_add(fixture.tensor, shifted_sparse(fixture.tensor))


def _reindex(fixture: SparseFixture) -> SparseTensor:
    return reindex_sparse(fixture.tensor, shifted_sparse(fixture.tensor))


def _prune_mask(fixture: SparseFixture) -> SparseTensor:
    rows = torch.arange(
        fixture.tensor.coords.shape[0], device=fixture.tensor.coords.device
    )
    return prune_mask(fixture.tensor, rows.remainder(2) == 0)


def _crop_center_half(fixture: SparseFixture) -> SparseTensor:
    x = fixture.tensor
    coords_max = tuple(
        max(1, int(x.coords[:, index].max().item()) // 2) for index in range(1, 4)
    )
    return F.spcrop(x, coords_min=(0, 0, 0), coords_max=coords_max)


def _batch_norm(fixture: SparseFixture) -> SparseTensor:
    x = fixture.tensor
    return (
        spnn.BatchNorm(x.feats.shape[1])
        .to(x.feats.device, dtype=x.feats.dtype)
        .eval()(x)
    )


def _group_norm(fixture: SparseFixture) -> SparseTensor:
    x = fixture.tensor
    groups = max(1, min(8, int(x.feats.shape[1])))
    return (
        spnn.GroupNorm(num_groups=groups, num_channels=x.feats.shape[1])
        .to(x.feats.device, dtype=x.feats.dtype)
        .eval()(x)
    )
