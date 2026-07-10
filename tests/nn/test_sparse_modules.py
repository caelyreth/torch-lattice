from __future__ import annotations

import pytest
import torch

import torch_lattice
from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F

pytestmark = pytest.mark.nn


def test_group_norm_single_batch_matches_dense_reference():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=torch.int32,
    )
    feats = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tensor = torch_lattice.SparseTensor(
        feats,
        coords,
        spatial_range=(1, 4, 1, 1),
    )
    norm = spnn.GroupNorm(num_groups=2, num_channels=4, affine=True)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 1.5, 2.0, 2.5]))
        norm.bias.copy_(torch.tensor([0.0, 0.25, 0.5, 0.75]))

    out = norm(tensor)
    ref = (
        torch.nn.functional.group_norm(
            feats.t().reshape(1, 4, -1),
            num_groups=2,
            weight=norm.weight,
            bias=norm.bias,
            eps=norm.eps,
        )
        .reshape(4, -1)
        .t()
    )

    assert out.coord_manager is tensor.coord_manager
    assert out.coord_key == tensor.coord_key
    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, ref)


def test_group_norm_multi_batch_matches_per_sample_reference():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    feats = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tensor = torch_lattice.SparseTensor(feats, coords)
    norm = spnn.GroupNorm(num_groups=2, num_channels=4, affine=False)

    out = norm(tensor)
    refs = []
    for batch_id in (0, 1):
        bfeats = feats[coords[:, 0] == batch_id]
        refs.append(
            torch.nn.functional.group_norm(
                bfeats.t().reshape(1, 4, -1),
                num_groups=2,
                weight=None,
                bias=None,
                eps=norm.eps,
            )
            .reshape(4, -1)
            .t()
        )
    ref = torch.cat(refs, dim=0)

    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, ref)


def test_global_pool_single_batch_matches_feature_reduction():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    feats = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]])
    tensor = torch_lattice.SparseTensor(
        feats,
        coords,
        spatial_range=(1, 3, 1, 1),
    )

    torch.testing.assert_close(
        F.global_avg_pool(tensor), feats.mean(dim=0, keepdim=True)
    )
    torch.testing.assert_close(
        F.global_max_pool(tensor), feats.max(dim=0, keepdim=True)[0]
    )


def test_global_pool_multi_batch_matches_per_sample_reduction():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    feats = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 8.0], [7.0, 6.0]])
    tensor = torch_lattice.SparseTensor(feats, coords)

    torch.testing.assert_close(
        F.global_avg_pool(tensor),
        torch.tensor([[2.0, 3.0], [6.0, 7.0]]),
    )
    torch.testing.assert_close(
        F.global_max_pool(tensor),
        torch.tensor([[3.0, 4.0], [7.0, 8.0]]),
    )


def test_global_pool_preserves_declared_empty_batches():
    tensor = torch_lattice.SparseTensor(
        torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        torch.tensor([[0, 0, 0, 0], [2, 0, 0, 0]], dtype=torch.int32),
        spatial_range=(3, 1, 1, 1),
        batch_counts=(1, 0, 1),
    )

    torch.testing.assert_close(
        F.global_sum_pool(tensor),
        torch.tensor([[1.0, 3.0], [0.0, 0.0], [5.0, 7.0]]),
    )
    torch.testing.assert_close(
        F.global_avg_pool(tensor),
        torch.tensor([[1.0, 3.0], [0.0, 0.0], [5.0, 7.0]]),
    )
    with pytest.raises(ValueError, match="empty batches"):
        F.global_max_pool(tensor)
