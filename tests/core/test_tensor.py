from __future__ import annotations

import pytest
import torch

import torch_lattice
from tests.support import line_sparse_case

pytestmark = pytest.mark.core


def test_sparse_tensor_exposes_coordinate_and_feature_aliases() -> None:
    tensor = line_sparse_case(channels=2).tensor()

    assert tensor.F is tensor.feats
    assert tensor.C is tensor.coords
    assert tensor.feats.shape == (3, 2)
    assert tensor.coords.dtype == torch.int32
    assert tensor.spatial_range == (1, 3, 1, 1)


def test_sparse_tensor_to_moves_features_and_coords_together() -> None:
    tensor = line_sparse_case(channels=1).tensor()

    out = tensor.to(torch.device('cpu'))

    assert out.feats.device.type == 'cpu'
    assert out.coords.device.type == 'cpu'
    torch.testing.assert_close(out.feats, tensor.feats)
    torch.testing.assert_close(out.coords, tensor.coords)


def test_sparse_tensor_half_only_converts_features() -> None:
    tensor = line_sparse_case(channels=1).tensor()

    out = tensor.half()

    assert out.feats.dtype == torch.float16
    assert out.coords.dtype == torch.int32
    torch.testing.assert_close(out.coords, tensor.coords)


def test_sparse_tensor_add_uses_coordinate_aligned_sparse_binary() -> None:
    lhs = line_sparse_case(channels=1).tensor()
    rhs = torch_lattice.SparseTensor(lhs.feats * 2, lhs.coords.clone(), spatial_range=lhs.spatial_range)

    out = lhs + rhs

    torch.testing.assert_close(out.coords, lhs.coords)
    torch.testing.assert_close(out.feats, lhs.feats * 3)
