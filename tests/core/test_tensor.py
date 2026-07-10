from __future__ import annotations

import pytest
import torch

import torch_lattice
from tests.support import line_sparse_case

pytestmark = pytest.mark.core


def test_sparse_tensor_exposes_explicit_coordinate_and_feature_surface() -> None:
    tensor = line_sparse_case(channels=2).tensor()

    assert tensor.feats.shape == (3, 2)
    assert tensor.coords.dtype == torch.int32
    assert tensor.spatial_range == (1, 3, 1, 1)
    assert tensor.coord_manager.get(tensor.coord_key).coords is tensor.coords


def test_sparse_tensor_to_moves_features_and_coords_together() -> None:
    tensor = line_sparse_case(channels=1).tensor()

    out = tensor.to(torch.device("cpu"))

    assert out.feats.device.type == "cpu"
    assert out.coords.device.type == "cpu"
    torch.testing.assert_close(out.feats, tensor.feats)
    torch.testing.assert_close(out.coords, tensor.coords)


def test_sparse_tensor_half_only_converts_features() -> None:
    tensor = line_sparse_case(channels=1).tensor()

    out = tensor.half()

    assert out.feats.dtype == torch.float16
    assert out.coords.dtype == torch.int32
    torch.testing.assert_close(out.coords, tensor.coords)
    assert out.coord_manager is tensor.coord_manager
    assert out.coord_key == tensor.coord_key


def test_sparse_tensor_row_change_creates_coordinate_identity() -> None:
    tensor = line_sparse_case(channels=1).tensor()

    out = tensor.with_coordinates(
        feats=tensor.feats[:2],
        coords=tensor.coords[:2].clone(),
        spatial_range=None,
    )

    assert out.coord_manager is tensor.coord_manager
    assert out.coord_key != tensor.coord_key
    assert out.spatial_range is None


def test_sparse_tensor_add_uses_coordinate_aligned_sparse_binary() -> None:
    lhs = line_sparse_case(channels=1).tensor()
    rhs = torch_lattice.SparseTensor(
        lhs.feats * 2, lhs.coords.clone(), spatial_range=lhs.spatial_range
    )

    out = lhs + rhs

    torch.testing.assert_close(out.coords, lhs.coords)
    torch.testing.assert_close(out.feats, lhs.feats * 3)
