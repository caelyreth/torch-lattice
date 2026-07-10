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


def test_sparse_tensor_decomposes_noncontiguous_batches() -> None:
    tensor = torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0], [30.0]]),
        torch.tensor(
            [[1, 1, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(3, 3, 1, 1),
    )

    coords, features = tensor.decomposed_coordinates_and_features

    assert [part.tolist() for part in tensor.batch_rows] == [[1], [0, 2], []]
    assert [part.tolist() for part in coords] == [
        [[0, 0, 0]],
        [[1, 0, 0], [2, 0, 0]],
        [],
    ]
    assert [part.tolist() for part in features] == [
        [[20.0]],
        [[10.0], [30.0]],
        [],
    ]


def test_sparse_pruning_keeps_order_and_gradients(
    selected_device: torch.device,
) -> None:
    feats = torch.tensor(
        [[1.0], [2.0], [3.0]],
        device=selected_device,
        requires_grad=True,
    )
    tensor = torch_lattice.SparseTensor(
        feats,
        torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0]],
            dtype=torch.int32,
            device=selected_device,
        ),
        spatial_range=(2, 3, 1, 1),
    )

    indexed = torch_lattice.prune(
        tensor,
        torch.tensor([2, 0], dtype=torch.int64, device=selected_device),
    )
    masked = torch_lattice.prune_mask(
        tensor,
        torch.tensor([True, False, True], device=selected_device),
    )

    assert indexed.coords.tolist() == [[1, 2, 0, 0], [0, 0, 0, 0]]
    assert masked.coords.tolist() == [[0, 0, 0, 0], [1, 2, 0, 0]]
    indexed.feats.sum().backward()
    torch.testing.assert_close(
        feats.grad,
        torch.tensor([[1.0], [0.0], [1.0]], device=selected_device),
    )
