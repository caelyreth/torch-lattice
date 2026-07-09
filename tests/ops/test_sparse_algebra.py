from __future__ import annotations

import pytest
import torch

import torch_lattice
from torch_lattice.nn.functional.hash import sphash
from torch_lattice.operators import generative_add
from tests.cases import algebra_cases
from tests.cases.types import ValueCase

pytestmark = [pytest.mark.ops, pytest.mark.feature]


def _params() -> list[pytest.ParameterSet]:
    return [pytest.param(case, id=case.name, marks=case.marks) for case in algebra_cases.cases()]


@pytest.mark.parametrize('case', _params())
def test_sparse_algebra_value_cases(case: ValueCase) -> None:
    assert case.run() == case.expected


def test_generative_add_shared_coords_uses_sparse_add_fast_path():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    a = torch_lattice.SparseTensor(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        coords,
        spatial_range=(1, 3, 1, 1),
    )
    b = torch_lattice.SparseTensor(
        torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]),
        coords,
        spatial_range=(1, 3, 1, 1),
    )

    out = generative_add(a, b)

    assert out.coords.data_ptr() == coords.data_ptr()
    assert out._caches is a._caches
    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, a.feats + b.feats)


def test_generative_add_equal_cloned_coords_uses_sparse_add_fast_path():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    a = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        coords,
        spatial_range=(1, 3, 1, 1),
    )
    b = torch_lattice.SparseTensor(
        torch.tensor([[4.0], [5.0], [6.0]]),
        coords.clone(),
        spatial_range=(1, 3, 1, 1),
    )

    out = generative_add(a, b)

    assert out.coords.data_ptr() == coords.data_ptr()
    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, a.feats + b.feats)


def test_generative_add_shifted_coords_keeps_union_semantics():
    a = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )
    b = torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0], [30.0]]),
        torch.tensor(
            [[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )

    out = generative_add(a, b)

    expected_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=torch.int32,
    )
    expected_feats = torch.tensor([[1.0], [12.0], [23.0], [30.0]])
    torch.testing.assert_close(out.coords, expected_coords)
    torch.testing.assert_close(out.feats, expected_feats)




def test_sparse_binary_alignment_join_and_fill_semantics():
    lhs = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )
    rhs = torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0], [30.0]]),
        torch.tensor(
            [[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )

    out = torch_lattice.sparse_sub(lhs, rhs, join="left", rhs_fill=1.5)

    expected_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    expected_feats = torch.tensor([[-0.5], [-8.0], [-17.0]])
    torch.testing.assert_close(out.coords, expected_coords)
    torch.testing.assert_close(out.feats, expected_feats)


def test_sparse_cat_outer_aligns_missing_rows_with_zero_features():
    lhs = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 3, 1, 1),
    )
    rhs = torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0]]),
        torch.tensor([[0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 3, 1, 1),
    )

    out = torch_lattice.cat([lhs, rhs], join="outer")

    expected_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    expected_feats = torch.tensor([[1.0, 0.0], [2.0, 10.0], [0.0, 20.0]])
    torch.testing.assert_close(out.coords, expected_coords)
    torch.testing.assert_close(out.feats, expected_feats)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_generative_add_shifted_cuda_matches_reference_by_coordinate():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    feats = torch.arange(12, dtype=torch.float16, device="cuda").reshape(4, 3)
    a = torch_lattice.SparseTensor(feats, coords, spatial_range=(1, 5, 1, 1))
    b = torch_lattice.SparseTensor(
        feats * 0.5,
        coords + torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="cuda"),
        spatial_range=(1, 5, 1, 1),
    )

    out = generative_add(a, b)
    expected_coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0],
            [0, 4, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_feats = torch.stack(
        [
            feats[0],
            feats[1] + feats[0] * 0.5,
            feats[2] + feats[1] * 0.5,
            feats[3] + feats[2] * 0.5,
            feats[3] * 0.5,
        ],
        dim=0,
    )

    out_order = torch.argsort(sphash(out.coords))
    expected_order = torch.argsort(sphash(expected_coords))
    torch.testing.assert_close(out.coords[out_order], expected_coords[expected_order])
    torch.testing.assert_close(out.feats[out_order], expected_feats[expected_order])


