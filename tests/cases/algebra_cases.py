from __future__ import annotations

import torch

import torch_lattice
from tests.cases.types import ValueCase


def cases() -> list[ValueCase]:
    return [
        ValueCase('sparse_add_inner', _sparse_add_inner, ([[0, 1, 0, 0], [0, 2, 0, 0]], [[12.0], [23.0]])),
        ValueCase('sparse_sub_left_fill', _sparse_sub_left_fill, ([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], [[-0.5], [-8.0], [-17.0]])),
        ValueCase('sparse_cat_outer', _sparse_cat_outer, ([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]], [[1.0, 0.0], [2.0, 10.0], [3.0, 20.0], [0.0, 30.0]])),
    ]


def _lhs() -> torch_lattice.SparseTensor:
    return torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 4, 1, 1),
    )


def _rhs() -> torch_lattice.SparseTensor:
    return torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0], [30.0]]),
        torch.tensor([[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 4, 1, 1),
    )


def _sparse_add_inner() -> tuple[list[list[int]], list[list[float]]]:
    out = torch_lattice.sparse_add(_lhs(), _rhs(), join='inner')
    return out.coords.tolist(), out.feats.tolist()


def _sparse_sub_left_fill() -> tuple[list[list[int]], list[list[float]]]:
    out = torch_lattice.sparse_sub(_lhs(), _rhs(), join='left', rhs_fill=1.5)
    return out.coords.tolist(), out.feats.tolist()


def _sparse_cat_outer() -> tuple[list[list[int]], list[list[float]]]:
    out = torch_lattice.cat([_lhs(), _rhs()], join='outer')
    return out.coords.tolist(), out.feats.tolist()
