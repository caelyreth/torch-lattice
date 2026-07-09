from __future__ import annotations

from dataclasses import dataclass

import torch

import torch_lattice


@dataclass(frozen=True)
class SparseCase:
    feats: torch.Tensor
    coords: torch.Tensor
    spatial_range: tuple[int, int, int, int]

    def tensor(self, *, device: torch.device | str | None = None) -> torch_lattice.SparseTensor:
        feats = self.feats if device is None else self.feats.to(device)
        coords = self.coords if device is None else self.coords.to(device)
        return torch_lattice.SparseTensor(feats, coords, spatial_range=self.spatial_range)


def line_sparse_case(channels: int = 1, *, rows: int = 3, dtype: torch.dtype = torch.float32) -> SparseCase:
    coords = torch.tensor([[0, row, 0, 0] for row in range(rows)], dtype=torch.int32)
    values = torch.arange(rows * channels, dtype=dtype).reshape(rows, channels) + 1
    return SparseCase(values, coords, (1, rows, 1, 1))


def two_batch_sparse_case(dtype: torch.dtype = torch.float32) -> SparseCase:
    return SparseCase(
        torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 8.0], [7.0, 6.0]], dtype=dtype),
        torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.int32),
        (2, 2, 1, 1),
    )


def assert_sparse_close(actual: torch_lattice.SparseTensor, expected: torch_lattice.SparseTensor, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    torch.testing.assert_close(actual.coords.cpu(), expected.coords.cpu())
    torch.testing.assert_close(actual.feats.cpu(), expected.feats.cpu(), rtol=rtol, atol=atol)


def assert_same_sparse_coords(actual: torch_lattice.SparseTensor, expected: torch_lattice.SparseTensor) -> None:
    torch.testing.assert_close(actual.coords.cpu(), expected.coords.cpu())
    assert actual.spatial_range == expected.spatial_range
