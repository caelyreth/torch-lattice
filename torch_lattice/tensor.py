from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from torch_lattice.core import CoordinateManager, CoordinateMapKey
from torch_lattice.utils import make_ntuple, to_dense

__all__ = ["SparseTensor"]

Triple = tuple[int, int, int]
_UNSET = object()


class SparseTensor:
    """Sparse feature value with explicit coordinate-support identity.

    Coordinates have shape ``(N, 4)`` and use ``(batch, x, y, z)`` order.
    Features have shape ``(N, C)`` and share their row order with coordinates.
    Coordinate managers own support identity and cached sparse relations;
    feature-only transformations preserve that identity, while row-changing
    operations create a new coordinate key.
    """

    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: int | Sequence[int] = 1,
        spatial_range: int | Sequence[int] | None = None,
        *,
        batch_counts: Sequence[int] | None = None,
        coord_manager: CoordinateManager | None = None,
        coord_key: CoordinateMapKey | None = None,
    ) -> None:
        normalized_stride = _triple(stride, name="stride")
        normalized_range = _spatial_range(spatial_range)
        normalized_counts = _batch_counts(
            batch_counts,
            rows=int(coords.shape[0]),
            spatial_range=normalized_range,
        )
        _validate_sparse_components(feats, coords)

        manager = coord_manager or CoordinateManager()
        if coord_key is None:
            key = manager.insert(
                coords,
                normalized_stride,
                spatial_range=normalized_range,
                batch_counts=normalized_counts,
            )
            owned_coords = coords
        else:
            coordinate_map = manager.get(coord_key)
            if coord_key.stride != normalized_stride:
                raise ValueError("coordinate key stride does not match tensor stride")
            if coordinate_map.coords is not coords:
                raise ValueError(
                    "coords must be the manager-owned tensor for coord_key"
                )
            if coordinate_map.spatial_range != normalized_range:
                raise ValueError("spatial_range does not match the coordinate map")
            if coordinate_map.batch_counts != normalized_counts:
                raise ValueError("batch_counts does not match the coordinate map")
            key = coord_key
            owned_coords = coordinate_map.coords

        self.feats = feats
        self.coords = owned_coords
        self.stride = normalized_stride
        self.spatial_range = normalized_range
        self.batch_counts = normalized_counts
        self.coord_manager = manager
        self.coord_key = key

    def replace(self, *, feats: torch.Tensor) -> SparseTensor:
        """Return a feature replacement on the same coordinate support."""

        return SparseTensor(
            feats,
            self.coords,
            self.stride,
            self.spatial_range,
            batch_counts=self.batch_counts,
            coord_manager=self.coord_manager,
            coord_key=self.coord_key,
        )

    def with_coordinates(
        self,
        *,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: int | Sequence[int] | None = None,
        spatial_range: int | Sequence[int] | None | object = _UNSET,
        batch_counts: Sequence[int] | None = None,
    ) -> SparseTensor:
        """Return a value on newly created coordinate support."""

        next_range = (
            self.spatial_range
            if spatial_range is _UNSET
            else cast(int | Sequence[int] | None, spatial_range)
        )
        return SparseTensor(
            feats,
            coords,
            self.stride if stride is None else stride,
            next_range,
            batch_counts=batch_counts,
            coord_manager=self.coord_manager,
        )

    def cpu(self) -> SparseTensor:
        return self.to("cpu")

    def cuda(self, device: torch.device | int | None = None) -> SparseTensor:
        target = torch.device(
            "cuda"
            if device is None
            else f"cuda:{device}"
            if isinstance(device, int)
            else device
        )
        return self.to(target)

    def half(self) -> SparseTensor:
        return self.replace(feats=self.feats.half())

    def detach(self) -> SparseTensor:
        return self.replace(feats=self.feats.detach())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> SparseTensor:
        target = torch.device(device)
        if self.coords.device == target and self.feats.device == target:
            return self
        return SparseTensor(
            self.feats.to(target, non_blocking=non_blocking),
            self.coords.to(target, non_blocking=non_blocking),
            self.stride,
            self.spatial_range,
            batch_counts=self.batch_counts,
        )

    def dense(self) -> torch.Tensor:
        if self.spatial_range is None:
            raise ValueError("dense conversion requires spatial_range")
        return to_dense(self.feats, self.coords, self.spatial_range)

    def __add__(self, other: SparseTensor) -> SparseTensor:
        from torch_lattice.operators import sparse_add

        return sparse_add(self, other)

    def __sub__(self, other: SparseTensor) -> SparseTensor:
        from torch_lattice.operators import sparse_sub

        return sparse_sub(self, other)

    def __mul__(self, other: SparseTensor) -> SparseTensor:
        from torch_lattice.operators import sparse_mul

        return sparse_mul(self, other)


def _validate_sparse_components(feats: torch.Tensor, coords: torch.Tensor) -> None:
    if not isinstance(feats, torch.Tensor) or not isinstance(coords, torch.Tensor):
        raise TypeError("feats and coords must be torch.Tensor values")
    if feats.ndim != 2:
        raise ValueError("feats must have shape (N, C)")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape (N, 4)")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError("coords and feats must have the same row count")
    if coords.dtype not in (torch.int32, torch.int64):
        raise TypeError("coords must use int32 or int64 dtype")
    if coords.device != feats.device:
        raise ValueError("coords and feats must be on the same device")


def _triple(value: int | Sequence[int], *, name: str) -> Triple:
    result = tuple(int(item) for item in make_ntuple(value, ndim=3))
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _spatial_range(
    value: int | Sequence[int] | None,
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        values = (value,) * 4
    else:
        values = tuple(int(item) for item in value)
    if len(values) != 4:
        raise ValueError("spatial_range must have shape (batch, x, y, z)")
    if any(item < 0 for item in values):
        raise ValueError("spatial_range values must be non-negative")
    return values


def _batch_counts(
    value: Sequence[int] | None,
    *,
    rows: int,
    spatial_range: tuple[int, int, int, int] | None,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    counts = tuple(int(item) for item in value)
    if any(item < 0 for item in counts) or sum(counts) != rows:
        raise ValueError("batch_counts must be non-negative and sum to N")
    if spatial_range is not None and len(counts) != spatial_range[0]:
        raise ValueError("batch_counts length must match spatial_range batch size")
    return counts
