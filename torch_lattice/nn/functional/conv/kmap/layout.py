from __future__ import annotations

from typing import Any

import torch


def set_neighbor_pairs(kmap: dict[str, Any], pairs: torch.Tensor) -> torch.Tensor:
    """Store the canonical pair-major native relation layout."""

    pairs = pairs.to(dtype=torch.int32).contiguous()
    _validate_map(pairs, name="neighbor_pairs", shape=(None, 2))
    kmap["neighbor_pairs"] = pairs
    return pairs


def set_fod_neighbor_maps(
    kmap: dict[str, Any],
    pairs: torch.Tensor,
    fod_map: torch.Tensor | None = None,
) -> None:
    """Store canonical pairs and the plane-major Fetch-on-Demand view."""

    pairs = set_neighbor_pairs(kmap, pairs)
    if fod_map is None:
        fod_map = pairs.t().contiguous()
    else:
        fod_map = fod_map.to(dtype=torch.int32).contiguous()
    _validate_map(fod_map, name="fod_neighbor_map", shape=(2, pairs.shape[0]))
    kmap["fod_neighbor_map"] = fod_map


def neighbor_pairs(kmap: dict[str, Any]) -> torch.Tensor:
    pairs = kmap["neighbor_pairs"]
    _validate_map(pairs, name="neighbor_pairs", shape=(None, 2))
    return pairs


def fod_neighbor_map(kmap: dict[str, Any]) -> torch.Tensor:
    fod_map = kmap["fod_neighbor_map"]
    _validate_map(fod_map, name="fod_neighbor_map", shape=(2, None))
    return fod_map


def _validate_map(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int | None, int | None],
) -> None:
    if value.dtype != torch.int32:
        raise TypeError(f"{name} must use int32 dtype, got {value.dtype}")
    if value.ndim != 2 or any(
        expected is not None and value.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        expected = tuple("M" if item is None else item for item in shape)
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
