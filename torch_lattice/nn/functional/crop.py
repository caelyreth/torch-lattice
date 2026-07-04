from typing import Optional, Tuple

import torch

import torch_lattice.backend
from torch_lattice import SparseTensor

__all__ = ["spcrop"]


def spcrop(
    input: SparseTensor,
    coords_min: Optional[Tuple[int, ...]] = None,
    coords_max: Optional[Tuple[int, ...]] = None,
) -> SparseTensor:
    coords, feats, stride = input.coords, input.feats, input.stride
    has_min = coords_min is not None
    has_max = coords_max is not None

    if (
        coords.device.type == "cuda"
        and coords.dtype == torch.int32
        and feats.device == coords.device
        and hasattr(torch_lattice.backend, "sparse_crop_cuda")
        and (has_min or has_max)
    ):
        coords_min_tensor = (
            torch.tensor(coords_min, dtype=torch.int32, device=coords.device)
            if has_min
            else torch.empty((0,), dtype=torch.int32, device=coords.device)
        )
        coords_max_tensor = (
            torch.tensor(coords_max, dtype=torch.int32, device=coords.device)
            if has_max
            else torch.empty((0,), dtype=torch.int32, device=coords.device)
        )
        out_feats, out_coords = torch_lattice.backend.sparse_crop_cuda(
            feats.contiguous(),
            coords.contiguous(),
            coords_min_tensor,
            coords_max_tensor,
            has_min,
            has_max,
        )
        output = SparseTensor(
            coords=out_coords,
            feats=out_feats,
            stride=stride,
            spatial_range=input.spatial_range,
        )
        output._caches = input._caches
        return output

    mask = torch.ones((coords.shape[0], 3), dtype=torch.bool, device=coords.device)
    if coords_min is not None:
        coords_min = torch.tensor(
            coords_min, dtype=torch.int, device=coords.device
        ).unsqueeze(dim=0)
        mask &= coords[:, 1:] >= coords_min
    if coords_max is not None:
        coords_max = torch.tensor(
            coords_max, dtype=torch.int, device=coords.device
        ).unsqueeze(dim=0)
        # Using "<" instead of "<=" is for the backward compatability (in
        # some existing detection codebase). We might need to reflect this
        # in the document or change it back to "<=" in the future.
        mask &= coords[:, 1:] < coords_max

    mask = torch.all(mask, dim=1)
    coords, feats = coords[mask], feats[mask]
    output = SparseTensor(
        coords=coords,
        feats=feats,
        stride=stride,
        spatial_range=input.spatial_range,
    )
    output._caches = input._caches
    return output
