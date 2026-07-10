from __future__ import annotations

from itertools import product

import torch

from torch_lattice.nn.functional.conv.kmap.layout import set_neighbor_pairs
from torch_lattice.utils import make_ntuple

Triple = tuple[int, int, int]

__all__ = [
    "build_pool_output_coords",
    "build_target_out_in_map",
    "gather_scatter_kmap_from_out_in_map",
]


def build_pool_output_coords(
    coords: torch.Tensor,
    *,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    spatial_range=None,
) -> torch.Tensor:
    """Generate convolution-style output support entirely on the input device."""

    _validate_coords(coords)
    if coords.shape[0] == 0:
        return coords.clone()
    size = _triple(kernel_size)
    step = _triple(stride)
    pad = _triple(padding)
    spacing = _triple(dilation)
    device = coords.device
    spatial = coords[:, 1:].to(torch.int64)
    batch = coords[:, :1].to(torch.int64)
    offsets = _kernel_offsets(size, device=device)
    numerator = (
        spatial[:, None, :]
        + _tensor(pad, device)[None, None, :]
        - offsets[None, :, :] * _tensor(spacing, device)[None, None, :]
    )
    step_tensor = _tensor(step, device)[None, None, :]
    valid = torch.all(torch.remainder(numerator, step_tensor) == 0, dim=2)
    output_spatial = torch.div(numerator, step_tensor, rounding_mode="floor")
    valid &= torch.all(output_spatial >= 0, dim=2)
    if spatial_range is not None:
        limit = _output_spatial_range(
            tuple(int(item) for item in spatial_range[1:]),
            size,
            step,
            pad,
            spacing,
        )
        valid &= torch.all(
            output_spatial < _tensor(limit, device)[None, None, :], dim=2
        )
    batch = batch[:, None, :].expand(-1, offsets.shape[0], -1)
    candidates = torch.cat((batch, output_spatial), dim=2)[valid]
    if candidates.shape[0] == 0:
        return coords.new_empty((0, 4))
    return torch.unique(candidates.to(coords.dtype), dim=0)


def build_target_out_in_map(
    input_coords: torch.Tensor,
    target_coords: torch.Tensor,
    *,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    """Build ``(N_target, kernel_volume)`` target-to-input row indices."""

    _validate_coords(input_coords)
    _validate_coords(target_coords)
    if input_coords.device != target_coords.device:
        raise ValueError("input and target coordinates must use the same device")
    size = _triple(kernel_size)
    step = _triple(stride)
    pad = _triple(padding)
    spacing = _triple(dilation)
    target_rows = int(target_coords.shape[0])
    kernel_volume = _volume(size)
    if target_rows == 0 or input_coords.shape[0] == 0:
        return torch.full(
            (target_rows, kernel_volume),
            -1,
            dtype=torch.int64,
            device=target_coords.device,
        )

    device = target_coords.device
    target = target_coords.to(torch.int64)
    offsets = _kernel_offsets(size, device=device)
    source_spatial = (
        target[:, None, 1:] * _tensor(step, device)[None, None, :]
        + offsets[None, :, :] * _tensor(spacing, device)[None, None, :]
        - _tensor(pad, device)[None, None, :]
    )
    batch = target[:, None, :1].expand(-1, kernel_volume, -1)
    candidates = _int32_coords(
        torch.cat((batch, source_spatial), dim=2),
        name="target convolution candidates",
    )

    from torch_lattice.nn.functional.hash import sphash
    from torch_lattice.nn.functional.query import sphashquery

    references = _int32_coords(input_coords, name="input coordinates")
    query_hashes = sphash(candidates.reshape(-1, 4))
    reference_hashes = sphash(references)
    return (
        sphashquery(query_hashes, reference_hashes)
        .reshape(target_rows, kernel_volume)
        .to(torch.int64)
    )


def gather_scatter_kmap_from_out_in_map(
    out_in_map: torch.Tensor,
    *,
    input_size: int,
) -> dict:
    """Convert a target relation to the gather/scatter kernel-map ABI."""

    relation = out_in_map.to(torch.long)
    transposed = relation.t().contiguous()
    nbsizes = torch.sum(transposed != -1, dim=1).to(torch.int32)
    nbmaps = torch.nonzero(transposed != -1, as_tuple=False)
    if nbmaps.numel() == 0:
        nbmaps = relation.new_empty((0, 2), dtype=torch.int64)
    else:
        flat = transposed.reshape(-1)
        nbmaps[:, 0] = flat[nbmaps[:, 0] * transposed.size(1) + nbmaps[:, 1]]
    output_size = int(relation.shape[0])
    kmap = {
        "out_in_map": relation,
        "nbsizes": nbsizes,
        "nbsizes_cpu": nbsizes.cpu().contiguous(),
        "sizes": (int(input_size), output_size),
    }
    nbmaps = set_neighbor_pairs(kmap, nbmaps)
    input_mask = torch.empty(0, dtype=torch.int32, device=relation.device)
    output_mask = torch.empty(0, dtype=torch.int32, device=relation.device)
    if relation.device.type == "cuda" and nbmaps.numel() > 0:
        import torch_lattice.backend

        input_mask, output_mask = torch_lattice.backend.build_mask_from_kmap(
            int(input_size),
            output_size,
            nbmaps,
            nbsizes.int(),
        )
    kmap["input_mask"] = input_mask
    kmap["output_mask"] = output_mask
    return kmap


def _kernel_offsets(kernel_size: Triple, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        list(product(*(range(item) for item in kernel_size))),
        dtype=torch.int64,
        device=device,
    )


def _output_spatial_range(
    spatial_range: Triple,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> Triple:
    return tuple(
        max(
            0,
            (
                spatial_range[index]
                + 2 * padding[index]
                - dilation[index] * (kernel_size[index] - 1)
                - 1
            )
            // stride[index]
            + 1,
        )
        for index in range(3)
    )


def _validate_coords(coords: torch.Tensor) -> None:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coordinates must have shape (N, 4)")
    if coords.dtype not in (torch.int32, torch.int64):
        raise TypeError("coordinates must use int32 or int64 dtype")


def _int32_coords(coords: torch.Tensor, *, name: str) -> torch.Tensor:
    if coords.dtype == torch.int32:
        return coords.contiguous()
    bounds = torch.iinfo(torch.int32)
    if torch.any(coords < bounds.min) or torch.any(coords > bounds.max):
        raise OverflowError(f"{name} exceed the int32 native hash range")
    return coords.to(torch.int32).contiguous()


def _tensor(values: Triple, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def _triple(value) -> Triple:
    return tuple(int(item) for item in make_ntuple(value, ndim=3))


def _volume(value: Triple) -> int:
    return value[0] * value[1] * value[2]
