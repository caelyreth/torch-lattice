from __future__ import annotations

from itertools import product
from typing import Iterable, Tuple

import torch

from torch_lattice.utils import make_ntuple

Triple = Tuple[int, int, int]

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
    """Return the generated sparse output support for local pooling.

    The relation follows the same coordinate equation as sparse convolution:
    ``input = output * stride + kernel_offset * dilation - padding``.
    Output coordinates are the unique coordinates that receive at least one
    input row.
    """

    if coords.numel() == 0:
        return coords.clone()

    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    device = coords.device
    outputs = []
    spatial_limit = None if spatial_range is None else tuple(int(v) for v in spatial_range[1:])
    output_limit = _output_spatial_range(spatial_limit, kernel_size, stride, padding, dilation)

    spatial = coords[:, 1:].to(torch.long)
    batch = coords[:, :1].to(torch.long)
    for offset in _kernel_offsets(kernel_size, device=device):
        numerator = spatial + _tensor(padding, device) - offset * _tensor(dilation, device)
        stride_tensor = _tensor(stride, device)
        valid = torch.all(torch.remainder(numerator, stride_tensor) == 0, dim=1)
        out_spatial = torch.div(numerator, stride_tensor, rounding_mode="floor")
        valid &= torch.all(out_spatial >= 0, dim=1)
        if output_limit is not None:
            valid &= torch.all(out_spatial < _tensor(output_limit, device), dim=1)
        if torch.any(valid):
            outputs.append(torch.cat([batch[valid], out_spatial[valid]], dim=1))

    if not outputs:
        return coords.new_empty((0, coords.shape[1]))
    return torch.unique(torch.cat(outputs, dim=0).to(coords.dtype), dim=0)


def build_target_out_in_map(
    input_coords: torch.Tensor,
    target_coords: torch.Tensor,
    *,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    """Build a dense ``(N_target, kernel_volume)`` target-to-input relation."""

    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    device = target_coords.device
    out_in_map = torch.full(
        (target_coords.shape[0], _volume(kernel_size)),
        -1,
        dtype=torch.int64,
        device=device,
    )
    if input_coords.numel() == 0 or target_coords.numel() == 0:
        return out_in_map

    lookup = {
        tuple(int(item) for item in row): index
        for index, row in enumerate(input_coords.detach().cpu().tolist())
    }
    target_cpu = target_coords.detach().cpu().to(torch.long)
    stride_cpu = torch.tensor(stride, dtype=torch.long)
    padding_cpu = torch.tensor(padding, dtype=torch.long)
    dilation_cpu = torch.tensor(dilation, dtype=torch.long)
    values = out_in_map.cpu()

    for kernel_index, offset in enumerate(_kernel_offsets(kernel_size, device=torch.device("cpu"))):
        source_spatial = (
            target_cpu[:, 1:] * stride_cpu
            + offset.to(torch.long) * dilation_cpu
            - padding_cpu
        )
        source = torch.cat([target_cpu[:, :1], source_spatial], dim=1)
        for row_index, coord in enumerate(source.tolist()):
            input_index = lookup.get(tuple(int(item) for item in coord))
            if input_index is not None:
                values[row_index, kernel_index] = int(input_index)
    return values.to(device=device)


def gather_scatter_kmap_from_out_in_map(
    out_in_map: torch.Tensor,
    *,
    input_size: int,
) -> dict:
    """Convert an ``out_in_map`` relation to gather/scatter convolution kmap fields."""

    relation = out_in_map.to(torch.long)
    transposed = relation.t().contiguous()
    nbsizes = torch.sum(transposed != -1, dim=1).to(torch.int32)
    nbmaps = torch.nonzero(transposed != -1, as_tuple=False)
    if nbmaps.numel() == 0:
        nbmaps = relation.new_empty((0, 2), dtype=torch.int64)
    else:
        flat = transposed.reshape(-1)
        nbmaps[:, 0] = flat[nbmaps[:, 0] * transposed.size(1) + nbmaps[:, 1]]
    nbmaps = nbmaps.contiguous()
    output_size = int(relation.shape[0])
    input_mask = torch.empty(0, dtype=torch.int32, device=relation.device)
    output_mask = torch.empty(0, dtype=torch.int32, device=relation.device)
    if relation.device.type == "cuda" and nbmaps.numel() > 0:
        try:
            import torch_lattice.backend

            input_mask, output_mask = torch_lattice.backend.build_mask_from_kmap(
                int(input_size),
                output_size,
                nbmaps.int(),
                nbsizes[:output_size].int(),
            )
        except Exception:
            input_mask = torch.empty(0, dtype=torch.int32, device=relation.device)
            output_mask = torch.empty(0, dtype=torch.int32, device=relation.device)
    return {
        "out_in_map": relation,
        "nbmaps": nbmaps,
        "nbsizes": nbsizes,
        "nbsizes_cpu": nbsizes.cpu().contiguous(),
        "sizes": (int(input_size), output_size),
        "input_mask": input_mask,
        "output_mask": output_mask,
    }


def _kernel_offsets(kernel_size: Triple, *, device: torch.device) -> Iterable[torch.Tensor]:
    for offset in product(range(kernel_size[0]), range(kernel_size[1]), range(kernel_size[2])):
        yield torch.tensor(offset, dtype=torch.long, device=device)


def _output_spatial_range(
    spatial_range: Triple | None,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> Triple | None:
    if spatial_range is None:
        return None
    return tuple(
        max(0, (spatial_range[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1)
        for i in range(3)
    )


def _tensor(values: Triple, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def _triple(value) -> Triple:
    return make_ntuple(value, ndim=3)


def _volume(value: Triple) -> int:
    return int(value[0] * value[1] * value[2])
