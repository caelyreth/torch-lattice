from __future__ import annotations

from typing import Literal

import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd

import torch_lattice.backend
from torch_lattice import SparseTensor
from torch_lattice.utils import make_ntuple

__all__ = ["spvoxelize", "voxelize"]


class VoxelizeFunction(Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx, feats: torch.Tensor, coords: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().int()

        if feats.device.type == "cuda":
            output = torch_lattice.backend.voxelize_forward_cuda(feats, coords, counts)
        elif feats.device.type == "cpu":
            output = torch_lattice.backend.voxelize_forward_cpu(feats, coords, counts)
        else:
            device = feats.device
            output = torch_lattice.backend.voxelize_forward_cpu(
                feats.cpu(), coords.cpu(), counts.cpu()
            ).to(device)

        ctx.for_backwards = (coords, counts, feats.shape[0])
        return output.to(feats.dtype)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, counts, input_size = ctx.for_backwards
        grad_output = grad_output.contiguous()

        if grad_output.device.type == "cuda":
            grad_feats = torch_lattice.backend.voxelize_backward_cuda(
                grad_output, coords, counts, input_size
            )
        elif grad_output.device.type == "cpu":
            grad_feats = torch_lattice.backend.voxelize_backward_cpu(
                grad_output, coords, counts, input_size
            )
        else:
            device = grad_output.device
            grad_feats = torch_lattice.backend.voxelize_backward_cpu(
                grad_output.cpu(), coords.cpu(), counts.cpu(), input_size
            ).to(device)

        return grad_feats, None, None


def spvoxelize(
    feats: torch.Tensor, coords: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    return VoxelizeFunction.apply(feats, coords, counts)


def voxelize(
    points: torch.Tensor,
    features: torch.Tensor,
    *,
    batch_indices: torch.Tensor | None = None,
    active_rows: torch.Tensor | int | None = None,
    voxel_size=1.0,
    origin=0.0,
    reduction: Literal["sum", "mean"] = "mean",
    stride=1,
) -> SparseTensor:
    """Quantize point rows into a sparse voxel tensor."""

    if reduction not in {"sum", "mean"}:
        raise ValueError("voxelize reduction must be 'sum' or 'mean'.")
    points, features, batch_indices = _active_point_rows(
        points,
        features,
        batch_indices,
        active_rows,
    )
    voxel_size = _float_triple(voxel_size, device=points.device)
    origin = _float_triple(origin, device=points.device)
    spatial = torch.floor((points - origin) / voxel_size).to(torch.int64)
    coords = torch.cat([batch_indices.to(torch.int64).view(-1, 1), spatial], dim=1)
    if coords.numel() == 0:
        return SparseTensor(
            feats=features.new_empty((0, features.shape[1])),
            coords=coords.to(torch.int32),
            stride=make_ntuple(stride, ndim=3),
            spatial_range=None,
        )
    unique_coords, inverse, counts = torch.unique(
        coords,
        sorted=True,
        return_inverse=True,
        return_counts=True,
        dim=0,
    )
    voxel_features = features.new_zeros((unique_coords.shape[0], features.shape[1]))
    voxel_features.index_add_(0, inverse, features)
    if reduction == "mean":
        voxel_features = voxel_features / counts.to(features.dtype).unsqueeze(1)
    return SparseTensor(
        feats=voxel_features,
        coords=unique_coords.to(torch.int32),
        stride=make_ntuple(stride, ndim=3),
        spatial_range=_spatial_range(unique_coords),
    )


def _active_point_rows(points, features, batch_indices, active_rows):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if features.ndim != 2 or features.shape[0] != points.shape[0]:
        raise ValueError("features must have shape (N, C) with the same N as points.")
    if batch_indices is None:
        batch_indices = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
    if active_rows is not None:
        active = int(active_rows.item() if isinstance(active_rows, torch.Tensor) else active_rows)
        points = points[:active]
        features = features[:active]
        batch_indices = batch_indices[:active]
    return points, features, batch_indices


def _float_triple(value, *, device) -> torch.Tensor:
    if isinstance(value, (int, float)):
        items = (float(value), float(value), float(value))
    else:
        items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError("expected scalar or length-3 tuple.")
    return torch.tensor(items, dtype=torch.float32, device=device)


def _spatial_range(coords: torch.Tensor):
    if coords.numel() == 0:
        return None
    max_coord = torch.max(coords, dim=0).values
    return tuple(int(value) + 1 for value in max_coord.tolist())
