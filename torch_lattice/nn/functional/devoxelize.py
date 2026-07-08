from __future__ import annotations

from itertools import product
from typing import Literal

import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd

import torch_lattice.backend
from torch_lattice import SparseTensor

__all__ = ["calc_ti_weights", "devoxelize", "spdevoxelize"]


def calc_ti_weights(
    coords: torch.Tensor, idx_query: torch.Tensor, scale: float = 1
) -> torch.Tensor:
    with torch.no_grad():
        p = coords
        if scale != 1:
            pf = torch.floor(coords / scale) * scale
        else:
            pf = torch.floor(coords)
        pc = pf + scale

        x = p[:, 0].view(-1, 1)
        y = p[:, 1].view(-1, 1)
        z = p[:, 2].view(-1, 1)

        xf = pf[:, 0].view(-1, 1).float()
        yf = pf[:, 1].view(-1, 1).float()
        zf = pf[:, 2].view(-1, 1).float()

        xc = pc[:, 0].view(-1, 1).float()
        yc = pc[:, 1].view(-1, 1).float()
        zc = pc[:, 2].view(-1, 1).float()

        w0 = (xc - x) * (yc - y) * (zc - z)
        w1 = (xc - x) * (yc - y) * (z - zf)
        w2 = (xc - x) * (y - yf) * (zc - z)
        w3 = (xc - x) * (y - yf) * (z - zf)
        w4 = (x - xf) * (yc - y) * (zc - z)
        w5 = (x - xf) * (yc - y) * (z - zf)
        w6 = (x - xf) * (y - yf) * (zc - z)
        w7 = (x - xf) * (y - yf) * (z - zf)

        w = torch.cat([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)
        if scale != 1:
            w /= scale**3
        w[idx_query == -1] = 0
        w /= torch.sum(w, dim=1).unsqueeze(1) + 1e-8
    return w


class DevoxelizeFunction(Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx, feats: torch.Tensor, coords: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().int()
        weights = weights.contiguous()

        if feats.device.type == "cuda":
            output = torch_lattice.backend.devoxelize_forward_cuda(feats, coords, weights)
        elif feats.device.type == "cpu":
            output = torch_lattice.backend.devoxelize_forward_cpu(feats, coords, weights)
        else:
            device = feats.device
            output = torch_lattice.backend.devoxelize_forward_cpu(
                feats.cpu(), coords.cpu(), weights.cpu()
            ).to(device)

        ctx.for_backwards = (coords, weights, feats.shape[0])
        return output.to(feats.dtype)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, weights, input_size = ctx.for_backwards
        grad_output = grad_output.contiguous()

        if grad_output.device.type == "cuda":
            grad_feats = torch_lattice.backend.devoxelize_backward_cuda(
                grad_output, coords, weights, input_size
            )
        elif grad_output.device.type == "cpu":
            grad_feats = torch_lattice.backend.devoxelize_backward_cpu(
                grad_output, coords, weights, input_size
            )
        else:
            device = grad_output.device
            grad_feats = torch_lattice.backend.devoxelize_backward_cpu(
                grad_output.cpu(), coords.cpu(), weights.cpu(), input_size
            ).to(device)

        return grad_feats, None, None


def spdevoxelize(
    feats: torch.Tensor, coords: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return DevoxelizeFunction.apply(feats, coords, weights)


def devoxelize(
    points: torch.Tensor,
    voxels: SparseTensor,
    *,
    batch_indices: torch.Tensor | None = None,
    point_active_rows: torch.Tensor | int | None = None,
    voxel_size=1.0,
    origin=0.0,
    interpolation: Literal["nearest", "linear"] = "nearest",
) -> torch.Tensor:
    """Sample sparse voxel features at dense point rows."""

    if interpolation not in {"nearest", "linear"}:
        raise ValueError("devoxelize interpolation must be 'nearest' or 'linear'.")
    points, batch_indices = _active_point_rows(points, batch_indices, point_active_rows)
    voxel_size = _float_triple(voxel_size, device=points.device)
    origin = _float_triple(origin, device=points.device)
    normalized = (points - origin) / voxel_size
    if interpolation == "nearest":
        nearest = torch.floor(normalized).to(torch.int64)
        indices = _lookup_indices(voxels.coords, batch_indices, nearest)
        return _gather_or_zero(voxels.feats, indices)
    return _linear_devoxelize(normalized, voxels, batch_indices)


def _linear_devoxelize(normalized, voxels, batch_indices):
    base = torch.floor(normalized).to(torch.int64)
    frac = (normalized - base.to(normalized.dtype)).to(voxels.feats.dtype)
    output = voxels.feats.new_zeros((normalized.shape[0], voxels.feats.shape[1]))
    for corner in product((0, 1), repeat=3):
        corner_tensor = torch.tensor(corner, dtype=torch.int64, device=normalized.device)
        spatial = base + corner_tensor
        weight = torch.ones(normalized.shape[0], dtype=voxels.feats.dtype, device=normalized.device)
        for axis, bit in enumerate(corner):
            weight = weight * (frac[:, axis] if bit else (1 - frac[:, axis]))
        indices = _lookup_indices(voxels.coords, batch_indices, spatial)
        output = output + _gather_or_zero(voxels.feats, indices) * weight.unsqueeze(1)
    return output


def _lookup_indices(voxel_coords, batch_indices, spatial):
    lookup = {
        tuple(int(item) for item in row): index
        for index, row in enumerate(voxel_coords.detach().cpu().tolist())
    }
    rows = torch.cat([batch_indices.to(torch.int64).view(-1, 1), spatial], dim=1)
    values = [lookup.get(tuple(int(item) for item in row), -1) for row in rows.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.long, device=spatial.device)


def _gather_or_zero(features, indices):
    output = features.new_zeros((indices.shape[0], features.shape[1]))
    valid = indices >= 0
    if torch.any(valid):
        output[valid] = features.index_select(0, indices[valid])
    return output


def _active_point_rows(points, batch_indices, active_rows):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if batch_indices is None:
        batch_indices = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
    if active_rows is not None:
        active = int(active_rows.item() if isinstance(active_rows, torch.Tensor) else active_rows)
        points = points[:active]
        batch_indices = batch_indices[:active]
    return points, batch_indices


def _float_triple(value, *, device) -> torch.Tensor:
    if isinstance(value, (int, float)):
        items = (float(value), float(value), float(value))
    else:
        items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError("expected scalar or length-3 tuple.")
    return torch.tensor(items, dtype=torch.float32, device=device)
