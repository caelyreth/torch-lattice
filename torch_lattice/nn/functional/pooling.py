from __future__ import annotations

from typing import Literal

import torch

from torch_lattice import SparseTensor
from torch_lattice.utils import make_ntuple

from .relation import build_pool_output_coords, build_target_out_in_map

__all__ = [
    "avg_pool3d",
    "global_avg_pool",
    "global_max_pool",
    "max_pool3d",
    "pool3d",
    "sum_pool3d",
]

PoolMode = Literal["sum", "max", "avg"]


def pool3d(
    inputs: SparseTensor,
    *,
    mode: PoolMode,
    kernel_size=2,
    stride=2,
    padding=0,
    dilation=1,
) -> SparseTensor:
    """Local sparse 3D pooling over convolution-style neighborhoods."""

    if mode not in {"sum", "max", "avg"}:
        raise ValueError("pool3d mode must be 'sum', 'max', or 'avg'.")
    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    padding = make_ntuple(padding, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)
    output_coords = build_pool_output_coords(
        inputs.coords,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        spatial_range=inputs.spatial_range,
    )
    relation = build_target_out_in_map(
        inputs.coords,
        output_coords,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    feats = _pool_features(inputs.feats, relation, mode)
    output_stride = tuple(inputs.stride[index] * stride[index] for index in range(3))
    output = SparseTensor(
        feats=feats,
        coords=output_coords,
        stride=output_stride,
        spatial_range=_pooled_spatial_range(
            inputs.spatial_range,
            kernel_size,
            stride,
            padding,
            dilation,
        ),
    )
    output._caches = inputs._caches
    output._caches.cmaps.setdefault(output.stride, (output.coords, output.spatial_range))
    return output


def sum_pool3d(inputs: SparseTensor, **kwargs) -> SparseTensor:
    return pool3d(inputs, mode="sum", **kwargs)


def max_pool3d(inputs: SparseTensor, **kwargs) -> SparseTensor:
    return pool3d(inputs, mode="max", **kwargs)


def avg_pool3d(inputs: SparseTensor, **kwargs) -> SparseTensor:
    return pool3d(inputs, mode="avg", **kwargs)


def global_avg_pool(inputs: SparseTensor) -> torch.Tensor:
    if (
        inputs.spatial_range is not None
        and len(inputs.spatial_range) > 0
        and inputs.spatial_range[0] == 1
    ):
        return torch.mean(inputs.feats, dim=0, keepdim=True)

    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.mean(input, dim=0)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def global_max_pool(inputs: SparseTensor) -> torch.Tensor:
    if (
        inputs.spatial_range is not None
        and len(inputs.spatial_range) > 0
        and inputs.spatial_range[0] == 1
    ):
        return torch.max(inputs.feats, dim=0, keepdim=True)[0]

    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.max(input, dim=0)[0]
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def _pool_features(feats: torch.Tensor, relation: torch.Tensor, mode: PoolMode) -> torch.Tensor:
    output_size = int(relation.shape[0])
    channels = int(feats.shape[1])
    valid = relation >= 0
    if not torch.any(valid):
        return feats.new_zeros((output_size, channels))
    out_rows = torch.nonzero(valid, as_tuple=False)[:, 0]
    in_rows = relation[valid].to(torch.long)
    gathered = feats.index_select(0, in_rows)
    if mode in {"sum", "avg"}:
        output = feats.new_zeros((output_size, channels))
        output.index_add_(0, out_rows, gathered)
        if mode == "avg":
            counts = valid.sum(dim=1).clamp_min(1).to(feats.dtype).unsqueeze(1)
            output = output / counts
        return output

    output = feats.new_full((output_size, channels), -torch.inf)
    scatter_rows = out_rows.view(-1, 1).expand(-1, channels)
    output.scatter_reduce_(0, scatter_rows, gathered, reduce="amax", include_self=True)
    empty = ~torch.any(valid, dim=1)
    if torch.any(empty):
        output[empty] = 0
    return output


def _pooled_spatial_range(spatial_range, kernel_size, stride, padding, dilation):
    if spatial_range is None:
        return None
    return tuple(spatial_range[:1]) + tuple(
        max(
            0,
            (
                int(spatial_range[index + 1])
                + 2 * padding[index]
                - dilation[index] * (kernel_size[index] - 1)
                - 1
            )
            // stride[index]
            + 1,
        )
        for index in range(3)
    )
