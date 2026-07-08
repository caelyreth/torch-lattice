from __future__ import annotations

import torch
from torch import nn

from torch_lattice import SparseTensor
from torch_lattice.nn.utils import fapply

__all__ = [
    "BatchNorm",
    "GroupNorm",
    "InstanceNorm",
    "LayerNorm",
    "RMSNorm",
]


class InstanceNorm(nn.InstanceNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LayerNorm(nn.LayerNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        if len(tuple(normalized_shape)) != 1:
            raise ValueError("Sparse RMSNorm expects one feature dimension.")
        self.normalized_shape = tuple(int(item) for item in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(*self.normalized_shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, input: SparseTensor) -> SparseTensor:
        def normalize(feats: torch.Tensor) -> torch.Tensor:
            variance = feats.float().pow(2).mean(dim=-1, keepdim=True)
            out = feats * torch.rsqrt(variance.to(feats.dtype) + self.eps)
            if self.weight is not None:
                out = out * self.weight.to(dtype=out.dtype).reshape(1, -1)
            return out

        return fapply(input, normalize)


class GroupNorm(nn.GroupNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.coords, input.feats, input.stride

        if coords.numel() == 0:
            output = SparseTensor(
                coords=coords,
                feats=feats,
                stride=stride,
                spatial_range=input.spatial_range,
            )
            output._caches = input._caches
            return output

        num_channels = feats.shape[1]
        single_batch = (
            input.spatial_range is not None
            and len(input.spatial_range) > 0
            and input.spatial_range[0] == 1
        )
        if single_batch:
            grouped = feats.reshape(feats.shape[0], self.num_groups, -1)
            grouped_float = grouped.float()
            var, mean = torch.var_mean(
                grouped_float, dim=(0, 2), unbiased=False, keepdim=True
            )
            nfeats = (grouped_float - mean) * torch.rsqrt(var + self.eps)
            nfeats = nfeats.to(feats.dtype).reshape_as(feats)
            if self.weight is not None:
                nfeats = nfeats * self.weight.reshape(1, num_channels)
            if self.bias is not None:
                nfeats = nfeats + self.bias.reshape(1, num_channels)
            output = SparseTensor(
                coords=coords,
                feats=nfeats,
                stride=stride,
                spatial_range=input.spatial_range,
            )
            output._caches = input._caches
            return output

        batch_size = torch.max(coords[:, 0]).item() + 1
        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, 0] == k
            bfeats = feats[indices]
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        output = SparseTensor(
            coords=coords,
            feats=nfeats,
            stride=stride,
            spatial_range=input.spatial_range,
        )
        output._caches = input._caches
        return output
