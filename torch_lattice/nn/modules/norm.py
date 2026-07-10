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
        if input.coords.shape[0] == 0:
            return input.replace(feats=input.feats)
        output = torch.empty_like(input.feats)
        batch_size = _batch_size(input)
        for batch_index in range(batch_size):
            rows = input.coords[:, 0] == batch_index
            feats = input.feats[rows]
            if feats.shape[0] == 0:
                continue
            variance, mean = torch.var_mean(
                feats.float(), dim=0, unbiased=False, keepdim=True
            )
            normalized = (feats.float() - mean) * torch.rsqrt(variance + self.eps)
            normalized = normalized.to(input.feats.dtype)
            if self.weight is not None:
                normalized = normalized * self.weight.reshape(1, -1)
            if self.bias is not None:
                normalized = normalized + self.bias.reshape(1, -1)
            output[rows] = normalized
        return input.replace(feats=output)


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
        coords, feats = input.coords, input.feats

        if coords.numel() == 0:
            return input.replace(feats=feats)

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
            return input.replace(feats=nfeats)

        batch_size = _batch_size(input)
        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, 0] == k
            bfeats = feats[indices]
            if bfeats.shape[0] == 0:
                continue
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        return input.replace(feats=nfeats)


def _batch_size(input: SparseTensor) -> int:
    if input.batch_counts is not None:
        return len(input.batch_counts)
    if input.spatial_range is not None:
        return int(input.spatial_range[0])
    if input.coords.shape[0] == 0:
        return 0
    return int(input.coords[:, 0].max().item()) + 1
