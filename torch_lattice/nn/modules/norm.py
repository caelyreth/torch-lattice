import torch
from torch import nn

from torch_lattice import SparseTensor
from torch_lattice.nn.utils import fapply

__all__ = ["BatchNorm", "GroupNorm", "InstanceNorm"]


class InstanceNorm(nn.InstanceNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


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

        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.
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
