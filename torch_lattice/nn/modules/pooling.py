from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from torch_lattice import SparseTensor
from torch_lattice.nn import functional as F

__all__ = [
    "AvgPool3d",
    "GlobalAvgPool",
    "GlobalMaxPool",
    "GlobalSumPool",
    "MaxPool3d",
    "Pool3d",
    "PoolTranspose3d",
    "SumPool3d",
    "TrilinearUpsample3d",
]


class Pool3d(nn.Module):
    """Local sparse 3D pooling over generated output support."""

    def __init__(
        self,
        *,
        mode: Literal["sum", "max", "avg"],
        kernel_size=2,
        stride=2,
        padding=0,
        dilation=1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.pool3d(
            input,
            mode=self.mode,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class SumPool3d(Pool3d):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1) -> None:
        super().__init__(
            mode="sum",
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


class MaxPool3d(Pool3d):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1) -> None:
        super().__init__(
            mode="max",
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


class AvgPool3d(Pool3d):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1) -> None:
        super().__init__(
            mode="avg",
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


class PoolTranspose3d(nn.Module):
    """Average-pooling transpose onto generated or explicit target support."""

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(
        self,
        input: SparseTensor,
        coordinates: SparseTensor | None = None,
    ) -> SparseTensor:
        return F.pool_transpose3d(
            input,
            coordinates,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class TrilinearUpsample3d(nn.Module):
    """Normalized trilinear upsampling on generated or target support."""

    def __init__(self, stride=2) -> None:
        super().__init__()
        self.stride = stride

    def forward(
        self,
        input: SparseTensor,
        coordinates: SparseTensor | None = None,
    ) -> SparseTensor:
        return F.trilinear_upsample3d(input, coordinates, stride=self.stride)


class GlobalAvgPool(nn.Module):
    def forward(
        self, input: SparseTensor, *, batch_size: int | None = None
    ) -> torch.Tensor:
        return F.global_avg_pool(input, batch_size=batch_size)


class GlobalMaxPool(nn.Module):
    def forward(
        self, input: SparseTensor, *, batch_size: int | None = None
    ) -> torch.Tensor:
        return F.global_max_pool(input, batch_size=batch_size)


class GlobalSumPool(nn.Module):
    def forward(
        self, input: SparseTensor, *, batch_size: int | None = None
    ) -> torch.Tensor:
        return F.global_sum_pool(input, batch_size=batch_size)
