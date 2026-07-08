from typing import List, Tuple, Union

import numpy as np
from torch import nn

from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.utils import make_ntuple

__all__ = ["SparseConvBlock", "SparseConvTransposeBlock", "SparseResBlock"]


def _support_preserving_or_strided_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    *,
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    dilation: int = 1,
):
    stride_tuple = make_ntuple(stride, ndim=3)
    if stride_tuple == (1, 1, 1):
        return spnn.SubmConv3d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
    return spnn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
    )


class SparseConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            _support_preserving_or_strided_conv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )


class SparseConvTransposeBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            spnn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )


class SparseResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            _support_preserving_or_strided_conv(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                stride=stride,
            ),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.SubmConv3d(out_channels, out_channels, kernel_size, dilation=dilation),
            spnn.BatchNorm(out_channels),
        )

        if in_channels != out_channels or np.prod(stride) != 1:
            self.shortcut = nn.Sequential(
                _support_preserving_or_strided_conv(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                ),
                spnn.BatchNorm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        x = self.relu(self.main(x) + self.shortcut(x))
        return x
