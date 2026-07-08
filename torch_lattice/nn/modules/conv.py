from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from torch_lattice import SparseTensor
from torch_lattice.nn import functional as F
from torch_lattice.utils import make_ntuple

__all__ = [
    "Conv3d",
    "SubmConv3d",
    "ConvTranspose3d",
    "GenerativeConvTranspose3d",
]


class _BaseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        bias: bool = False,
        *,
        subm: bool = False,
        transposed: bool = False,
        generative: bool = False,
        config: Dict | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.padding = make_ntuple(padding, 3)
        self.subm = bool(subm)
        self.transposed = bool(transposed)
        self.generative = bool(generative)

        if self.subm:
            if self.transposed or self.generative:
                raise ValueError("SubmConv3d is not a transposed convolution.")
            if self.stride != (1, 1, 1):
                raise ValueError("SubmConv3d preserves support and requires stride=1.")
            if any(size % 2 == 0 for size in self.kernel_size):
                raise ValueError("SubmConv3d requires odd kernel sizes.")
            self.padding = tuple((size - 1) // 2 for size in self.kernel_size)
        if self.generative and not self.transposed:
            raise ValueError("GenerativeConvTranspose3d requires transposed=True.")

        self._config = config

        self.kernel_volume = int(np.prod(self.kernel_size))
        if self.kernel_volume > 1 or self.stride != (1, 1, 1):
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels)
            )
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.stride != (1,) * len(self.stride):
            s += ", stride={stride}"
        if self.padding != (0, 0, 0) and not self.subm:
            s += ", padding={padding}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        fan_channels = self.out_channels if self.transposed else self.in_channels
        std = 1 / math.sqrt(fan_channels * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.conv3d(
            input,
            weight=self.kernel,
            kernel_size=self.kernel_size,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            subm=self.subm,
            transposed=self.transposed,
            generative=self.generative,
            config=self._config,
            training=self.training,
        )


class Conv3d(_BaseConv3d):
    """Support-generating sparse 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        bias: bool = False,
        config: Dict | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            config=config,
        )


class SubmConv3d(_BaseConv3d):
    """Support-preserving submanifold sparse 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        dilation: int = 1,
        bias: bool = False,
        config: Dict | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
            subm=True,
            config=config,
        )


class ConvTranspose3d(_BaseConv3d):
    """Sparse transposed 3D convolution using an existing inverse support map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        bias: bool = False,
        config: Dict | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            transposed=True,
            config=config,
        )


class GenerativeConvTranspose3d(_BaseConv3d):
    """Sparse transposed 3D convolution that generates its output support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        bias: bool = False,
        config: Dict | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            transposed=True,
            generative=True,
            config=config,
        )
