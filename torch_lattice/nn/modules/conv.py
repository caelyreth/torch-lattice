from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from lattice_contract import kernel_positions
from torch import nn

from torch_lattice import SparseTensor
from torch_lattice.nn import functional as F
from torch_lattice.utils import make_ntuple

__all__ = [
    "Conv3d",
    "ConvTranspose3d",
    "GenerativeConvTranspose3d",
    "NormalizedConvTranspose3d",
    "NormalizedGenerativeConvTranspose3d",
    "NormalizedSubmConv3d",
    "SubmConv3d",
]


class _BaseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        *,
        subm: bool = False,
        transposed: bool = False,
        generative: bool = False,
        normalized: bool = False,
        eps: float = 1e-8,
        config: Mapping | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = make_ntuple(dilation, ndim=3)
        self.padding = make_ntuple(padding, 3)
        self.subm = bool(subm)
        self.transposed = bool(transposed)
        self.generative = bool(generative)
        self.normalized = bool(normalized)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

        if self.subm:
            if self.transposed or self.generative:
                raise ValueError("SubmConv3d is not a transposed convolution.")
            if self.stride != (1, 1, 1):
                raise ValueError("SubmConv3d preserves support and requires stride=1.")
            if any(size % 2 == 0 for size in self.kernel_size):
                raise ValueError("SubmConv3d requires odd kernel sizes.")
            self.padding = tuple(
                self.dilation[index] * (size - 1) // 2
                for index, size in enumerate(self.kernel_size)
            )
        if self.generative and not self.transposed:
            raise ValueError("GenerativeConvTranspose3d requires transposed=True.")

        self._config = config

        self.kernel_volume = math.prod(self.kernel_size)
        # K rows are canonical x/y/z positions with z varying fastest. Keeping
        # the execution tensor kernel-major preserves CUDA GEMM efficiency while
        # the persistent position buffer makes checkpoint semantics explicit.
        self.weight = nn.Parameter(
            torch.zeros(self.kernel_volume, in_channels, out_channels)
        )
        self.register_buffer(
            'kernel_positions',
            torch.tensor(kernel_positions(self.kernel_size), dtype=torch.int32),
            persistent=True,
        )
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
        if self.dilation != (1, 1, 1):
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        fan_channels = self.out_channels if self.transposed else self.in_channels
        std = 1 / math.sqrt(fan_channels * self.kernel_volume)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(
        self,
        input: SparseTensor,
        coordinates: SparseTensor | None = None,
    ) -> SparseTensor:
        convolution = F.normalized_conv3d if self.normalized else F.conv3d
        return convolution(
            input,
            weight=self.weight,
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
            coordinates=coordinates,
            **({"eps": self.eps} if self.normalized else {}),
        )


class Conv3d(_BaseConv3d):
    """Support-generating sparse 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        config: Mapping | None = None,
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
        kernel_size: int | Sequence[int] = 3,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        config: Mapping | None = None,
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


class NormalizedSubmConv3d(SubmConv3d):
    """Weight-normalized convolution on input coordinate support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        eps: float = 1e-8,
        config: Mapping | None = None,
    ) -> None:
        _BaseConv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
            subm=True,
            normalized=True,
            eps=eps,
            config=config,
        )


class ConvTranspose3d(_BaseConv3d):
    """Sparse transposed 3D convolution using an existing inverse support map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        config: Mapping | None = None,
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


class NormalizedConvTranspose3d(ConvTranspose3d):
    """Weight-normalized sparse transpose convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        eps: float = 1e-8,
        config: Mapping | None = None,
    ) -> None:
        _BaseConv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            transposed=True,
            normalized=True,
            eps=eps,
            config=config,
        )


class GenerativeConvTranspose3d(_BaseConv3d):
    """Sparse transposed 3D convolution that generates its output support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        config: Mapping | None = None,
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


class NormalizedGenerativeConvTranspose3d(GenerativeConvTranspose3d):
    """Weight-normalized transpose convolution with generated support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = False,
        eps: float = 1e-8,
        config: Mapping | None = None,
    ) -> None:
        _BaseConv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            transposed=True,
            generative=True,
            normalized=True,
            eps=eps,
            config=config,
        )
