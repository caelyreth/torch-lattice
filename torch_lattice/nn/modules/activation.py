from __future__ import annotations

from torch import nn

from torch_lattice import SparseTensor
from torch_lattice.nn.utils import fapply

__all__ = [
    "GELU",
    "LeakyReLU",
    "ReLU",
    "SiLU",
    "Sigmoid",
    "Softplus",
    "Tanh",
]


class ReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class SiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class GELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class Sigmoid(nn.Sigmoid):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class Tanh(nn.Tanh):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class Softplus(nn.Softplus):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
