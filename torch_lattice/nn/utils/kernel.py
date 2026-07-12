import torch
from lattice_contract import sparse_kernel_offsets

from torch_lattice.utils import make_ntuple, make_tensor

__all__ = ["get_kernel_offsets"]


def get_kernel_offsets(
    size: int | tuple[int, ...],
    stride: int | tuple[int, ...] = 1,
    dilation: int | tuple[int, ...] = 1,
    device="cpu",
) -> torch.Tensor:
    size = make_ntuple(size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    scale = tuple(stride[index] * dilation[index] for index in range(3))
    return make_tensor(
        sparse_kernel_offsets(size, scale), dtype=torch.int, device=device
    )
