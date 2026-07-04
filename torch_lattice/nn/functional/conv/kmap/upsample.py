from typing import Tuple, Union, Optional

import torch

import torch_lattice
import torch_lattice.backend
from torch_lattice.utils import make_ntuple, make_tensor
from torch_lattice.nn.utils.kernel import get_kernel_offsets

__all__ = ["spupsample_generative"]


def spupsample_generative(
    _coords: torch.Tensor,
    stride: Union[int, Tuple[int, ...]] = 2,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    padding: torch.Tensor = 0,
    spatial_range: Optional[Tuple[int]] = None,
) -> torch.Tensor:
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    padding = make_ntuple(padding, ndim=3)
    sample_stride = make_tensor(
        stride, dtype=torch.int, device=_coords.device
    ).unsqueeze(0)
    # stride and dilation are both 1
    kernel_offsets = get_kernel_offsets(kernel_size, 1, 1, device=_coords.device)
    assert (
        spatial_range is not None
    ), "spatial range must be specified in generative mode"
    if (
        _coords.device.type == "cuda"
        and _coords.dtype == torch.int32
        and not torch_lattice.tensor.get_allow_negative_coordinates()
        and all(p == 0 for p in padding)
        and all(stride[k] == kernel_size[k] for k in range(3))
        and all(
            spatial_range[k + 1]
            >= int(_coords[:, k + 1].max().item()) * stride[k] + kernel_size[k]
            for k in range(3)
        )
        and hasattr(torch_lattice.backend, "upsample_generative_cuda")
    ):
        stride_t = make_tensor(stride, dtype=torch.int, device=_coords.device)
        return torch_lattice.backend.upsample_generative_cuda(
            _coords.contiguous(),
            kernel_offsets.contiguous(),
            stride_t,
        )

    coords = _coords.clone()
    coords[:, 1:] *= sample_stride
    coords = coords.unsqueeze(1).repeat(1, kernel_offsets.size(0), 1)
    coords[:, :, 1:] = coords[:, :, 1:] + kernel_offsets.unsqueeze(0)
    for i in range(1, coords.size(-1)):
        coords[:, :, i].clamp_(min=0, max=spatial_range[i] - 1)
    coords = coords.reshape(-1, coords.size(-1))
    coords = torch.unique(coords, dim=0)
    return coords
