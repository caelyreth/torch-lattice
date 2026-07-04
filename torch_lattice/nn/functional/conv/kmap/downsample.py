from typing import Tuple, Union, Optional

import torch

import torch_lattice
import torch_lattice.backend
from torch_lattice.utils import make_ntuple, make_tensor

__all__ = ["spdownsample"]


def spdownsample(
    _coords: torch.Tensor,
    stride: Union[int, Tuple[int, ...]] = 2,
    kernel_size: Union[int, Tuple[int, ...]] = 2,
    padding: torch.Tensor = 0,
    spatial_range: Optional[Tuple[int]] = None,
    downsample_mode: str = "spconv",
) -> torch.Tensor:
    assert downsample_mode in ["spconv", "minkowski"]

    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    padding = make_ntuple(padding, ndim=3)

    sample_stride = tuple([stride[k] for k in range(3)])
    sample_stride = make_tensor(
        sample_stride, dtype=torch.int, device=_coords.device
    ).unsqueeze(dim=0)

    if (
        all(stride[k] in [1, kernel_size[k]] for k in range(3))
        or downsample_mode == "minkowski"
    ):
        if (
            _coords.device.type == "cuda"
            and _coords.dtype == torch.int32
            and not torch_lattice.tensor.get_allow_negative_coordinates()
            and hasattr(torch_lattice.backend, "downsample_simple_cuda")
        ):
            stride_t = make_tensor(stride, dtype=torch.int, device=_coords.device)
            if spatial_range is not None:
                coords_max = make_tensor(
                    (0,)
                    + tuple(
                        (int(spatial_range[k]) - 1) // stride[k]
                        for k in range(3)
                    ),
                    dtype=torch.int,
                    device=_coords.device,
                )
            else:
                coords_max = _coords.max(0).values
                coords_max[1:] = coords_max[1:] // stride_t
            return torch_lattice.backend.downsample_simple_cuda(
                _coords.contiguous(), coords_max, stride_t
            )

        coords = _coords.clone()
        coords[:, 1:] = torch.div(coords[:, 1:], sample_stride.float()).floor()
        coords = torch.unique(coords, dim=0)
        return coords
    else:
        if _coords.device.type == "cuda":
            _coords = _coords.contiguous()

            padding_t = make_tensor(padding, dtype=torch.int, device=_coords.device)
            kernel_size_t = make_tensor(
                kernel_size, dtype=torch.int, device=_coords.device
            )
            stride_t = make_tensor(stride, dtype=torch.int, device=_coords.device)

            if spatial_range is not None:
                coords_max_tuple = tuple(x - 1 for x in spatial_range)
                coords_max = make_tensor(
                    coords_max_tuple, dtype=torch.int, device=_coords.device
                )
            else:
                coords_max = _coords.max(0).values
                coords_max[1:] = (
                    coords_max[1:] + 2 * padding_t - (kernel_size_t - 1)
                ) // stride_t

            if torch_lattice.tensor.get_allow_negative_coordinates():
                coords_min = _coords.min(0).values
                coords_min[1:] = torch.div(
                    coords_min[1:] - 2 * padding_t + (kernel_size_t - 1), stride_t
                )
            else:
                coords_min = make_tensor(
                    (0, 0, 0, 0), dtype=torch.int, device=_coords.device
                )

            out_coords = torch_lattice.backend.downsample_cuda(
                _coords,
                coords_max,
                coords_min,
                kernel_size_t,
                stride_t,
                padding_t,
            )
            return out_coords
        else:
            raise NotImplementedError
