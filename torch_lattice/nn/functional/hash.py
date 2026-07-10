from typing import Optional

import torch

import torch_lattice.backend

__all__ = ["sphash"]


def sphash(
    coords: torch.Tensor, offsets: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Hash ``(batch, x, y, z)`` int32 coordinate rows."""

    if coords.dtype != torch.int32:
        raise TypeError("coords must use int32 dtype")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape (N, 4)")
    coords = coords.contiguous()

    if offsets is None:
        if coords.device.type == "cuda":
            return torch_lattice.backend.hash_cuda(coords)
        elif coords.device.type == "cpu":
            return torch_lattice.backend.hash_cpu(coords)
        else:
            device = coords.device
            return torch_lattice.backend.hash_cpu(coords.cpu()).to(device)
    else:
        if offsets.dtype != torch.int32:
            raise TypeError("offsets must use int32 dtype")
        if offsets.ndim != 2 or offsets.shape[1] != 3:
            raise ValueError("offsets must have shape (K, 3)")
        offsets = offsets.contiguous()

        if coords.device.type == "cuda":
            return torch_lattice.backend.kernel_hash_cuda(coords, offsets)
        elif coords.device.type == "cpu":
            return torch_lattice.backend.kernel_hash_cpu(coords, offsets)
        else:
            device = coords.device
            return torch_lattice.backend.kernel_hash_cpu(
                coords.cpu(), offsets.cpu()
            ).to(device)
