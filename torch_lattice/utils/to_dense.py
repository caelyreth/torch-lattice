import torch
from torch.autograd import Function

from typing import Tuple

import torch_lattice.backend
from torch_lattice.utils.utils import make_tensor

__all__ = ["to_dense"]


class ToDenseFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_range: Tuple[int],
    ) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().int()
        outputs = torch.zeros(
            spatial_range + (feats.size(1),), dtype=feats.dtype, device=feats.device
        )
        spatial_range = make_tensor(spatial_range, dtype=torch.int, device=feats.device)

        if feats.device.type == "cuda":
            torch_lattice.backend.to_dense_forward_cuda(
                feats, coords, spatial_range, outputs
            )
        else:
            raise NotImplementedError(
                f"dense conversion is not implemented for {feats.device.type}"
            )

        ctx.for_backwards = (coords, spatial_range)
        return outputs.to(feats.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        coords, spatial_range = ctx.for_backwards
        grad_output = grad_output.contiguous()
        grad_feats = torch.zeros(
            coords.size(0),
            grad_output.size(-1),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        if grad_output.device.type == "cuda":
            torch_lattice.backend.to_dense_backward_cuda(
                grad_output, coords, spatial_range, grad_feats
            )
        else:
            raise NotImplementedError(
                "dense conversion backward is not implemented for "
                f"{grad_output.device.type}"
            )

        return grad_feats, None, None


def to_dense(
    feats: torch.Tensor, coords: torch.Tensor, spatial_range: Tuple[int, ...]
) -> torch.Tensor:
    return ToDenseFunction.apply(feats, coords, spatial_range)
