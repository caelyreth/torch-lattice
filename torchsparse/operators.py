from typing import List, Optional

import torch

import torchsparse.backend
from torchsparse.tensor import SparseTensor

# from torch_scatter import scatter_sum

__all__ = ["cat", "generative_add"]


def cat(inputs: List[SparseTensor]) -> SparseTensor:
    feats = torch.cat([input.feats for input in inputs], dim=1)
    output = SparseTensor(coords=inputs[0].coords, feats=feats, stride=inputs[0].stride)
    output._caches = inputs[0]._caches
    return output


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def generative_add(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    if (
        a.C.shape == b.C.shape
        and a.C.stride() == b.C.stride()
        and a.C.dtype == b.C.dtype
        and a.C.device == b.C.device
        and a.s == b.s
        and a.spatial_range == b.spatial_range
        and (
            a.C.data_ptr() == b.C.data_ptr()
            or torch.equal(a.C, b.C)
        )
    ):
        out_tensor = SparseTensor(
            a.F + b.F,
            a.C,
            a.s,
            spatial_range=a.spatial_range,
        )
        out_tensor._caches = a._caches
        return out_tensor

    input_a = a if a.F.size(0) >= b.F.size(0) else b
    input_b = b if a.F.size(0) >= b.F.size(0) else a
    if (
        input_a.C.device.type == "cuda"
        and input_b.C.device.type == "cuda"
        and input_a.C.dtype == torch.int32
        and input_b.C.dtype == torch.int32
        and input_a.F.device == input_a.C.device
        and input_b.F.device == input_b.C.device
        and input_a.F.size(1) == input_b.F.size(1)
    ):
        from torchsparse.nn.functional.hash import sphash
        from torchsparse.nn.functional.query import sphashquery

        hashes_a = sphash(input_a.C)
        hashes_b = sphash(input_b.C)
        matches = sphashquery(hashes_a, hashes_b).int()
        if hasattr(torchsparse.backend, "generative_add_compress_cuda"):
            out_features, out_coords = torchsparse.backend.generative_add_compress_cuda(
                input_a.F,
                input_a.C,
                input_b.F,
                input_b.C,
                matches,
            )
            out_tensor = SparseTensor(
                out_features,
                out_coords,
                input_a.s,
                spatial_range=input_a.spatial_range,
            )
            out_tensor._caches = input_a._caches
            return out_tensor

        matches = matches.long()
        overlap = matches >= 0

        out_features_a = input_a.F.clone()
        overlap_matches = matches[overlap]
        out_features_a[overlap] = out_features_a[overlap] + input_b.F[overlap_matches]
        matched_b = torch.zeros(
            (input_b.F.size(0),), dtype=torch.bool, device=input_b.F.device
        )
        matched_b[overlap_matches] = True

        input_b_only = ~matched_b
        out_tensor = SparseTensor(
            torch.cat([out_features_a, input_b.F[input_b_only]], dim=0),
            torch.cat([input_a.C, input_b.C[input_b_only]], dim=0),
            input_a.s,
            spatial_range=input_a.spatial_range,
        )
        out_tensor._caches = input_a._caches
        return out_tensor

    union_coords = torch.cat([input_a.C, input_b.C], dim=0)
    union_features = torch.cat([input_a.F, input_b.F], dim=0)
    unique_coords, unique_idx = torch.unique(union_coords, dim=0, return_inverse=True)
    out_feature = scatter_sum(union_features, unique_idx, dim=0)
    out_tensor = SparseTensor(
        out_feature, unique_coords, input_a.s, spatial_range=input_a.spatial_range
    )
    out_tensor._caches = input_a._caches
    return out_tensor
