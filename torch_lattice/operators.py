from __future__ import annotations

from typing import Literal

import torch

import torch_lattice.backend
from torch_lattice.tensor import SparseTensor

# from torch_scatter import scatter_sum

SparseJoin = Literal["inner", "left", "right", "outer"]
SparseBinaryOp = Literal["add", "sub", "mul", "maximum", "minimum"]

__all__ = [
    "cat",
    "generative_add",
    "sparse_add",
    "sparse_binary",
    "sparse_cat",
    "sparse_maximum",
    "sparse_minimum",
    "sparse_mul",
    "sparse_sub",
]


def cat(
    inputs: list[SparseTensor],
    *,
    join: SparseJoin = "inner",
) -> SparseTensor:
    return sparse_cat(inputs, join=join)


def sparse_cat(
    inputs: list[SparseTensor],
    *,
    join: SparseJoin = "inner",
) -> SparseTensor:
    if not inputs:
        raise ValueError("sparse_cat requires at least one sparse tensor.")
    output = inputs[0]
    for rhs in inputs[1:]:
        output = _sparse_cat_pair(output, rhs, join=join)
    return output


def sparse_binary(
    lhs: SparseTensor,
    rhs: SparseTensor,
    op: SparseBinaryOp,
    *,
    join: SparseJoin = "outer",
    lhs_fill: float = 0.0,
    rhs_fill: float = 0.0,
) -> SparseTensor:
    _require_compatible(lhs, rhs)
    if lhs.feats.size(1) != rhs.feats.size(1):
        raise ValueError("sparse binary operands must have matching channels.")
    if _same_coords(lhs, rhs):
        return _replace_sparse(lhs, _apply_binary(lhs.feats, rhs.feats, op))
    alignment = _align_sparse(lhs, rhs, join=join)
    lhs_features = _gather_aligned(lhs.feats, alignment.lhs_rows, fill=lhs_fill)
    rhs_features = _gather_aligned(rhs.feats, alignment.rhs_rows, fill=rhs_fill)
    return _new_sparse(
        lhs,
        coords=alignment.coords,
        feats=_apply_binary(lhs_features, rhs_features, op),
    )


def sparse_add(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = "outer",
    lhs_fill: float = 0.0,
    rhs_fill: float = 0.0,
) -> SparseTensor:
    return sparse_binary(
        lhs,
        rhs,
        "add",
        join=join,
        lhs_fill=lhs_fill,
        rhs_fill=rhs_fill,
    )


def sparse_sub(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = "outer",
    lhs_fill: float = 0.0,
    rhs_fill: float = 0.0,
) -> SparseTensor:
    return sparse_binary(
        lhs,
        rhs,
        "sub",
        join=join,
        lhs_fill=lhs_fill,
        rhs_fill=rhs_fill,
    )


def sparse_mul(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = "inner",
    lhs_fill: float = 0.0,
    rhs_fill: float = 0.0,
) -> SparseTensor:
    return sparse_binary(
        lhs,
        rhs,
        "mul",
        join=join,
        lhs_fill=lhs_fill,
        rhs_fill=rhs_fill,
    )


def sparse_maximum(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = "inner",
    lhs_fill: float = 0.0,
    rhs_fill: float = 0.0,
) -> SparseTensor:
    return sparse_binary(
        lhs,
        rhs,
        "maximum",
        join=join,
        lhs_fill=lhs_fill,
        rhs_fill=rhs_fill,
    )


def sparse_minimum(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = "inner",
    lhs_fill: float = 0.0,
    rhs_fill: float = 0.0,
) -> SparseTensor:
    return sparse_binary(
        lhs,
        rhs,
        "minimum",
        join=join,
        lhs_fill=lhs_fill,
        rhs_fill=rhs_fill,
    )


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
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
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
    return out.scatter_add_(dim, index, src)


def generative_add(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    if _same_coords(a, b):
        out_tensor = sparse_add(a, b, join="inner")
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
        from torch_lattice.nn.functional.hash import sphash
        from torch_lattice.nn.functional.query import sphashquery

        hashes_a = sphash(input_a.C)
        hashes_b = sphash(input_b.C)
        matches = sphashquery(hashes_a, hashes_b).int()
        if hasattr(torch_lattice.backend, "generative_add_compress_cuda"):
            out_features, out_coords = torch_lattice.backend.generative_add_compress_cuda(
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

    return sparse_add(a, b, join="outer")


class _SparseAlignment:
    def __init__(
        self,
        coords: torch.Tensor,
        lhs_rows: torch.Tensor,
        rhs_rows: torch.Tensor,
    ) -> None:
        self.coords = coords
        self.lhs_rows = lhs_rows
        self.rhs_rows = rhs_rows


def _sparse_cat_pair(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin,
) -> SparseTensor:
    _require_compatible(lhs, rhs)
    if _same_coords(lhs, rhs):
        return _replace_sparse(lhs, torch.cat([lhs.feats, rhs.feats], dim=1))
    alignment = _align_sparse(lhs, rhs, join=join)
    lhs_features = _gather_aligned(lhs.feats, alignment.lhs_rows)
    rhs_features = _gather_aligned(rhs.feats, alignment.rhs_rows)
    return _new_sparse(
        lhs,
        coords=alignment.coords,
        feats=torch.cat([lhs_features, rhs_features], dim=1),
    )


def _align_sparse(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin,
) -> _SparseAlignment:
    _validate_join(join)
    coords = torch.cat([lhs.coords, rhs.coords], dim=0)
    unique, inverse = torch.unique(coords, dim=0, return_inverse=True)
    lhs_inverse = inverse[: lhs.coords.size(0)]
    rhs_inverse = inverse[lhs.coords.size(0) :]
    lhs_rows = torch.full(
        (unique.size(0),),
        -1,
        dtype=torch.long,
        device=unique.device,
    )
    rhs_rows = torch.full_like(lhs_rows, -1)
    lhs_rows[lhs_inverse] = torch.arange(
        lhs.coords.size(0),
        dtype=torch.long,
        device=unique.device,
    )
    rhs_rows[rhs_inverse] = torch.arange(
        rhs.coords.size(0),
        dtype=torch.long,
        device=unique.device,
    )
    lhs_present = lhs_rows >= 0
    rhs_present = rhs_rows >= 0
    if join == "inner":
        mask = lhs_present & rhs_present
    elif join == "left":
        mask = lhs_present
    elif join == "right":
        mask = rhs_present
    else:
        mask = lhs_present | rhs_present
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    return _SparseAlignment(
        unique[selected],
        lhs_rows[selected],
        rhs_rows[selected],
    )


def _gather_aligned(
    features: torch.Tensor,
    rows: torch.Tensor,
    *,
    fill: float = 0.0,
) -> torch.Tensor:
    clipped = rows.clamp_min(0)
    gathered = features.index_select(0, clipped)
    valid = rows >= 0
    if bool(torch.all(valid)):
        return gathered
    if fill == 0.0:
        return gathered * valid.to(features.dtype).unsqueeze(1)
    filled = torch.full_like(gathered, float(fill))
    return torch.where(valid.unsqueeze(1), gathered, filled)


def _apply_binary(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    op: SparseBinaryOp,
) -> torch.Tensor:
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "mul":
        return lhs * rhs
    if op == "maximum":
        return torch.maximum(lhs, rhs)
    if op == "minimum":
        return torch.minimum(lhs, rhs)
    raise ValueError(f"unsupported sparse binary op: {op}")


def _same_coords(lhs: SparseTensor, rhs: SparseTensor) -> bool:
    return (
        lhs.C.shape == rhs.C.shape
        and lhs.C.stride() == rhs.C.stride()
        and lhs.C.dtype == rhs.C.dtype
        and lhs.C.device == rhs.C.device
        and lhs.s == rhs.s
        and lhs.spatial_range == rhs.spatial_range
        and (lhs.C.data_ptr() == rhs.C.data_ptr() or torch.equal(lhs.C, rhs.C))
    )


def _require_compatible(lhs: SparseTensor, rhs: SparseTensor) -> None:
    if lhs.stride != rhs.stride:
        raise ValueError("sparse tensor strides must match.")
    if lhs.coords.dtype != rhs.coords.dtype:
        raise ValueError("sparse tensor coordinate dtypes must match.")
    if lhs.coords.device != rhs.coords.device:
        raise ValueError("sparse tensor coordinate devices must match.")


def _replace_sparse(source: SparseTensor, feats: torch.Tensor) -> SparseTensor:
    return _new_sparse(source, coords=source.coords, feats=feats)


def _new_sparse(
    source: SparseTensor,
    *,
    coords: torch.Tensor,
    feats: torch.Tensor,
) -> SparseTensor:
    output = SparseTensor(
        coords=coords,
        feats=feats,
        stride=source.stride,
        spatial_range=source.spatial_range,
    )
    output._caches = source._caches
    return output


def _validate_join(join: str) -> None:
    if join not in {"inner", "left", "right", "outer"}:
        raise ValueError("join must be 'inner', 'left', 'right', or 'outer'.")
