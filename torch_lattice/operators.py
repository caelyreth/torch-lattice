from __future__ import annotations

from typing import Literal

import torch

import torch_lattice.backend
from torch_lattice.tensor import SparseTensor

SparseJoin = Literal["inner", "left", "right", "outer"]
SparseBinaryOp = Literal["add", "sub", "mul", "maximum", "minimum"]

__all__ = [
    "cat",
    "generative_add",
    "reindex_sparse",
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


def reindex_sparse(
    input: SparseTensor,
    target: SparseTensor,
    *,
    fill: float = 0.0,
) -> SparseTensor:
    """Gather ``input`` features onto the exact row order of ``target``."""
    _require_compatible(input, target)
    if _same_coords(input, target):
        return target.replace(feats=input.feats)
    rows = _coordinate_rows(input.coords, target.coords)
    return target.replace(feats=_gather_aligned(input.feats, rows, fill=fill))


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
        return sparse_add(a, b, join="inner")

    input_a = a if a.feats.size(0) >= b.feats.size(0) else b
    input_b = b if a.feats.size(0) >= b.feats.size(0) else a
    if (
        input_a.coords.device.type == "cuda"
        and input_b.coords.device.type == "cuda"
        and input_a.coords.dtype == torch.int32
        and input_b.coords.dtype == torch.int32
        and input_a.feats.device == input_a.coords.device
        and input_b.feats.device == input_b.coords.device
        and input_a.feats.size(1) == input_b.feats.size(1)
    ):
        from torch_lattice.nn.functional.hash import sphash
        from torch_lattice.nn.functional.query import sphashquery

        hashes_a = sphash(input_a.coords)
        hashes_b = sphash(input_b.coords)
        matches = sphashquery(hashes_a, hashes_b).int()
        if hasattr(torch_lattice.backend, "generative_add_compress_cuda"):
            out_features, out_coords = (
                torch_lattice.backend.generative_add_compress_cuda(
                    input_a.feats,
                    input_a.coords,
                    input_b.feats,
                    input_b.coords,
                    matches,
                )
            )
            return input_a.with_coordinates(
                feats=out_features,
                coords=out_coords,
            )

        matches = matches.long()
        overlap = matches >= 0

        out_features_a = input_a.feats.clone()
        overlap_matches = matches[overlap]
        out_features_a[overlap] = (
            out_features_a[overlap] + input_b.feats[overlap_matches]
        )
        matched_b = torch.zeros(
            (input_b.feats.size(0),),
            dtype=torch.bool,
            device=input_b.feats.device,
        )
        matched_b[overlap_matches] = True

        input_b_only = ~matched_b
        return input_a.with_coordinates(
            feats=torch.cat([out_features_a, input_b.feats[input_b_only]], dim=0),
            coords=torch.cat([input_a.coords, input_b.coords[input_b_only]], dim=0),
        )

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
    if features.shape[0] == 0:
        return features.new_full((rows.shape[0], features.shape[1]), float(fill))
    clipped = rows.clamp_min(0)
    gathered = features.index_select(0, clipped)
    valid = rows >= 0
    if bool(torch.all(valid)):
        return gathered
    if fill == 0.0:
        return gathered * valid.to(features.dtype).unsqueeze(1)
    filled = torch.full_like(gathered, float(fill))
    return torch.where(valid.unsqueeze(1), gathered, filled)


def _coordinate_rows(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if source.shape[0] == 0:
        return torch.full(
            (target.shape[0],), -1, dtype=torch.long, device=target.device
        )
    if source.dtype == torch.int32:
        from torch_lattice.nn.functional.hash import sphash
        from torch_lattice.nn.functional.query import sphashquery

        rows = sphashquery(sphash(target), sphash(source)).to(torch.long)
        clipped = rows.clamp_min(0)
        exact = torch.all(source.index_select(0, clipped) == target, dim=1)
        return torch.where(exact, rows, torch.full_like(rows, -1))

    coordinates = torch.cat((source, target), dim=0)
    unique, inverse = torch.unique(coordinates, dim=0, return_inverse=True)
    source_ids = inverse[: source.shape[0]]
    target_ids = inverse[source.shape[0] :]
    lookup = torch.full(
        (unique.shape[0],),
        -1,
        dtype=torch.long,
        device=source.device,
    )
    lookup[source_ids] = torch.arange(
        source.shape[0], dtype=torch.long, device=source.device
    )
    return lookup[target_ids]


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
    if lhs.coord_manager is rhs.coord_manager and lhs.coord_key == rhs.coord_key:
        return True
    return (
        lhs.coords.shape == rhs.coords.shape
        and lhs.coords.stride() == rhs.coords.stride()
        and lhs.coords.dtype == rhs.coords.dtype
        and lhs.coords.device == rhs.coords.device
        and lhs.stride == rhs.stride
        and lhs.spatial_range == rhs.spatial_range
        and lhs.batch_counts == rhs.batch_counts
        and (
            lhs.coords.data_ptr() == rhs.coords.data_ptr()
            or torch.equal(lhs.coords, rhs.coords)
        )
    )


def _require_compatible(lhs: SparseTensor, rhs: SparseTensor) -> None:
    if lhs.stride != rhs.stride:
        raise ValueError("sparse tensor strides must match.")
    if lhs.coords.dtype != rhs.coords.dtype:
        raise ValueError("sparse tensor coordinate dtypes must match.")
    if lhs.coords.device != rhs.coords.device:
        raise ValueError("sparse tensor coordinate devices must match.")
    if lhs.feats.device != rhs.feats.device:
        raise ValueError("sparse tensor feature devices must match.")
    if lhs.spatial_range != rhs.spatial_range:
        raise ValueError("sparse tensor spatial ranges must match.")
    if (
        lhs.batch_counts is not None
        and rhs.batch_counts is not None
        and len(lhs.batch_counts) != len(rhs.batch_counts)
    ):
        raise ValueError("sparse tensor batch cardinality must match.")


def _replace_sparse(source: SparseTensor, feats: torch.Tensor) -> SparseTensor:
    return source.replace(feats=feats)


def _new_sparse(
    source: SparseTensor,
    *,
    coords: torch.Tensor,
    feats: torch.Tensor,
) -> SparseTensor:
    if coords is source.coords:
        return source.replace(feats=feats)
    return source.with_coordinates(feats=feats, coords=coords)


def _validate_join(join: str) -> None:
    if join not in {"inner", "left", "right", "outer"}:
        raise ValueError("join must be 'inner', 'left', 'right', or 'outer'.")
