import torch

import torch_lattice.backend

__all__ = ["convert_transposed_out_in_map", "sphashquery"]


def sphashquery(queries: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """Return the row index of every query hash, or ``-1`` when absent."""

    queries = queries.contiguous()
    references = references.contiguous()
    if queries.dtype != torch.int64 or references.dtype != torch.int64:
        raise TypeError("hash queries and references must use int64 dtype")
    if queries.device != references.device:
        raise ValueError("hash queries and references must use the same device")

    sizes = queries.size()
    queries = queries.view(-1)

    if queries.device.type == "cuda":
        capacity = max(2, 2 * references.shape[0])
        hashmap_keys = torch.zeros(
            capacity, dtype=torch.int64, device=references.device
        )
        hashmap_vals = torch.zeros(
            capacity, dtype=torch.int32, device=references.device
        )
        hashmap = torch_lattice.backend.GPUHashTable(hashmap_keys, hashmap_vals)
        hashmap.insert_vals(references)
        output = hashmap.lookup_vals(queries)[: queries.shape[0]]
    elif queries.device.type == "cpu":
        indices = torch.arange(len(references), device=queries.device, dtype=torch.long)
        output = torch_lattice.backend.hash_query_cpu(queries, references, indices)
    else:
        device = queries.device
        indices = torch.arange(len(references), device=queries.device, dtype=torch.long)
        output = torch_lattice.backend.hash_query_cpu(
            queries.cpu(), references.cpu(), indices.cpu()
        ).to(device)

    output = (output - 1).view(*sizes)
    return output


def convert_transposed_out_in_map(
    out_in_map: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Invert an int32 output-to-input relation for transposed convolution."""

    if out_in_map.dtype != torch.int32 or out_in_map.ndim != 2:
        raise ValueError("out_in_map must be a rank-2 int32 tensor")
    if out_in_map.device.type != "cuda":
        raise ValueError("transposed relation conversion requires CUDA")
    if size < 0:
        raise ValueError("transposed relation size must be non-negative")
    output = torch.full(
        (size, out_in_map.shape[1]),
        -1,
        device=out_in_map.device,
        dtype=torch.int32,
    )
    torch_lattice.backend.convert_transposed_out_in_map(out_in_map.contiguous(), output)
    return output
