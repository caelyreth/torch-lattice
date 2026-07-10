from typing import Any

import numpy as np
import torch

from torch_lattice.operators import (
    DuplicateReduction,
    sparse_from_coordinates,
)
from torch_lattice.tensor import SparseTensor

__all__ = ["sparse_collate", "sparse_collate_fn"]


def sparse_collate(
    inputs: list[SparseTensor],
    *,
    duplicate_reduction: DuplicateReduction = "none",
) -> SparseTensor:
    if not inputs:
        raise ValueError("sparse_collate requires at least one tensor")
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if x.stride != stride:
            raise ValueError("all sparse tensors must have the same stride")

        input_size = x.coords.shape[0]
        batch = torch.full((input_size, 1), k, device=x.coords.device, dtype=torch.int)
        spatial = x.coords[:, 1:] if x.coords.shape[1] == 4 else x.coords
        coords.append(torch.cat((batch, spatial), dim=1))
        feats.append(x.feats)

    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    output = sparse_from_coordinates(
        coords=coords,
        feats=feats,
        stride=stride,
        batch_counts=tuple(int(x.coords.shape[0]) for x in inputs),
        duplicate_reduction=duplicate_reduction,
    )
    return output


def sparse_collate_fn(inputs: list[Any]) -> Any:
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn([input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(input[name]) for input in inputs], dim=0
                )
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([input[name] for input in inputs], dim=0)
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs
