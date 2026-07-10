from typing import Callable

import torch

from torch_lattice import SparseTensor

__all__ = ["fapply"]


def fapply(
    input: SparseTensor, fn: Callable[..., torch.Tensor], *args, **kwargs
) -> SparseTensor:
    feats = fn(input.feats, *args, **kwargs)
    return input.replace(feats=feats)
