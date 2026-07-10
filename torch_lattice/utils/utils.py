from itertools import repeat
from collections.abc import Sequence
from functools import lru_cache
import torch

__all__ = ["make_ntuple", "make_tensor", "make_divisible"]


def make_ntuple(x: int | Sequence[int] | torch.Tensor, ndim: int) -> tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    else:
        x = tuple(int(value) for value in x)
    if len(x) != ndim:
        raise ValueError(f"expected {ndim} values, got {len(x)}")
    return tuple(int(value) for value in x)


@lru_cache()
def make_tensor(x: tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def make_divisible(x: int, divisor: int):
    return (x + divisor - 1) // divisor * divisor
