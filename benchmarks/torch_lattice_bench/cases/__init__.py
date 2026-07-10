from __future__ import annotations

import torch

from torch_lattice_bench.catalog import GROUPS
from torch_lattice_bench.harness import BenchmarkCase

from . import conv, dense, hash, kmap, nn, tensor, train


def all_cases(
    preset: str,
    *,
    groups: tuple[str, ...] = GROUPS,
    n_values: tuple[int, ...] | None = None,
    channels: tuple[int, ...] | None = None,
    layouts: tuple[str, ...] | None = None,
    dtype: str = "fp16",
    device: torch.device | None = None,
) -> tuple[BenchmarkCase, ...]:
    selected: list[BenchmarkCase] = []
    resolved_device = device or torch.device("cuda")
    for group in groups:
        if group == "tensor":
            selected.extend(
                tensor.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    layouts=layouts,
                    dtype=dtype,
                    device=resolved_device,
                )
            )
        elif group == "hash":
            selected.extend(
                hash.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    layouts=layouts,
                    dtype=dtype,
                    device=resolved_device,
                )
            )
        elif group == "dense":
            selected.extend(
                dense.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    layouts=layouts,
                    dtype=dtype,
                    device=resolved_device,
                )
            )
        elif group == "kmap":
            selected.extend(
                kmap.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    layouts=layouts,
                    dtype=dtype,
                    device=resolved_device,
                )
            )
        elif group == "conv":
            selected.extend(
                conv.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    layouts=layouts,
                    dtype=dtype,
                    device=resolved_device,
                )
            )
        elif group == "nn":
            selected.extend(
                nn.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    layouts=layouts,
                    dtype=dtype,
                    device=resolved_device,
                )
            )
        elif group == "train":
            selected.extend(
                train.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    layouts=layouts,
                    dtype=dtype,
                    device=resolved_device,
                )
            )
        else:
            raise ValueError(f"unknown benchmark group: {group}")
    return tuple(selected)


__all__ = ["GROUPS", "all_cases"]
