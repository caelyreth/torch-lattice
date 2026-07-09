from __future__ import annotations

import torch

from torch_lattice.nn.functional.hash import sphash
from torch_lattice.nn.functional.query import sphashquery

from torch_lattice_bench.cases.common import F, SparseFixture, sparse_cases
from torch_lattice_bench.harness import BenchmarkCase


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None,
    channels: tuple[int, ...] | None,
    layouts: tuple[str, ...] | None,
    dtype: str,
    device,
) -> tuple[BenchmarkCase, ...]:
    return sparse_cases(
        preset,
        group='hash',
        specs=(
            ('sphash', lambda f: sphash(f.tensor.coords), ('n_in',), None),
            ('kernel_sphash_k27', _kernel_sphash, ('n_in',), None),
            ('sphashquery_self', _sphashquery_self, ('n_in',), None),
            ('spcount_mod4096', _spcount, ('n_in',), None),
        ),
        n_values=n_values,
        channels=channels,
        layouts=layouts,
        dtype=dtype,
        device=device,
    )


def _kernel_sphash(fixture: SparseFixture) -> torch.Tensor:
    offsets = torch.tensor(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=torch.int32,
        device=fixture.tensor.coords.device,
    )
    return sphash(fixture.tensor.coords, offsets)


def _sphashquery_self(fixture: SparseFixture) -> torch.Tensor:
    hashes = sphash(fixture.tensor.coords)
    return sphashquery(hashes, hashes)


def _spcount(fixture: SparseFixture) -> torch.Tensor:
    rows = fixture.tensor.feats.shape[0]
    indices = torch.arange(rows, device=fixture.tensor.coords.device, dtype=torch.int32) % 4096
    return F.spcount(indices, 4096)
