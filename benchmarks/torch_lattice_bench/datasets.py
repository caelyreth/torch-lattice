from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torch_lattice import SparseTensor

SPARSE_LAYOUTS = (
    'isolated',
    'line',
    'plane',
    'grid',
    'block2',
    'block3',
    'block4',
    'block8',
)

DTYPES = {'fp16': torch.float16, 'fp32': torch.float32}


@dataclass(frozen=True, slots=True)
class SparseFixture:
    tensor: SparseTensor
    layout: str
    dtype_name: str

    @property
    def points(self) -> int:
        return int(self.tensor.feats.shape[0])

    @property
    def channels(self) -> int:
        return int(self.tensor.feats.shape[1])


def sparse_fixture(params: dict[str, object], *, device: torch.device) -> SparseFixture:
    points = int(params.get('N', params.get('points', 8192)))
    channels = int(params.get('channels', params.get('channels_in', 16)))
    layout = str(params.get('layout', 'grid'))
    dtype_name = str(params.get('dtype', 'fp16'))
    dtype = DTYPES[dtype_name]
    coords = sparse_coords(layout, points, device)
    generator = torch.Generator(device='cpu').manual_seed(int(params.get('seed', 0)))
    feats = torch.randn((points, channels), generator=generator, dtype=dtype, device='cpu').to(device)
    spatial_range = tuple(int(coords[:, index].max().item()) + 1 for index in range(4))
    return SparseFixture(
        SparseTensor(feats=feats, coords=coords, spatial_range=spatial_range),
        layout,
        dtype_name,
    )


def clone_sparse(x: SparseTensor, *, clone_feats: bool = True) -> SparseTensor:
    out = SparseTensor(
        feats=x.feats.clone() if clone_feats else x.feats,
        coords=x.coords.clone(),
        stride=x.stride,
        spatial_range=x.spatial_range,
    )
    out._caches = x._caches
    return out


def fresh_sparse(x: SparseTensor) -> SparseTensor:
    return SparseTensor(x.feats, x.coords, x.stride, x.spatial_range)


def sparse_coords(layout: str, n: int, device: torch.device) -> torch.Tensor:
    if layout not in SPARSE_LAYOUTS:
        raise ValueError(f'unknown sparse layout: {layout}')
    if layout == 'isolated':
        i = torch.arange(n, device=device, dtype=torch.int64)
        return torch.stack(
            [
                torch.zeros_like(i),
                i * 17,
                (i * 37) % (n * 19 + 97),
                (i * 97) % (n * 23 + 193),
            ],
            dim=1,
        ).int()
    if layout == 'line':
        i = torch.arange(n, device=device, dtype=torch.int64)
        return torch.stack([torch.zeros_like(i), i, torch.zeros_like(i), torch.zeros_like(i)], dim=1).int()
    if layout == 'plane':
        side = _ceil_div(n, int(math.sqrt(n)))
        axis = torch.arange(side, device=device, dtype=torch.int64)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        zeros = torch.zeros_like(xx.reshape(-1))
        coords = torch.stack([zeros, xx.reshape(-1), yy.reshape(-1), zeros], dim=1)
        return _trim(coords, n).int()
    if layout == 'grid':
        side = math.ceil(n ** (1 / 3))
        axis = torch.arange(side, device=device, dtype=torch.int64)
        z, y, x = torch.meshgrid(axis, axis, axis, indexing='ij')
        coords = torch.stack(
            [torch.zeros_like(x.reshape(-1)), x.reshape(-1), y.reshape(-1), z.reshape(-1)],
            dim=1,
        )
        return _trim(coords, n).int()
    block = int(layout.removeprefix('block'))
    blocks_needed = _ceil_div(n, block**3)
    grid_side = math.ceil(blocks_needed ** (1 / 3))
    axis = torch.arange(grid_side, device=device, dtype=torch.int64)
    bz, by, bx = torch.meshgrid(axis, axis, axis, indexing='ij')
    base = torch.stack([bx.reshape(-1), by.reshape(-1), bz.reshape(-1)], dim=1)[:blocks_needed]
    offsets = torch.arange(block, device=device, dtype=torch.int64)
    oz, oy, ox = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    del oz
    offs = torch.stack([ox.reshape(-1), oy.reshape(-1), offsets.repeat_interleave(block * block)[: block**3]], dim=1)
    xyz = base.repeat_interleave(block**3, dim=0) * (block + 1) + offs.repeat(blocks_needed, 1)
    batch = torch.zeros((xyz.size(0), 1), device=device, dtype=torch.int64)
    return _trim(torch.cat([batch, xyz], dim=1), n).int()


def params_matrix(
    preset: str,
    *,
    n_values: tuple[int, ...] | None,
    channels: tuple[int, ...] | None,
    layouts: tuple[str, ...] | None,
    dtype: str,
) -> tuple[dict[str, object], ...]:
    sizes = n_values or _preset_sizes(preset)
    channel_values = channels or _preset_channels(preset)
    layout_values = layouts or _preset_layouts(preset)
    return tuple(
        {'N': n, 'channels': c, 'layout': layout, 'dtype': dtype}
        for n in sizes
        for c in channel_values
        for layout in layout_values
    )


def _preset_sizes(preset: str) -> tuple[int, ...]:
    if preset == 'smoke':
        return (8192,)
    if preset == 'standard':
        return (65_536, 262_144)
    return (65_536, 262_144, 600_000)


def _preset_channels(preset: str) -> tuple[int, ...]:
    if preset == 'smoke':
        return (16,)
    if preset == 'standard':
        return (16, 32)
    return (16, 32, 64)


def _preset_layouts(preset: str) -> tuple[str, ...]:
    if preset == 'smoke':
        return ('isolated', 'line', 'plane', 'block2', 'grid')
    if preset == 'standard':
        return ('isolated', 'line', 'plane', 'block3', 'grid')
    return SPARSE_LAYOUTS


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _trim(coords: torch.Tensor, n: int) -> torch.Tensor:
    if coords.size(0) < n:
        raise ValueError(f'pattern produced {coords.size(0)} points, expected {n}')
    return coords[:n].contiguous()


__all__ = ['DTYPES', 'SPARSE_LAYOUTS', 'SparseFixture', 'clone_sparse', 'fresh_sparse', 'params_matrix', 'sparse_fixture']
