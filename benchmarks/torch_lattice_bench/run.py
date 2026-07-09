#!/usr/bin/env python3
"""Benchmark TorchLattice hot-path sparse tensor operations.

The default input size is 600K active coordinates.  Use ``--smoke`` for a quick
correctness/runability check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F
from torch_lattice.nn.functional.conv.kmap.downsample import spdownsample
from torch_lattice.nn.functional.conv.kmap.upsample import spupsample_generative
from torch_lattice.nn.functional.devoxelize import calc_ti_weights
from torch_lattice.nn.functional.devoxelize import spdevoxelize
from torch_lattice.nn.functional.hash import sphash
from torch_lattice.nn.functional.query import sphashquery
from torch_lattice.nn.functional.voxelize import spvoxelize
from torch_lattice.operators import cat, generative_add
from torch_lattice.utils import to_dense


PATTERNS = ("isolated", "line", "plane", "block2", "block3", "block5", "block8", "grid")
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


@dataclass
class BenchResult:
    pattern: str
    op: str
    dtype: str
    points: int
    channels: int
    warmup: int
    iters: int
    mean_ms: float | None
    median_ms: float | None
    p90_ms: float | None
    min_ms: float | None
    max_ms: float | None
    memory_mb: float | None
    output_points: int | None = None
    skipped: bool = False
    notes: str = ""


class CudaTimer:
    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __call__(self, fn: Callable[[], object], warmup: int, iters: int) -> list[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            self.start.record()
            out = fn()
            self.end.record()
            torch.cuda.synchronize()
            # Keep the result live until after synchronization.
            if isinstance(out, torch.Tensor) and out.numel() == -1:
                raise RuntimeError("unreachable")
            times.append(self.start.elapsed_time(self.end))
        return times


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return sorted(values)[idx]


def make_result(
    *,
    pattern: str,
    op: str,
    dtype_name: str,
    points: int,
    channels: int,
    warmup: int,
    iters: int,
    times: list[float],
    output_points: int | None = None,
    notes: str = "",
) -> BenchResult:
    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return BenchResult(
        pattern=pattern,
        op=op,
        dtype=dtype_name,
        points=points,
        channels=channels,
        warmup=warmup,
        iters=iters,
        mean_ms=statistics.fmean(times),
        median_ms=statistics.median(times),
        p90_ms=percentile(times, 0.90),
        min_ms=min(times),
        max_ms=max(times),
        memory_mb=memory_mb,
        output_points=output_points,
        skipped=False,
        notes=notes,
    )


def make_skip_result(
    *,
    pattern: str,
    op: str,
    dtype_name: str,
    points: int,
    channels: int,
    warmup: int,
    iters: int,
    notes: str,
) -> BenchResult:
    return BenchResult(
        pattern=pattern,
        op=op,
        dtype=dtype_name,
        points=points,
        channels=channels,
        warmup=warmup,
        iters=iters,
        mean_ms=None,
        median_ms=None,
        p90_ms=None,
        min_ms=None,
        max_ms=None,
        memory_mb=None,
        output_points=None,
        skipped=True,
        notes=notes,
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _trim(coords: torch.Tensor, n: int) -> torch.Tensor:
    if coords.size(0) < n:
        raise ValueError(f"pattern produced {coords.size(0)} points, expected {n}")
    return coords[:n].contiguous()


def make_coords(pattern: str, n: int, device: torch.device) -> torch.Tensor:
    """Return int32 coordinates in TorchLattice order [batch, x, y, z]."""
    if pattern == "isolated":
        i = torch.arange(n, device=device, dtype=torch.int64)
        # Widely spaced points minimize local neighbor hits.
        coords = torch.stack(
            [
                torch.zeros_like(i),
                i * 17,
                (i * 37) % (n * 19 + 97),
                (i * 97) % (n * 23 + 193),
            ],
            dim=1,
        )
        return coords.int()

    if pattern == "line":
        i = torch.arange(n, device=device, dtype=torch.int64)
        coords = torch.stack([torch.zeros_like(i), i, torch.zeros_like(i), torch.zeros_like(i)], dim=1)
        return coords.int()

    if pattern == "plane":
        side = _ceil_div(n, int(math.sqrt(n)))
        x = torch.arange(side, device=device, dtype=torch.int64)
        y = torch.arange(side, device=device, dtype=torch.int64)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        zeros = torch.zeros_like(xx.reshape(-1))
        coords = torch.stack([zeros, xx.reshape(-1), yy.reshape(-1), zeros], dim=1)
        return _trim(coords, n).int()

    if pattern.startswith("block"):
        block = int(pattern.replace("block", ""))
        blocks_needed = _ceil_div(n, block**3)
        grid_side = math.ceil(blocks_needed ** (1 / 3))
        b = torch.arange(grid_side, device=device, dtype=torch.int64)
        bz, by, bx = torch.meshgrid(b, b, b, indexing="ij")
        base = torch.stack([bx.reshape(-1), by.reshape(-1), bz.reshape(-1)], dim=1)[:blocks_needed]
        offsets = torch.arange(block, device=device, dtype=torch.int64)
        oz, oy, ox = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
        offs = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=1)
        xyz = base.repeat_interleave(block**3, dim=0) * (block + 1) + offs.repeat(blocks_needed, 1)
        batch = torch.zeros((xyz.size(0), 1), device=device, dtype=torch.int64)
        coords = torch.cat([batch, xyz], dim=1)
        return _trim(coords, n).int()

    if pattern == "grid":
        side = math.ceil(n ** (1 / 3))
        axis = torch.arange(side, device=device, dtype=torch.int64)
        z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
        coords = torch.stack(
            [torch.zeros_like(x.reshape(-1)), x.reshape(-1), y.reshape(-1), z.reshape(-1)],
            dim=1,
        )
        return _trim(coords, n).int()

    raise ValueError(f"unknown pattern: {pattern}")


def make_case(pattern: str, points: int, channels: int, dtype: torch.dtype, device: torch.device) -> SparseTensor:
    coords = make_coords(pattern, points, device)
    feats = torch.randn(points, channels, dtype=dtype, device=device)
    spatial_range = tuple(int(coords[:, i].max().item()) + 1 for i in range(4))
    return SparseTensor(feats=feats, coords=coords, spatial_range=spatial_range)


def clone_sparse(x: SparseTensor) -> SparseTensor:
    out = SparseTensor(feats=x.feats.clone(), coords=x.coords, stride=x.stride, spatial_range=x.spatial_range)
    out._caches = x._caches
    return out


def set_conv_config(
    dataflow: F.Dataflow,
    *,
    ifsort: bool = False,
    split_mask_num: int = 1,
    wgrad_split_k: int | str = "auto",
    FOD_fusion: bool = True,
    IGEMM_center_only: bool = False,
) -> None:
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = dataflow
    config.kmap_mode = "hashmap_on_the_fly"
    config.ifsort = ifsort
    config.split_mask_num = split_mask_num
    config.split_mask_num_bwd = 3
    config.wgrad_split_k = wgrad_split_k
    config.FOD_fusion = FOD_fusion
    config.IGEMM_center_only = IGEMM_center_only
    F.conv_config.set_global_conv_config(config)


def resolved_wgrad_split_label(value: int | str, *, ifsort: bool, pattern: str) -> str:
    if value != "auto":
        return str(value)
    if ifsort:
        return "auto32"
    if pattern == "line":
        return "auto64"
    if pattern == "plane":
        return "auto16"
    return "auto8"


def conv_module(channels: int, dtype: torch.dtype, device: torch.device, kernel_size: int, stride: int = 1) -> spnn.Conv3d:
    module = spnn.Conv3d(channels, channels, kernel_size=kernel_size, stride=stride, bias=False).to(device)
    if dtype == torch.float16:
        module = module.half()
    module.eval()
    return module


def benchmark_tensor_ops(x: SparseTensor, timer: CudaTimer, args, dtype_name: str, pattern: str) -> list[BenchResult]:
    results = []
    points, channels = x.feats.shape
    twin = SparseTensor(x.feats * 0.5, x.coords.clone(), x.stride, x.spatial_range)
    shifted = SparseTensor(x.feats * 0.25, x.coords.clone(), x.stride, x.spatial_range)
    shifted.coords[:, 1] += 1
    batch_norm = spnn.BatchNorm(channels).to(x.feats.device, dtype=x.feats.dtype).eval()
    group_norm = spnn.GroupNorm(num_groups=max(1, min(8, channels)), num_channels=channels).to(
        x.feats.device, dtype=x.feats.dtype
    ).eval()

    ops: list[tuple[str, Callable[[], object], str]] = [
        ("sparse_tensor_construct", lambda: SparseTensor(x.feats, x.coords, x.stride, x.spatial_range), ""),
        ("sparse_tensor_to_device_noop", lambda: x.to(x.feats.device), ""),
        ("sparse_tensor_half", lambda: clone_sparse(x).half(), ""),
        ("cat_features", lambda: cat([x, twin]), "feature concat, same coords"),
        ("generative_add_overlap", lambda: generative_add(x, twin), "same coords"),
        ("generative_add_shifted", lambda: generative_add(x, shifted), "50-100% union depending pattern"),
        ("global_avg_pool", lambda: F.global_avg_pool(x), ""),
        ("global_max_pool", lambda: F.global_max_pool(x), ""),
        ("crop_center_half", lambda: F.spcrop(x, coords_min=(0, 0, 0), coords_max=tuple(max(1, int(x.coords[:, i].max().item()) // 2) for i in range(1, 4))), ""),
        ("relu", lambda: F.relu(clone_sparse(x), inplace=False), ""),
        ("silu", lambda: F.silu(clone_sparse(x), inplace=False), ""),
        ("leaky_relu", lambda: F.leaky_relu(clone_sparse(x), inplace=False), ""),
        ("batch_norm", lambda: batch_norm(x), "module wrapper over feature tensor"),
        ("group_norm", lambda: group_norm(x), "per-batch sparse feature normalization"),
    ]
    for op, fn, notes in ops:
        torch.cuda.reset_peak_memory_stats()
        times = timer(fn, args.warmup, args.iters)
        results.append(make_result(pattern=pattern, op=op, dtype_name=dtype_name, points=points, channels=channels, warmup=args.warmup, iters=args.iters, times=times, notes=notes))
    return results


def benchmark_hash_ops(x: SparseTensor, timer: CudaTimer, args, dtype_name: str, pattern: str) -> list[BenchResult]:
    results = []
    points, channels = x.feats.shape
    offsets = torch.tensor(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=torch.int32,
        device=x.coords.device,
    )
    hashes = sphash(x.coords)

    ops: list[tuple[str, Callable[[], object], str]] = [
        ("sphash", lambda: sphash(x.coords), ""),
        ("kernel_sphash_k27", lambda: sphash(x.coords, offsets), "27 offsets"),
        ("sphashquery_self", lambda: sphashquery(hashes, hashes), ""),
        ("spcount_mod4096", lambda: F.spcount((torch.arange(points, device=x.coords.device, dtype=torch.int32) % 4096), 4096), "atomic count"),
    ]
    for op, fn, notes in ops:
        torch.cuda.reset_peak_memory_stats()
        times = timer(fn, args.warmup, args.iters)
        results.append(make_result(pattern=pattern, op=op, dtype_name=dtype_name, points=points, channels=channels, warmup=args.warmup, iters=args.iters, times=times, notes=notes))
    return results


def benchmark_dense_voxel_ops(x: SparseTensor, timer: CudaTimer, args, dtype_name: str, pattern: str) -> list[BenchResult]:
    results = []
    points, channels = x.feats.shape
    device = x.feats.device
    dense_range = x.spatial_range
    dense_elements = math.prod(dense_range) * channels

    voxel_bins = max(1, points // 4)
    voxel_idx = torch.arange(points, device=device, dtype=torch.int32) % voxel_bins
    counts = F.spcount(voxel_idx, voxel_bins)
    devox_idx = torch.randint(0, points, (points, 8), dtype=torch.int32, device=device)
    weights = torch.rand(points, 8, dtype=x.feats.dtype, device=device)
    weights /= weights.sum(dim=1, keepdim=True)

    ops: list[tuple] = [
        (
            "to_dense_forward",
            lambda: to_dense(x.feats, x.coords, dense_range),
            f"range={dense_range}, dense_elements={dense_elements}",
            dense_elements <= args.max_dense_elements,
        ),
        ("spvoxelize_forward", lambda: spvoxelize(x.feats, voxel_idx, counts), f"bins={voxel_bins}"),
        ("spdevoxelize_forward", lambda: spdevoxelize(x.feats, devox_idx, weights), ""),
        ("calc_ti_weights", lambda: calc_ti_weights(x.coords[:, 1:].float() + 0.25, devox_idx), ""),
    ]
    for item in ops:
        if len(item) == 4:
            op, fn, notes, enabled = item
        else:
            op, fn, notes = item
            enabled = True
        if not enabled:
            results.append(
                make_skip_result(
                    pattern=pattern,
                    op=op,
                    dtype_name=dtype_name,
                    points=points,
                    channels=channels,
                    warmup=args.warmup,
                    iters=args.iters,
                    notes=f"skipped: dense_elements={dense_elements} exceeds --max-dense-elements={args.max_dense_elements}",
                )
            )
            continue
        torch.cuda.reset_peak_memory_stats()
        times = timer(fn, args.warmup, args.iters)
        results.append(make_result(pattern=pattern, op=op, dtype_name=dtype_name, points=points, channels=channels, warmup=args.warmup, iters=args.iters, times=times, notes=notes))
    return results


def benchmark_kmap_ops(x: SparseTensor, timer: CudaTimer, args, dtype_name: str, pattern: str) -> list[BenchResult]:
    results = []
    points, channels = x.feats.shape
    kernel3 = torch.tensor((3, 3, 3), dtype=torch.int, device=x.coords.device)
    stride1 = torch.tensor((1, 1, 1), dtype=torch.int, device=x.coords.device)
    padding1 = torch.tensor((1, 1, 1), dtype=torch.int, device=x.coords.device)

    def conv_kmap_fn(dataflow: F.Dataflow, **kwargs) -> Callable[[], dict]:
        def fn() -> dict:
            set_conv_config(dataflow, **kwargs)
            cfg = F.conv_config.get_global_conv_config()
            return F.build_kernel_map(
                x.coords,
                x.feats.size(0),
                kernel3,
                stride1,
                padding1,
                None,
                None,
                x.spatial_range,
                cfg.kmap_mode,
                cfg.dataflow,
                downsample_mode=cfg.downsample_mode,
                training=False,
                ifsort=cfg.ifsort,
                split_mask_num=cfg.split_mask_num,
                split_mask_num_bwd=cfg.split_mask_num_bwd,
                FOD_fusion=cfg.FOD_fusion,
                IGEMM_center_only=cfg.get("IGEMM_center_only", False),
                inference=True,
            )

        return fn

    ops: list[tuple[str, Callable[[], object], str]] = [
        ("spdownsample_stride2_k2", lambda: spdownsample(x.coords, stride=2, kernel_size=2, spatial_range=x.spatial_range[1:]), ""),
        ("spupsample_generative_stride2_k2", lambda: spupsample_generative(x.coords, stride=2, kernel_size=2, spatial_range=tuple(v * 2 for v in x.spatial_range)), ""),
        ("build_kmap_igemm_unsorted_subm_k3", conv_kmap_fn(F.Dataflow.ImplicitGEMM, ifsort=False, split_mask_num=1), "conv3 submanifold kmap"),
        ("build_kmap_igemm_sorted_subm_k3", conv_kmap_fn(F.Dataflow.ImplicitGEMM, ifsort=True, split_mask_num=3), "conv3 sorted submanifold kmap"),
        ("build_kmap_fod_fused_subm_k3", conv_kmap_fn(F.Dataflow.FetchOnDemand, FOD_fusion=True), "conv3 FOD fused metadata"),
        ("build_kmap_fod_no_fusion_subm_k3", conv_kmap_fn(F.Dataflow.FetchOnDemand, FOD_fusion=False), "conv3 FOD no-fusion metadata"),
        ("build_kmap_gather_scatter_subm_k3", conv_kmap_fn(F.Dataflow.GatherScatter), "conv3 gather-scatter metadata"),
    ]
    for op, fn, notes in ops:
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        torch.cuda.synchronize()
        if isinstance(out, torch.Tensor):
            output_points = int(out.size(0))
        elif isinstance(out, dict) and out.get("coords") is not None:
            output_points = int(out["coords"].size(0))
        else:
            output_points = None
        times = timer(fn, args.warmup, args.iters)
        results.append(make_result(pattern=pattern, op=op, dtype_name=dtype_name, points=points, channels=channels, warmup=args.warmup, iters=args.iters, times=times, output_points=output_points, notes=notes))
    return results


def benchmark_conv_ops(x: SparseTensor, timer: CudaTimer, args, dtype_name: str, pattern: str) -> list[BenchResult]:
    results = []
    points, channels = x.feats.shape
    conv_specs = [
        ("conv1x1_matmul", None, None, conv_module(channels, x.feats.dtype, x.feats.device, 1, 1), False, 1),
        ("conv3_implicit_gemm_unsorted", F.Dataflow.ImplicitGEMM, {"ifsort": False, "split_mask_num": 1}, conv_module(channels, x.feats.dtype, x.feats.device, 3, 1), True, 1),
        ("conv3_implicit_gemm_sorted", F.Dataflow.ImplicitGEMM, {"ifsort": True, "split_mask_num": 3}, conv_module(channels, x.feats.dtype, x.feats.device, 3, 1), True, 1),
        ("conv3_fetch_on_demand_fused", F.Dataflow.FetchOnDemand, {"FOD_fusion": True}, conv_module(channels, x.feats.dtype, x.feats.device, 3, 1), True, 1),
        ("conv3_fetch_on_demand_no_fusion", F.Dataflow.FetchOnDemand, {"FOD_fusion": False}, conv_module(channels, x.feats.dtype, x.feats.device, 3, 1), True, 1),
        ("conv3_gather_scatter", F.Dataflow.GatherScatter, {"ifsort": False, "split_mask_num": 1}, conv_module(channels, x.feats.dtype, x.feats.device, 3, 1), True, 1),
        ("conv2_stride2_implicit", F.Dataflow.ImplicitGEMM, {"ifsort": True, "split_mask_num": 2}, conv_module(channels, x.feats.dtype, x.feats.device, 2, 2), True, 2),
    ]
    if pattern == "isolated":
        conv_specs[3:3] = [
            ("conv3_implicit_gemm_unsorted_center_optin", F.Dataflow.ImplicitGEMM, {"ifsort": False, "split_mask_num": 1, "IGEMM_center_only": True}, conv_module(channels, x.feats.dtype, x.feats.device, 3, 1), True, 1),
            ("conv3_implicit_gemm_sorted_center_optin", F.Dataflow.ImplicitGEMM, {"ifsort": True, "split_mask_num": 3, "IGEMM_center_only": True}, conv_module(channels, x.feats.dtype, x.feats.device, 3, 1), True, 1),
        ]
    for op, dataflow, kwargs, module, include_warm, min_spatial_extent in conv_specs:
        if min(x.spatial_range[1:]) < min_spatial_extent:
            results.append(
                make_skip_result(
                    pattern=pattern,
                    op=f"{op}_cold",
                    dtype_name=dtype_name,
                    points=points,
                    channels=channels,
                    warmup=args.warmup,
                    iters=args.iters,
                    notes=(
                        "skipped: spatial range "
                        f"{x.spatial_range[1:]} is too thin for kernel extent {min_spatial_extent}"
                    ),
                )
            )
            if include_warm:
                results.append(
                    make_skip_result(
                        pattern=pattern,
                        op=f"{op}_warm",
                        dtype_name=dtype_name,
                        points=points,
                        channels=channels,
                        warmup=args.warmup,
                        iters=args.iters,
                        notes=(
                            "skipped: spatial range "
                            f"{x.spatial_range[1:]} is too thin for kernel extent {min_spatial_extent}"
                        ),
                    )
                )
            continue

        if dataflow is not None:
            set_conv_config(dataflow, **kwargs)

        def cold_fn(module=module) -> SparseTensor:
            inp = SparseTensor(x.feats, x.coords, x.stride, x.spatial_range)
            with torch.no_grad():
                return module(inp)

        torch.cuda.reset_peak_memory_stats()
        out = cold_fn()
        torch.cuda.synchronize()
        output_points = int(out.feats.size(0))
        times = timer(cold_fn, args.warmup, args.iters)
        results.append(
            make_result(
                pattern=pattern,
                op=f"{op}_cold",
                dtype_name=dtype_name,
                points=points,
                channels=channels,
                warmup=args.warmup,
                iters=args.iters,
                times=times,
                output_points=output_points,
                notes="fresh SparseTensor cache per iteration",
            )
        )

        if not include_warm:
            continue

        warm_input = SparseTensor(x.feats, x.coords, x.stride, x.spatial_range)
        with torch.no_grad():
            module(warm_input)
        torch.cuda.synchronize()

        def warm_fn(module=module, warm_input=warm_input) -> SparseTensor:
            with torch.no_grad():
                return module(warm_input)

        torch.cuda.reset_peak_memory_stats()
        out = warm_fn()
        torch.cuda.synchronize()
        output_points = int(out.feats.size(0))
        times = timer(warm_fn, args.warmup, args.iters)
        results.append(
            make_result(
                pattern=pattern,
                op=f"{op}_warm",
                dtype_name=dtype_name,
                points=points,
                channels=channels,
                warmup=args.warmup,
                iters=args.iters,
                times=times,
                output_points=output_points,
                notes="reuses SparseTensor kernel-map cache",
            )
        )

    if min(x.spatial_range[1:]) < 2:
        for op in ("conv2_stride2_then_conv3_reuse_hashmap", "conv2_stride2_then_conv3_clear_hashmap"):
            results.append(
                make_skip_result(
                    pattern=pattern,
                    op=op,
                    dtype_name=dtype_name,
                    points=points,
                    channels=channels,
                    warmup=args.warmup,
                    iters=args.iters,
                    notes=(
                        "skipped: spatial range "
                        f"{x.spatial_range[1:]} is too thin for stride-2 chain"
                    ),
                )
            )
    else:
        set_conv_config(F.Dataflow.ImplicitGEMM, ifsort=False, split_mask_num=1)
        stride2_module = conv_module(channels, x.feats.dtype, x.feats.device, 2, 2)
        subm_module = conv_module(channels, x.feats.dtype, x.feats.device, 3, 1)

        def chain_fn(clear_hashmap: bool) -> SparseTensor:
            inp = SparseTensor(x.feats, x.coords, x.stride, x.spatial_range)
            with torch.no_grad():
                mid = stride2_module(inp)
            if clear_hashmap:
                mid._caches.hashmaps.clear()
            with torch.no_grad():
                return subm_module(mid)

        for op, clear_hashmap in (
            ("conv2_stride2_then_conv3_reuse_hashmap", False),
            ("conv2_stride2_then_conv3_clear_hashmap", True),
        ):
            torch.cuda.reset_peak_memory_stats()
            out = chain_fn(clear_hashmap)
            torch.cuda.synchronize()
            output_points = int(out.feats.size(0))
            times = timer(lambda clear_hashmap=clear_hashmap: chain_fn(clear_hashmap), args.warmup, args.iters)
            results.append(
                make_result(
                    pattern=pattern,
                    op=op,
                    dtype_name=dtype_name,
                    points=points,
                    channels=channels,
                    warmup=args.warmup,
                    iters=args.iters,
                    times=times,
                    output_points=output_points,
                    notes="two-layer hot path: stride-2 convolution followed by cached submanifold convolution",
                )
            )
    return results


def benchmark_training_ops(x: SparseTensor, timer: CudaTimer, args, dtype_name: str, pattern: str) -> list[BenchResult]:
    if args.no_backward:
        return []
    points, channels = x.feats.shape
    results = []

    for ifsort, split_mask_num, label in (
        (False, 1, "unsorted"),
        (True, 3, "sorted"),
    ):
        set_conv_config(
            F.Dataflow.ImplicitGEMM,
            ifsort=ifsort,
            split_mask_num=split_mask_num,
            wgrad_split_k=args.wgrad_split_k,
        )
        module = conv_module(channels, x.feats.dtype, x.feats.device, 3, 1)
        module.train()

        def conv_forward_backward(module=module) -> object:
            module.zero_grad(set_to_none=True)
            feats = x.feats.detach().clone().requires_grad_(True)
            inp = SparseTensor(feats, x.coords, x.stride, x.spatial_range)
            out = module(inp).feats
            loss = out.float().square().mean()
            loss.backward()
            return loss

        torch.cuda.reset_peak_memory_stats()
        times = timer(conv_forward_backward, args.warmup, args.iters)
        results.append(
            make_result(
                pattern=pattern,
                op=(
                    f"conv3_implicit_{label}_wgrad_split"
                    f"{resolved_wgrad_split_label(args.wgrad_split_k, ifsort=ifsort, pattern=pattern)}_forward_backward"
                ),
                dtype_name=dtype_name,
                points=points,
                channels=channels,
                warmup=args.warmup,
                iters=args.iters,
                times=times,
                notes="module.train() builds backward kernel maps",
            )
        )

    set_conv_config(F.Dataflow.FetchOnDemand, FOD_fusion=False)
    module = conv_module(channels, x.feats.dtype, x.feats.device, 3, 1)
    module.train()
    warm_input = SparseTensor(
        x.feats.detach(), x.coords, x.stride, x.spatial_range
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Fetch_On_Demand does not have backward kernels.*",
            category=UserWarning,
        )
        module(warm_input)
        torch.cuda.synchronize()

        def fod_forward_backward() -> object:
            module.zero_grad(set_to_none=True)
            feats = x.feats.detach().clone().requires_grad_(True)
            inp = SparseTensor(feats, x.coords, x.stride, x.spatial_range)
            inp._caches = warm_input._caches
            out = module(inp).feats
            loss = out.float().square().mean()
            loss.backward()
            return loss

        torch.cuda.reset_peak_memory_stats()
        times = timer(fod_forward_backward, args.warmup, args.iters)
    results.append(
        make_result(
            pattern=pattern,
            op="conv3_fetch_on_demand_no_fusion_forward_backward",
            dtype_name=dtype_name,
            points=points,
            channels=channels,
            warmup=args.warmup,
            iters=args.iters,
            times=times,
            notes="cached FOD kmap; center-only cases use matmul backward, others fall back to gather-scatter backward",
        )
    )
    return results


def benchmark_pattern(pattern: str, args, timer: CudaTimer) -> list[BenchResult]:
    dtype = DTYPES[args.dtype]
    x = make_case(pattern, args.points, args.channels, dtype, torch.device(args.device))
    torch.cuda.synchronize()
    results: list[BenchResult] = []
    for group in args.groups:
        print(f"[benchmark]   group={group}", flush=True)
        try:
            if group == "tensor":
                results.extend(benchmark_tensor_ops(x, timer, args, args.dtype, pattern))
            elif group == "hash":
                results.extend(benchmark_hash_ops(x, timer, args, args.dtype, pattern))
            elif group == "dense":
                results.extend(benchmark_dense_voxel_ops(x, timer, args, args.dtype, pattern))
            elif group == "kmap":
                results.extend(benchmark_kmap_ops(x, timer, args, args.dtype, pattern))
            elif group == "conv":
                results.extend(benchmark_conv_ops(x, timer, args, args.dtype, pattern))
            elif group == "train":
                results.extend(benchmark_training_ops(x, timer, args, args.dtype, pattern))
            else:
                raise ValueError(f"unknown group: {group}")
        except Exception as exc:
            results.append(
                make_skip_result(
                    pattern=pattern,
                    op=f"{group}_group",
                    dtype_name=args.dtype,
                    points=int(x.feats.shape[0]),
                    channels=int(x.feats.shape[1]),
                    warmup=args.warmup,
                    iters=args.iters,
                    notes=f"skipped after benchmark failure: {type(exc).__name__}: {exc}",
                )
            )
    return results


def write_results(results: list[BenchResult], output: Path | None) -> None:
    rows = [asdict(r) for r in results]
    payload = {"environment": environment(), "results": rows}
    print(summary_table(results))
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif output.suffix == ".csv":
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("output must end in .json or .csv")


def environment() -> dict[str, object]:
    return {
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_lattice_version": getattr(torch_lattice, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def summary_table(results: list[BenchResult]) -> str:
    if not results:
        return "No benchmark results."
    headers = ["case", "dtype", "points", "channels", "median_ms", "p90_ms", "memory_mb"]
    rows: list[list[str]] = []
    for result in results:
        rows.append([
            f"{result.pattern}/{result.op}",
            result.dtype,
            str(result.points),
            str(result.channels),
            _number(result.median_ms),
            _number(result.p90_ms),
            _number(result.memory_mb),
        ])
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    lines = [
        "  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(row[col].ljust(widths[col]) for col in range(len(row)))
        for row in rows
    )
    return "\n".join(lines)


def _number(value: float | None) -> str:
    if value is None:
        return "skip"
    return f"{value:.3f}"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=600_000)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", "--repeats", dest="iters", type=int, default=30)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS))
    parser.add_argument("--groups", "--group", nargs="+", choices=("tensor", "hash", "dense", "kmap", "conv", "train"), default=["tensor", "hash", "dense", "kmap", "conv", "train"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-dense-elements",
        type=int,
        default=256_000_000,
        help="Skip dense materialization benchmarks whose output element count exceeds this limit.",
    )
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--wgrad-split-k", default="auto")
    parser.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", default=True)
    parser.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false")
    parser.add_argument("--allow-fp16", dest="allow_fp16", action="store_true", default=True)
    parser.add_argument("--no-allow-fp16", dest="allow_fp16", action="store_false")
    parser.add_argument("--smoke", action="store_true", help="Alias for --preset smoke.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke or args.preset == "smoke":
        args.points = min(args.points, 8_192)
        args.channels = min(args.channels, 16)
        args.warmup = 1
        args.iters = 2
        args.patterns = ["isolated", "line", "plane", "block2", "grid"]
        args.groups = ["hash", "conv"]

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This benchmark suite is intended for CUDA hot-path benchmarking.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch_lattice.backends.allow_tf32 = args.allow_tf32
    torch_lattice.backends.allow_fp16 = args.allow_fp16
    torch_lattice.backends.hash_rsv_ratio = max(64, torch_lattice.backends.hash_rsv_ratio)
    torch_lattice.backends.benchmark = True

    timer = CudaTimer()
    results: list[BenchResult] = []
    for pattern in args.patterns:
        print(f"[benchmark] pattern={pattern} points={args.points} channels={args.channels} dtype={args.dtype}", flush=True)
        results.extend(benchmark_pattern(pattern, args, timer))
    write_results(results, args.output)


if __name__ == "__main__":
    main()
