from __future__ import annotations

from collections.abc import Callable

import torch

from torch_lattice.nn.functional.conv.kmap.downsample import spdownsample
from torch_lattice.nn.functional.conv.kmap.upsample import spupsample_generative

from torch_lattice_bench.cases.common import (
    F,
    SparseFixture,
    set_conv_config,
    sparse_cases,
)
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
        group="kmap",
        specs=(
            ("spdownsample_stride2_k2", _downsample, ("n_in",), None),
            ("spupsample_generative_stride2_k2", _upsample, ("n_in",), None),
            (
                "build_kmap_igemm_unsorted_subm_k3",
                _conv_kmap(F.Dataflow.ImplicitGEMM, ifsort=False, split_mask_num=1),
                ("edges",),
                None,
            ),
            (
                "build_kmap_igemm_sorted_subm_k3",
                _conv_kmap(F.Dataflow.ImplicitGEMM, ifsort=True, split_mask_num=3),
                ("edges",),
                None,
            ),
            (
                "build_kmap_fod_fused_subm_k3",
                _conv_kmap(F.Dataflow.FetchOnDemand, FOD_fusion=True),
                ("edges",),
                None,
            ),
            (
                "build_kmap_fod_no_fusion_subm_k3",
                _conv_kmap(F.Dataflow.FetchOnDemand, FOD_fusion=False),
                ("edges",),
                None,
            ),
            (
                "build_kmap_gather_scatter_subm_k3",
                _conv_kmap(F.Dataflow.GatherScatter),
                ("edges",),
                None,
            ),
        ),
        n_values=n_values,
        channels=channels,
        layouts=layouts,
        dtype=dtype,
        device=device,
    )


def _downsample(fixture: SparseFixture) -> torch.Tensor:
    return spdownsample(
        fixture.tensor.coords,
        stride=2,
        kernel_size=2,
        spatial_range=fixture.tensor.spatial_range[1:],
    )


def _upsample(fixture: SparseFixture) -> torch.Tensor:
    return spupsample_generative(
        fixture.tensor.coords,
        stride=2,
        kernel_size=2,
        spatial_range=tuple(value * 2 for value in fixture.tensor.spatial_range),
    )


def _conv_kmap(dataflow: F.Dataflow, **kwargs) -> Callable[[SparseFixture], dict]:
    def run(fixture: SparseFixture) -> dict:
        x = fixture.tensor
        set_conv_config(dataflow, **kwargs)
        cfg = F.conv_config.get_global_conv_config()
        return F.build_kernel_map(
            x.coords,
            x.feats.size(0),
            torch.tensor((3, 3, 3), dtype=torch.int, device=x.coords.device),
            torch.tensor((1, 1, 1), dtype=torch.int, device=x.coords.device),
            torch.tensor((1, 1, 1), dtype=torch.int, device=x.coords.device),
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

    return run
