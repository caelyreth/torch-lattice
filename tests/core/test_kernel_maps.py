from __future__ import annotations

import pytest
import torch

import torch_lattice
from torch_lattice.nn.functional.conv.kmap.downsample import spdownsample
from torch_lattice.nn.functional.conv.kmap.upsample import spupsample_generative
from torch_lattice.nn.functional.hash import sphash

pytestmark = [pytest.mark.core, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_spdownsample_simple_fast_path_matches_python_fallback(monkeypatch):
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [0, 3, 3, 3],
            [0, 4, 4, 4],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    spatial_range = (5, 5, 5)

    fast = spdownsample(
        coords,
        stride=2,
        kernel_size=2,
        spatial_range=spatial_range,
    )
    monkeypatch.delattr(torch_lattice.backend, "downsample_simple_cuda")
    fallback = spdownsample(
        coords,
        stride=2,
        kernel_size=2,
        spatial_range=spatial_range,
    )

    torch.testing.assert_close(torch.sort(sphash(fast)).values, torch.sort(sphash(fallback)).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_spdownsample_simple_fast_path_skips_negative_coordinate_mode(monkeypatch):
    coords = torch.tensor(
        [[0, -1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    calls = {"fast": 0}

    def fake_fast(*args, **kwargs):
        calls["fast"] += 1
        raise AssertionError("fast path should be skipped")

    monkeypatch.setattr(torch_lattice.backend, "downsample_simple_cuda", fake_fast)
    old = torch_lattice.tensor.get_allow_negative_coordinates()
    torch_lattice.tensor.set_allow_negative_coordinates(True)
    try:
        out = spdownsample(coords, stride=2, kernel_size=2)
    finally:
        torch_lattice.tensor.set_allow_negative_coordinates(old)

    assert calls["fast"] == 0
    assert out.numel() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_spupsample_generative_fast_path_matches_fallback_set():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6]],
        dtype=torch.int32,
        device="cuda",
    )
    spatial_range = (1, 10, 12, 14)

    fast = spupsample_generative(
        coords,
        stride=2,
        kernel_size=2,
        spatial_range=spatial_range,
    )
    fallback = spupsample_generative(
        coords,
        stride=2,
        kernel_size=2,
        padding=(1, 0, 0),
        spatial_range=spatial_range,
    )

    torch.testing.assert_close(torch.sort(sphash(fast)).values, torch.sort(sphash(fallback)).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_spupsample_generative_fast_path_matches_python_fallback(monkeypatch):
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6]],
        dtype=torch.int32,
        device="cuda",
    )
    spatial_range = (1, 10, 12, 14)

    fast = spupsample_generative(
        coords,
        stride=2,
        kernel_size=2,
        spatial_range=spatial_range,
    )
    monkeypatch.delattr(torch_lattice.backend, "upsample_generative_cuda")
    fallback = spupsample_generative(
        coords,
        stride=2,
        kernel_size=2,
        spatial_range=spatial_range,
    )

    fast_order = torch.argsort(sphash(fast))
    fallback_order = torch.argsort(sphash(fallback))
    torch.testing.assert_close(fast[fast_order], fallback[fallback_order])


