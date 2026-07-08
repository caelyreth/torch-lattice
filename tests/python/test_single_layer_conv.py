from typing import Any, Dict, Tuple, Union, Optional, List

import numpy as np
import pytest
import torch
from torch import nn

import torch_lattice
from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F
from torch_lattice.nn.functional.conv.kmap.downsample import spdownsample
from torch_lattice.nn.functional.conv.kmap.upsample import spupsample_generative
from torch_lattice.nn.functional.hash import sphash
from torch_lattice.operators import generative_add
from torch_lattice.utils import make_ntuple

from .test_utils import *

__all__ = ["test_single_layer_convolution_forward"]


def test_generative_add_shared_coords_uses_sparse_add_fast_path():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    a = torch_lattice.SparseTensor(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        coords,
        spatial_range=(1, 3, 1, 1),
    )
    b = torch_lattice.SparseTensor(
        torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]),
        coords,
        spatial_range=(1, 3, 1, 1),
    )

    out = generative_add(a, b)

    assert out.coords.data_ptr() == coords.data_ptr()
    assert out._caches is a._caches
    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, a.feats + b.feats)


def test_generative_add_equal_cloned_coords_uses_sparse_add_fast_path():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    a = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        coords,
        spatial_range=(1, 3, 1, 1),
    )
    b = torch_lattice.SparseTensor(
        torch.tensor([[4.0], [5.0], [6.0]]),
        coords.clone(),
        spatial_range=(1, 3, 1, 1),
    )

    out = generative_add(a, b)

    assert out.coords.data_ptr() == coords.data_ptr()
    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, a.feats + b.feats)


def test_generative_add_shifted_coords_keeps_union_semantics():
    a = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )
    b = torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0], [30.0]]),
        torch.tensor(
            [[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )

    out = generative_add(a, b)

    expected_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=torch.int32,
    )
    expected_feats = torch.tensor([[1.0], [12.0], [23.0], [30.0]])
    torch.testing.assert_close(out.coords, expected_coords)
    torch.testing.assert_close(out.feats, expected_feats)




def test_sparse_binary_alignment_join_and_fill_semantics():
    lhs = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )
    rhs = torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0], [30.0]]),
        torch.tensor(
            [[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )

    out = torch_lattice.sparse_sub(lhs, rhs, join="left", rhs_fill=1.5)

    expected_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    expected_feats = torch.tensor([[-0.5], [-8.0], [-17.0]])
    torch.testing.assert_close(out.coords, expected_coords)
    torch.testing.assert_close(out.feats, expected_feats)


def test_sparse_cat_outer_aligns_missing_rows_with_zero_features():
    lhs = torch_lattice.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 3, 1, 1),
    )
    rhs = torch_lattice.SparseTensor(
        torch.tensor([[10.0], [20.0]]),
        torch.tensor([[0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 3, 1, 1),
    )

    out = torch_lattice.cat([lhs, rhs], join="outer")

    expected_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    expected_feats = torch.tensor([[1.0, 0.0], [2.0, 10.0], [0.0, 20.0]])
    torch.testing.assert_close(out.coords, expected_coords)
    torch.testing.assert_close(out.feats, expected_feats)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_generative_add_shifted_cuda_matches_reference_by_coordinate():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    feats = torch.arange(12, dtype=torch.float16, device="cuda").reshape(4, 3)
    a = torch_lattice.SparseTensor(feats, coords, spatial_range=(1, 5, 1, 1))
    b = torch_lattice.SparseTensor(
        feats * 0.5,
        coords + torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="cuda"),
        spatial_range=(1, 5, 1, 1),
    )

    out = generative_add(a, b)
    expected_coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0],
            [0, 4, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_feats = torch.stack(
        [
            feats[0],
            feats[1] + feats[0] * 0.5,
            feats[2] + feats[1] * 0.5,
            feats[3] + feats[2] * 0.5,
            feats[3] * 0.5,
        ],
        dim=0,
    )

    out_order = torch.argsort(sphash(out.coords))
    expected_order = torch.argsort(sphash(expected_coords))
    torch.testing.assert_close(out.coords[out_order], expected_coords[expected_order])
    torch.testing.assert_close(out.feats[out_order], expected_feats[expected_order])


def test_group_norm_single_batch_matches_dense_reference():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=torch.int32,
    )
    feats = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tensor = torch_lattice.SparseTensor(
        feats,
        coords,
        spatial_range=(1, 4, 1, 1),
    )
    norm = spnn.GroupNorm(num_groups=2, num_channels=4, affine=True)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 1.5, 2.0, 2.5]))
        norm.bias.copy_(torch.tensor([0.0, 0.25, 0.5, 0.75]))

    out = norm(tensor)
    ref = torch.nn.functional.group_norm(
        feats.t().reshape(1, 4, -1),
        num_groups=2,
        weight=norm.weight,
        bias=norm.bias,
        eps=norm.eps,
    ).reshape(4, -1).t()

    assert out._caches is tensor._caches
    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, ref)


def test_group_norm_multi_batch_matches_per_sample_reference():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    feats = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tensor = torch_lattice.SparseTensor(feats, coords)
    norm = spnn.GroupNorm(num_groups=2, num_channels=4, affine=False)

    out = norm(tensor)
    refs = []
    for batch_id in (0, 1):
        bfeats = feats[coords[:, 0] == batch_id]
        refs.append(
            torch.nn.functional.group_norm(
                bfeats.t().reshape(1, 4, -1),
                num_groups=2,
                weight=None,
                bias=None,
                eps=norm.eps,
            ).reshape(4, -1).t()
        )
    ref = torch.cat(refs, dim=0)

    torch.testing.assert_close(out.coords, coords)
    torch.testing.assert_close(out.feats, ref)


def test_global_pool_single_batch_matches_feature_reduction():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    feats = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]])
    tensor = torch_lattice.SparseTensor(
        feats,
        coords,
        spatial_range=(1, 3, 1, 1),
    )

    torch.testing.assert_close(F.global_avg_pool(tensor), feats.mean(dim=0, keepdim=True))
    torch.testing.assert_close(F.global_max_pool(tensor), feats.max(dim=0, keepdim=True)[0])


def test_global_pool_multi_batch_matches_per_sample_reduction():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    feats = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 8.0], [7.0, 6.0]])
    tensor = torch_lattice.SparseTensor(feats, coords)

    torch.testing.assert_close(
        F.global_avg_pool(tensor),
        torch.tensor([[2.0, 3.0], [6.0, 7.0]]),
    )
    torch.testing.assert_close(
        F.global_max_pool(tensor),
        torch.tensor([[3.0, 4.0], [7.0, 8.0]]),
    )


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


class TestSparseConv(nn.Module):
    __test__ = False

    def __init__(
        self,
        num_layers,
        shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        device,
    ):
        super().__init__()
        layers = [
            spnn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
            )
        ]

        for i in range(1, num_layers):
            layers.append(
                spnn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )
            )
        self.net = nn.Sequential(
            *layers,
        ).to(device)
        self.shape = shape

    def forward(self, feats, coords):
        coords = coords.int()
        ts_tensor = torch_lattice.SparseTensor(feats, coords)
        return self.net(ts_tensor)


class TestTorchConv(nn.Module):
    __test__ = False

    def __init__(
        self,
        num_layers,
        shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        device,
    ):
        super().__init__()
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=False,
            )
        ]

        for i in range(1, num_layers):
            layers.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=False,
                )
            )
        self.net = nn.Sequential(
            *layers,
        ).to(device)
        self.shape = shape

    def forward(self, x):
        return self.net(x)


def check_single_layer_convolution_forward(
    batch_size: int = 1,
    shape: Union[int, Tuple[int, ...]] = 5,
    num_points: int = 20,
    IC: int = 16,
    OC: int = 32,
    kernel_size: int = 3,
    stride: int = 1,
    device="cuda:0",
    is_half=True,
):

    np.random.seed(0)
    torch.manual_seed(0)

    shape = make_ntuple(shape, ndim=3)
    if num_points > np.prod(shape):
        print("Warning: num_points exceeds coords range!")
        print("         reduce num_points to %d!" % np.prod(shape))
        num_points = np.prod(shape)
    num_points = [num_points] * batch_size

    if kernel_size % 2 == 0:
        layer_padding = 0
    else:
        layer_padding = (kernel_size - 1) // 2

    model = TestSparseConv(
        num_layers=1,
        shape=shape,
        in_channels=IC,
        out_channels=OC,
        kernel_size=kernel_size,
        stride=stride,
        padding=layer_padding,
        dilation=1,
        device=device,
    )

    ref_model = TestTorchConv(
        num_layers=1,
        shape=shape,
        in_channels=IC,
        out_channels=OC,
        kernel_size=kernel_size,
        stride=stride,
        padding=layer_padding,
        dilation=1,
        device=device,
    )

    if is_half:
        torch_dtype = torch.float16
        np_dtype = np.float16
        model.half()
        ref_model.half()

    else:
        torch_dtype = torch.float32
        np_dtype = np.float32

    sparse_dict = generate_feature_map(shape, num_points, IC, dtype=np_dtype)

    feats = np.ascontiguousarray(sparse_dict["feats"])
    coords = np.ascontiguousarray(sparse_dict["coords"])
    dense_feats = sparse_dict["dense_feats"]

    # print(feats)
    # print(coords)
    # print(dense_feats)

    coords_t = torch.from_numpy(coords).int().to(device)
    feats_t = torch.from_numpy(feats).to(torch_dtype).to(device)
    dense_feats_t = torch.from_numpy(dense_feats).to(torch_dtype).to(device)

    filters = np.random.uniform(
        -1, 1, size=[kernel_size, kernel_size, kernel_size, IC, OC]
    ).astype(np_dtype)
    filters_t = torch.from_numpy(filters).to(torch_dtype).to(device)

    if kernel_size % 2 == 1:
        ref_model.net[0].weight.data[:] = filters_t.permute(4, 3, 2, 1, 0).contiguous()
    else:
        ref_model.net[0].weight.data[:] = filters_t.permute(4, 3, 0, 1, 2).contiguous()

    model.net[0].kernel.data[:] = filters_t.reshape(-1, IC, OC)

    if kernel_size % 2 == 0:  # manually pad
        dense_feats_t = dense_pad(dense_feats_t, kernel_size)

    ref_out = ref_model(dense_feats_t)
    out = model(feats_t, coords_t)

    ts_coords = out.C
    ts_coords_np = np.asarray(ts_coords.detach().cpu())

    ref_out_np = ref_out.detach().cpu().numpy()
    ref_out_subm_np = dense_to_subm(ref_out_np, ts_coords_np)

    out_dense_np = sparse_tensor_to_dense(out, ref_out_np.shape[2:], OC, dtype=np_dtype)

    # print(out.C)
    # print(out.F)

    # print(ref_out_np)
    # print(out_dense_np)
    mean_adiff = np.sum(np.abs(out_dense_np - ref_out_subm_np)) / ts_coords.shape[0]
    max_adiff = np.max(np.abs(out_dense_np - ref_out_subm_np))
    max_rdiff = max_adiff / np.mean(np.abs(out_dense_np))
    assert mean_adiff <= 5e-2
    assert max_rdiff <= 5e-2
    return mean_adiff, max_rdiff


def test_single_layer_convolution_forward():
    check_single_layer_convolution_forward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_subm_implicit_gemm_prunes_impossible_thin_shape_offsets():
    points = 128
    channels = 4
    coords = torch.stack(
        [
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.arange(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
        ],
        dim=1,
    )
    feats = torch.randn(points, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(
        27, channels, channels, device="cuda", dtype=torch.float16, requires_grad=True
    )
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = F.Dataflow.ImplicitGEMM
    config.kmap_mode = "hashmap_on_the_fly"
    config.ifsort = False

    x = torch_lattice.SparseTensor(
        feats.detach().clone().requires_grad_(True),
        coords,
        spatial_range=(1, points, 1, 1),
    )
    out = F.conv3d(x, weight, 3, padding=1, config=config, training=True, subm=True)
    kmap = next(iter(x._caches.kmaps.values()))
    assert kmap["out_in_map"].shape[1] == 3
    assert kmap["active_kernel_offsets"].detach().cpu().tolist() == [12, 13, 14]

    ref_feats = feats.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref = F.conv3d(
        torch_lattice.SparseTensor(ref_feats, coords, spatial_range=None),
        ref_weight,
        3,
        padding=1,
        config=config.copy(),
        training=True,
        subm=True,
    )
    grad = torch.randn_like(out.feats)
    out.feats.backward(grad)
    ref.feats.backward(grad)

    torch.testing.assert_close(out.feats, ref.feats, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.feats.grad, ref_feats.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_implicit_gemm_conv3d_no_grad_fast_path_matches_autograd_path():
    points = 256
    channels = 8
    coords = torch.stack(
        [
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.arange(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
        ],
        dim=1,
    )
    feats = torch.randn(points, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(27, channels, channels, device="cuda", dtype=torch.float16)
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = F.Dataflow.ImplicitGEMM
    config.kmap_mode = "hashmap_on_the_fly"
    config.ifsort = False

    ref_out = F.conv3d(
        torch_lattice.SparseTensor(feats.clone().requires_grad_(True), coords),
        weight.clone().requires_grad_(True),
        3,
        padding=1,
        config=config.copy(),
        training=False,
    )
    fast_out = F.conv3d(
        torch_lattice.SparseTensor(feats.clone(), coords),
        weight.clone(),
        3,
        padding=1,
        config=config.copy(),
        training=False,
    )

    torch.testing.assert_close(fast_out.coords, ref_out.coords)
    torch.testing.assert_close(fast_out.feats, ref_out.feats, rtol=1e-2, atol=1e-2)
    assert fast_out.feats.grad_fn is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_implicit_gemm_conv3d_no_grad_fast_path_dispatch(monkeypatch):
    points = 16
    channels = 4
    coords = torch.stack(
        [
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.arange(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
        ],
        dim=1,
    )
    feats = torch.randn(points, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(27, channels, channels, device="cuda", dtype=torch.float16)
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = F.Dataflow.ImplicitGEMM
    config.kmap_mode = "hashmap_on_the_fly"
    config.ifsort = False
    seen = {"fast": 0, "apply": 0}

    def fake_fast(input, weight, kmap, config, transposed):
        seen["fast"] += 1
        return torch.zeros(
            kmap["sizes"][1],
            weight.size(-1),
            dtype=weight.dtype,
            device=weight.device,
        )

    def fake_apply(input, weight, kmap, config, transposed):
        seen["apply"] += 1
        return torch.zeros(
            kmap["sizes"][1],
            weight.size(-1),
            dtype=weight.dtype,
            device=weight.device,
        )

    monkeypatch.setattr(F.conv, "implicit_gemm_forward_no_grad", fake_fast)
    monkeypatch.setattr(F.conv.ImplicitGEMMConvolutionFuntion, "apply", fake_apply)

    F.conv3d(
        torch_lattice.SparseTensor(feats, coords),
        weight,
        3,
        padding=1,
        config=config.copy(),
    )
    with torch.no_grad():
        F.conv3d(
            torch_lattice.SparseTensor(feats.clone().requires_grad_(True), coords),
            weight.clone().requires_grad_(True),
            3,
            padding=1,
            config=config.copy(),
        )
    F.conv3d(
        torch_lattice.SparseTensor(feats.clone().requires_grad_(True), coords),
        weight,
        3,
        padding=1,
        config=config.copy(),
    )
    F.conv3d(
        torch_lattice.SparseTensor(feats, coords),
        weight.clone().requires_grad_(True),
        3,
        padding=1,
        config=config.copy(),
    )

    assert seen == {"fast": 2, "apply": 2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dataflow", "fast_name", "function_name", "kwargs"),
    [
        (
            F.Dataflow.FetchOnDemand,
            "fetch_on_demand_forward_no_grad",
            "FetchOnDemandConvolutionFuntion",
            {"FOD_fusion": False},
        ),
        (
            F.Dataflow.GatherScatter,
            "gather_scatter_forward_no_grad",
            "GatherScatterConvolutionFuntion",
            {},
        ),
    ],
)
def test_conv3d_no_grad_fast_path_dispatches_non_igemm_dataflows(
    monkeypatch, dataflow, fast_name, function_name, kwargs
):
    points = 16
    channels = 4
    coords = torch.stack(
        [
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.arange(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
            torch.zeros(points, device="cuda", dtype=torch.int32),
        ],
        dim=1,
    )
    feats = torch.randn(points, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(27, channels, channels, device="cuda", dtype=torch.float16)
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = dataflow
    config.kmap_mode = "hashmap_on_the_fly"
    for key, value in kwargs.items():
        config[key] = value
    seen = {"fast": 0, "apply": 0}

    def fake_fast(input, weight, kmap, config, transposed):
        seen["fast"] += 1
        return torch.zeros(
            kmap["sizes"][1],
            weight.size(-1),
            dtype=weight.dtype,
            device=weight.device,
        )

    def fake_apply(input, weight, kmap, config, transposed):
        seen["apply"] += 1
        return torch.zeros(
            kmap["sizes"][1],
            weight.size(-1),
            dtype=weight.dtype,
            device=weight.device,
        )

    monkeypatch.setattr(F.conv, fast_name, fake_fast)
    monkeypatch.setattr(getattr(F.conv, function_name), "apply", fake_apply)

    with torch.no_grad():
        F.conv3d(
            torch_lattice.SparseTensor(feats.clone().requires_grad_(True), coords),
            weight.clone().requires_grad_(True),
            3,
            padding=1,
            config=config.copy(),
        )
    F.conv3d(
        torch_lattice.SparseTensor(feats.clone().requires_grad_(True), coords),
        weight,
        3,
        padding=1,
        config=config.copy(),
    )

    assert seen == {"fast": 1, "apply": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dataflow", [F.Dataflow.FetchOnDemand, F.Dataflow.GatherScatter])
@pytest.mark.parametrize("spatial_range", [(1, 96, 1, 1), (1, 32, 32, 1)])
def test_non_igemm_no_grad_compact_kmap_matches_full_weight_reference(
    dataflow, spatial_range
):
    channels = 4
    if spatial_range[2] == 1:
        points = spatial_range[1]
        coords = torch.stack(
            [
                torch.zeros(points, device="cuda", dtype=torch.int32),
                torch.arange(points, device="cuda", dtype=torch.int32),
                torch.zeros(points, device="cuda", dtype=torch.int32),
                torch.zeros(points, device="cuda", dtype=torch.int32),
            ],
            dim=1,
        )
    else:
        side = spatial_range[1]
        yy, xx = torch.meshgrid(
            torch.arange(side, device="cuda", dtype=torch.int32),
            torch.arange(side, device="cuda", dtype=torch.int32),
            indexing="ij",
        )
        points = side * side
        coords = torch.stack(
            [
                torch.zeros(points, device="cuda", dtype=torch.int32),
                xx.reshape(-1),
                yy.reshape(-1),
                torch.zeros(points, device="cuda", dtype=torch.int32),
            ],
            dim=1,
        )

    feats = torch.randn(points, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(27, channels, channels, device="cuda", dtype=torch.float16)
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = dataflow
    config.kmap_mode = "hashmap_on_the_fly"
    config.ifsort = False
    config.FOD_fusion = False

    compact = F.conv3d(
        torch_lattice.SparseTensor(feats.clone(), coords, spatial_range=spatial_range),
        weight,
        3,
        padding=1,
        config=config.copy(),
        subm=True,
    )
    ref = F.conv3d(
        torch_lattice.SparseTensor(
            feats.clone().requires_grad_(True),
            coords,
            spatial_range=None,
        ),
        weight.clone().requires_grad_(True),
        3,
        padding=1,
        config=config.copy(),
        subm=True,
    )

    torch.testing.assert_close(compact.coords, ref.coords)
    torch.testing.assert_close(compact.feats, ref.feats, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_fod_compactor_matches_reference_layout():
    out_in_map = torch.tensor(
        [
            [0, -1, 2],
            [-1, 1, 3],
            [2, -1, -1],
            [3, 4, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    nbmaps, nbsizes, nbaddrs, qnbaddrs = torch_lattice.backend.compact_out_in_map_fod(
        out_in_map
    )
    results = torch.t(out_in_map).contiguous()
    ref_nbsizes = torch.sum(results != -1, dim=1).to(torch.int32)
    ref_nbmaps = torch.nonzero(results != -1)
    ref_nbmaps[:, 0] = results.view(-1)[
        ref_nbmaps[:, 0] * results.size(1) + ref_nbmaps[:, 1]
    ]
    ref_nbmaps = ref_nbmaps.transpose(0, 1).int().contiguous()
    ref_nbaddrs = torch.zeros((ref_nbsizes.numel() + 1), dtype=torch.int32, device="cuda")
    ref_qnbaddrs = torch.zeros_like(ref_nbaddrs)
    torch_lattice.backend.exclusive_scan_quantified_wrapper(
        ref_nbsizes.numel(), ref_nbsizes, ref_nbaddrs, ref_qnbaddrs
    )

    torch.testing.assert_close(nbmaps, ref_nbmaps)
    torch.testing.assert_close(nbsizes, ref_nbsizes)
    torch.testing.assert_close(nbaddrs, ref_nbaddrs)
    torch.testing.assert_close(qnbaddrs, ref_qnbaddrs)


if __name__ == "__main__":
    # Only support single conv layer
    # Cannot support even kernel sizes >= 4 (because of the different definition of anchor point)

    # Set conv_configuration
    config = F.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap_on_the_fly"
    config.dataflow = F.Dataflow.ImplicitGEMM
    config.ifsort = True
    F.conv_config.set_global_conv_config(config)

    kernel_sizes = [2, 3, 5]
    strides = [1, 2, 3]

    for kernel_size in kernel_sizes:
        config.split_mask_num = kernel_size
        F.conv_config.set_global_conv_config(config)
        for stride in strides:
            mean_adiff, max_rdiff = check_single_layer_convolution_forward(
                kernel_size=kernel_size, stride=stride
            )
            print("****************************")
            print("kernel_size, stride:", kernel_size, stride)
            print("mean_adiff, max_rdiff:", mean_adiff, max_rdiff)
            print("****************************")
