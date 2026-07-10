from __future__ import annotations

from importlib import import_module
from typing import Tuple, Union

import numpy as np
import pytest
import torch
from torch import nn

import torch_lattice
from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F
from torch_lattice.utils import make_ntuple
from tests.conftest import cuda_required
from tests.cases.dense_reference import (
    dense_pad,
    dense_to_subm,
    generate_feature_map,
    sparse_tensor_to_dense,
)

pytestmark = [pytest.mark.ops, pytest.mark.conv]
conv_impl = import_module("torch_lattice.nn.functional.conv.conv")


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

    ts_coords = out.coords
    ts_coords_np = np.asarray(ts_coords.detach().cpu())

    ref_out_np = ref_out.detach().cpu().numpy()
    ref_out_subm_np = dense_to_subm(ref_out_np, ts_coords_np)

    out_dense_np = sparse_tensor_to_dense(out, ref_out_np.shape[2:], OC, dtype=np_dtype)

    # print(ref_out_np)
    # print(out_dense_np)
    mean_adiff = np.sum(np.abs(out_dense_np - ref_out_subm_np)) / ts_coords.shape[0]
    max_adiff = np.max(np.abs(out_dense_np - ref_out_subm_np))
    max_rdiff = max_adiff / np.mean(np.abs(out_dense_np))
    assert mean_adiff <= 5e-2
    assert max_rdiff <= 5e-2
    return mean_adiff, max_rdiff


@cuda_required
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
    kmap = x.coord_manager.cached_relations[0]
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

    monkeypatch.setattr(conv_impl, "implicit_gemm_forward_no_grad", fake_fast)
    monkeypatch.setattr(conv_impl.ImplicitGEMMConvolutionFuntion, "apply", fake_apply)

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
        setattr(config, key, value)
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

    monkeypatch.setattr(conv_impl, fast_name, fake_fast)
    monkeypatch.setattr(getattr(conv_impl, function_name), "apply", fake_apply)

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
@pytest.mark.parametrize(
    "dataflow", [F.Dataflow.FetchOnDemand, F.Dataflow.GatherScatter]
)
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

    fod_map, neighbor_pairs, nbsizes, nbaddrs, qnbaddrs = (
        torch_lattice.backend.compact_out_in_map_fod(out_in_map)
    )
    results = torch.t(out_in_map).contiguous()
    ref_nbsizes = torch.sum(results != -1, dim=1).to(torch.int32)
    ref_nbmaps = torch.nonzero(results != -1)
    ref_nbmaps[:, 0] = results.view(-1)[
        ref_nbmaps[:, 0] * results.size(1) + ref_nbmaps[:, 1]
    ]
    ref_nbmaps = ref_nbmaps.transpose(0, 1).int().contiguous()
    ref_nbaddrs = torch.zeros(
        (ref_nbsizes.numel() + 1), dtype=torch.int32, device="cuda"
    )
    ref_qnbaddrs = torch.zeros_like(ref_nbaddrs)
    torch_lattice.backend.exclusive_scan_quantified_wrapper(
        ref_nbsizes.numel(), ref_nbsizes, ref_nbaddrs, ref_qnbaddrs
    )

    torch.testing.assert_close(fod_map, ref_nbmaps)
    torch.testing.assert_close(neighbor_pairs, ref_nbmaps.t().contiguous())
    torch.testing.assert_close(nbsizes, ref_nbsizes)
    torch.testing.assert_close(nbaddrs, ref_nbaddrs)
    torch.testing.assert_close(qnbaddrs, ref_qnbaddrs)


def _dataflow_config(dataflow, *, kmap_mode, fusion=False):
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = dataflow
    config.kmap_mode = kmap_mode
    config.ifsort = False
    config.FOD_fusion = fusion
    return config


def _training_case(*, dtype, kmap_mode, fusion, subm):
    side = 5
    grid = torch.stack(
        torch.meshgrid(
            *(torch.arange(side, device="cuda", dtype=torch.int32) for _ in range(3)),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    coords = torch.cat(
        (torch.zeros((grid.shape[0], 1), device="cuda", dtype=torch.int32), grid),
        dim=1,
    )
    spatial_range = (1, side, side, side)
    features = torch.randn(grid.shape[0], 8, device="cuda", dtype=dtype)
    weight = torch.randn(27, 8, 8, device="cuda", dtype=dtype)
    stride = 1 if subm else 2

    def run(dataflow):
        input_features = features.clone().requires_grad_(True)
        runtime_weight = weight.clone().requires_grad_(True)
        sparse = torch_lattice.SparseTensor(
            input_features,
            coords,
            spatial_range=spatial_range,
        )
        output = F.conv3d(
            sparse,
            runtime_weight,
            3,
            stride=stride,
            padding=1,
            config=_dataflow_config(
                dataflow,
                kmap_mode=kmap_mode,
                fusion=fusion,
            ),
            training=True,
            subm=subm,
        )
        return sparse, output, input_features, runtime_weight

    return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kmap_mode", ["hashmap", "hashmap_on_the_fly"])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("subm", [False, True])
@pytest.mark.parametrize("tensor_dtype", [torch.float32, torch.float16])
def test_fetch_on_demand_training_matches_gather_scatter(
    tensor_dtype, kmap_mode, fusion, subm
):
    run = _training_case(
        dtype=tensor_dtype,
        kmap_mode=kmap_mode,
        fusion=fusion,
        subm=subm,
    )
    fod_input, fod_output, fod_features, fod_weight = run(F.Dataflow.FetchOnDemand)
    _, reference, reference_features, reference_weight = run(F.Dataflow.GatherScatter)

    torch.testing.assert_close(fod_output.coords, reference.coords)
    relative_tolerance = 2e-2 if tensor_dtype == torch.float16 else 5e-3
    absolute_tolerance = 3e-2 if tensor_dtype == torch.float16 else 2e-2
    torch.testing.assert_close(
        fod_output.feats,
        reference.feats,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    pairs = fod_input.coord_manager.cached_relations[0]["neighbor_pairs"]
    fod_map = fod_input.coord_manager.cached_relations[0]["fod_neighbor_map"]
    assert pairs.dtype == torch.int32
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert pairs.is_contiguous()
    assert fod_map.dtype == torch.int32
    assert fod_map.shape == (2, pairs.shape[0])
    assert fod_map.is_contiguous()
    torch.testing.assert_close(fod_map, pairs.t().contiguous())

    grad = torch.randn_like(fod_output.feats)
    fod_output.feats.backward(grad)
    reference.feats.backward(grad)
    gradient_tolerance = 1e-2 if tensor_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(
        fod_features.grad,
        reference_features.grad,
        rtol=gradient_tolerance,
        atol=gradient_tolerance,
    )
    torch.testing.assert_close(
        fod_weight.grad,
        reference_weight.grad,
        rtol=gradient_tolerance,
        atol=gradient_tolerance,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fetch_on_demand_backward_is_repeatable():
    run = _training_case(
        dtype=torch.float32,
        kmap_mode="hashmap_on_the_fly",
        fusion=True,
        subm=False,
    )
    gradients = []
    grad = None
    for _ in range(3):
        _, output, features, weight = run(F.Dataflow.FetchOnDemand)
        if grad is None:
            grad = torch.randn_like(output.feats)
        output.feats.backward(grad)
        gradients.append((features.grad.clone(), weight.grad.clone()))

    for features_grad, weight_grad in gradients[1:]:
        torch.testing.assert_close(features_grad, gradients[0][0], rtol=0, atol=0)
        torch.testing.assert_close(weight_grad, gradients[0][1], rtol=0, atol=0)
