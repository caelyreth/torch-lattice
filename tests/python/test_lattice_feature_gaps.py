from __future__ import annotations

import torch
from torch import nn
from safetensors.torch import load_file

import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.export import (
    LatticeExportOptions,
    TorchLatticeExportBuilder,
    export_lattice_artifact,
    lower_fx_module,
)


def _sparse(feats=None):
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=torch.int32,
    )
    if feats is None:
        feats = torch.tensor([[1.0], [2.0], [4.0]])
    return SparseTensor(feats=feats, coords=coords, spatial_range=(1, 3, 1, 1))


def test_sparse_pool3d_runtime_sum_avg_max():
    x = _sparse()
    summed = torch_lattice.nn.functional.pool3d(
        x,
        mode="sum",
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(1, 0, 0),
    )
    averaged = torch_lattice.nn.functional.avg_pool3d(
        x,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(1, 0, 0),
    )
    maxed = torch_lattice.nn.functional.max_pool3d(
        x,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(1, 0, 0),
    )
    torch.testing.assert_close(summed.feats, torch.tensor([[3.0], [7.0], [6.0]]))
    torch.testing.assert_close(averaged.feats, torch.tensor([[1.5], [7.0 / 3.0], [3.0]]))
    torch.testing.assert_close(maxed.feats, torch.tensor([[2.0], [4.0], [4.0]]))


def test_target_conv3d_runtime_uses_target_support():
    x = _sparse(torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]))
    target = SparseTensor(
        feats=torch.empty((2, 1)),
        coords=torch.tensor([[0, 0, 0, 0], [0, 2, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 3, 1, 1),
    )
    conv = spnn.TargetConv3d(2, 1, kernel_size=1, bias=True)
    with torch.no_grad():
        conv.kernel.copy_(torch.tensor([[2.0], [3.0]]))
        conv.bias.copy_(torch.tensor([0.5]))
    out = conv(x, target)
    assert torch.equal(out.coords, target.coords)
    torch.testing.assert_close(out.feats, torch.tensor([[8.5], [47.5]]))


def test_high_level_voxelize_and_devoxelize_runtime():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 0.0, 0.0]])
    feats = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]])
    batches = torch.zeros(3, dtype=torch.int32)
    voxels = torch_lattice.voxelize(
        points,
        feats,
        batch_indices=batches,
        active_rows=torch.tensor([3], dtype=torch.int32),
        reduction="mean",
    )
    torch.testing.assert_close(voxels.feats, torch.tensor([[2.0, 3.0], [9.0, 10.0]]))
    sampled = torch_lattice.devoxelize(
        points,
        voxels,
        batch_indices=batches,
        point_active_rows=torch.tensor([3], dtype=torch.int32),
        interpolation="nearest",
    )
    torch.testing.assert_close(sampled, torch.tensor([[2.0, 3.0], [2.0, 3.0], [9.0, 10.0]]))


def test_export_pool_and_target_conv_mlir(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.target = spnn.TargetConv3d(2, 3, kernel_size=1)
            self.pool = spnn.AvgPool3d(kernel_size=1, stride=1)

        def forward(self, x, target):
            return self.pool(self.target(x, target))

    builder = TorchLatticeExportBuilder(input_dtype="f32")
    target = builder.sparse_input()
    lower_fx_module(builder, Model().eval(), inputs=(builder.current, target))
    graph = builder.to_mlir()
    assert "lattice.target_conv3d" in graph
    assert "lattice.pool3d" in graph


def test_export_voxelize_and_devoxelize_mlir():
    builder = TorchLatticeExportBuilder(input_dtype="f32")
    points = builder.dense_argument("points", "tensor<?x3xf32>")
    features = builder.dense_argument("point_features", "tensor<?x2xf32>", channels=2)
    batches = builder.dense_argument("batch_indices", "tensor<?xi32>")
    active = builder.dense_argument("active_rows", "tensor<1xi32>")
    voxels = builder.voxelize(
        "voxels",
        points=points,
        features=features,
        batch_indices=batches,
        active_rows=active,
        voxel_size=(1.0, 1.0, 1.0),
    )
    out = builder.devoxelize(
        "sampled",
        points=points,
        voxels=voxels,
        batch_indices=batches,
        point_active_rows=active,
        voxel_size=(1.0, 1.0, 1.0),
    )
    builder.output(out)
    graph = builder.to_mlir()
    assert "lattice.voxelize" in graph
    assert "lattice.devoxelize" in graph


def test_quantized_artifact_export_stores_packed_weights(tmp_path):
    model = nn.Sequential(spnn.Conv3d(2, 3, kernel_size=1, bias=False)).eval()
    report = export_lattice_artifact(
        model,
        tmp_path,
        options=LatticeExportOptions(quantize_bits=8, quantize_group_size=32),
    )
    graph = report.graph_path.read_text()
    weights = load_file(report.weights_path)
    assert "#lattice.packing<int8" in graph
    assert "0.weight.weight" in weights
    assert "0.weight.scales" in weights
    assert "0.weight.biases" in weights


def test_sparse_activation_modules_and_norm_runtime():
    x = _sparse(torch.tensor([[0.3, -0.4, 0.5, -0.6], [0.7, 0.2, -0.1, 0.9]]))
    for module in (
        spnn.GELU(),
        spnn.Sigmoid(),
        spnn.Tanh(),
        spnn.Softplus(),
        spnn.LayerNorm(4),
        spnn.RMSNorm(4),
    ):
        out = module(x)
        assert isinstance(out, SparseTensor)
        assert torch.equal(out.coords, x.coords)
        assert out.feats.shape == x.feats.shape


def test_export_activation_and_norm_modules_mlir():
    model = nn.Sequential(
        spnn.Conv3d(4, 4, kernel_size=1),
        spnn.GELU(approximate="tanh"),
        spnn.Sigmoid(),
        spnn.Tanh(),
        spnn.Softplus(beta=2.0, threshold=10.0),
        spnn.LayerNorm(4),
        spnn.RMSNorm(4),
    ).eval()
    builder = TorchLatticeExportBuilder(input_dtype="f32")
    lower_fx_module(builder, model)
    graph = builder.to_mlir()
    assert graph.count(" = lattice.activation ") == 4
    assert "kind = #lattice.activation<gelu>" in graph
    assert "approximate = #lattice.gelu_approx<tanh>" in graph
    assert "kind = #lattice.activation<sigmoid>" in graph
    assert "kind = #lattice.activation<tanh>" in graph
    assert "kind = #lattice.activation<softplus>" in graph
    assert "lattice.layer_norm" in graph
    assert "lattice.rms_norm" in graph


def test_export_dense_head_activations_and_norms_mlir():
    model = nn.Sequential(
        spnn.GlobalAvgPool(),
        nn.LayerNorm(2),
        nn.GELU(),
        nn.Sigmoid(),
        nn.Tanh(),
        nn.Softplus(),
        nn.Linear(2, 2),
    ).eval()
    builder = TorchLatticeExportBuilder(input_dtype="f32", batch_size=1)
    lower_fx_module(builder, model)
    graph = builder.to_mlir()
    assert "lattice.global_pool" in graph
    assert "lattice.layer_norm" in graph
    assert graph.count(" = lattice.activation ") == 4
    assert "lattice.linear" in graph
