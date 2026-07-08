from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

import torch_lattice
from lattice_contract import DIALECT_SCHEMA_DIGEST
from torch_lattice import nn as spnn
from torch_lattice.export import (
    LatticeExportOptions,
    TorchLatticeExportBuilder,
    export_lattice_artifact,
)


class TinySparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = spnn.Conv3d(2, 3, kernel_size=1, bias=True)
        self.act = spnn.ReLU()
        self.pool = spnn.GlobalAvgPool()
        self.head = nn.Linear(3, 2)

    def forward(self, x):
        return self.head(self.pool(self.act(self.stem(x))))


class SkipAddSparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = spnn.Conv3d(2, 2, kernel_size=1)
        self.right = spnn.Conv3d(2, 2, kernel_size=1)
        self.act = spnn.ReLU()

    def forward(self, x):
        return self.act(self.left(x) + self.right(x))


class CatSparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = spnn.Conv3d(2, 3, kernel_size=1)
        self.right = spnn.Conv3d(2, 4, kernel_size=1)

    def forward(self, x):
        return torch_lattice.cat([self.left(x), self.right(x)])


def test_export_fx_tiny_sparse_pool_linear_artifact(tmp_path):
    torch.manual_seed(0)
    model = TinySparseModel().eval()
    sample = _sample_sparse_tensor()

    report = export_lattice_artifact(
        model,
        tmp_path / "tiny_sparse.lattice",
        sample_input=sample,
    )

    assert sorted(path.name for path in report.artifact_dir.iterdir()) == [
        "graph.mlir",
        "weights.safetensors",
    ]
    graph = report.graph_path.read_text(encoding="utf-8")
    assert "lattice.ir_version = 0" in graph
    assert f'lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}"' in graph
    assert 'lattice.input_names = ["coords", "features", "active"]' in graph
    assert 'lattice.output_names = ["output"]' in graph
    assert "lattice.sparse.make" in graph
    assert "lattice.subm_conv3d" in graph
    assert "lattice.activation" in graph
    assert "lattice.global_pool" in graph
    assert "lattice.linear" in graph
    assert "manifest.json" not in {path.name for path in report.artifact_dir.iterdir()}

    weights = load_file(report.weights_path)
    assert weights["stem.weight"].shape == (3, 1, 1, 1, 2)
    assert weights["stem.bias"].shape == (3,)
    assert weights["head.weight"].shape == (2, 3)
    assert weights["head.bias"].shape == (2,)


def test_export_fx_branch_add_artifact(tmp_path):
    model = SkipAddSparseModel().eval()

    report = export_lattice_artifact(
        model,
        tmp_path / "skip_add.lattice",
        options=LatticeExportOptions(batch_size=2),
    )
    graph = report.graph_path.read_text(encoding="utf-8")

    assert "lattice.sparse.binary" in graph
    assert "op = #lattice.binary_op<add>" in graph
    assert "join = #lattice.join<outer>" in graph
    assert "lattice.activation" in graph


def test_export_fx_branch_cat_artifact(tmp_path):
    model = CatSparseModel().eval()

    report = export_lattice_artifact(
        model,
        tmp_path / "cat.lattice",
        options=LatticeExportOptions(batch_size=2),
    )
    graph = report.graph_path.read_text(encoding="utf-8")
    weights = load_file(report.weights_path)

    assert "lattice.sparse.cat" in graph
    assert "join = #lattice.join<inner>" in graph
    assert weights["left.weight"].shape == (3, 1, 1, 1, 2)
    assert weights["right.weight"].shape == (4, 1, 1, 1, 2)


def test_export_conv3d_weight_layout(tmp_path):
    conv = spnn.Conv3d(2, 3, kernel_size=(2, 1, 2), stride=(2, 1, 2), bias=False)
    with torch.no_grad():
        conv.kernel.copy_(torch.arange(conv.kernel.numel()).reshape_as(conv.kernel))

    report = export_lattice_artifact(
        conv,
        tmp_path / "conv.lattice",
        options=LatticeExportOptions(batch_size=2),
    )
    weights = load_file(report.weights_path)

    exported = weights["conv3d.weight"]
    expected = conv.kernel.detach().reshape(2, 1, 2, 2, 3).permute(4, 0, 1, 2, 3)
    torch.testing.assert_close(exported, expected)


def test_explicit_builder_exports_same_artifact_shape(tmp_path):
    model = TinySparseModel().eval()
    builder = TorchLatticeExportBuilder(batch_size=2)
    builder.module("stem", model.stem)
    builder.module("act", model.act)
    builder.module("pool", model.pool)
    builder.module("head", model.head)

    report = builder.save(tmp_path / "explicit.lattice")
    graph = report.graph_path.read_text(encoding="utf-8")

    assert "lattice.global_pool" in graph
    assert "lattice.linear" in graph
    assert sorted(path.name for path in report.artifact_dir.iterdir()) == [
        "graph.mlir",
        "weights.safetensors",
    ]


@pytest.mark.parametrize("module", [spnn.InstanceNorm(3), spnn.GroupNorm(1, 3)])
def test_export_rejects_norms_without_mlx_mlir_semantics(tmp_path, module):
    model = nn.Sequential(spnn.Conv3d(2, 3, kernel_size=1), module).eval()

    with pytest.raises(ValueError, match="not supported"):
        export_lattice_artifact(
            model,
            tmp_path / "unsupported.lattice",
            options=LatticeExportOptions(batch_size=2),
        )


def _sample_sparse_tensor() -> torch_lattice.SparseTensor:
    return torch_lattice.SparseTensor(
        feats=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        coords=torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.int32,
        ),
        spatial_range=(2, 2, 1, 1),
    )
