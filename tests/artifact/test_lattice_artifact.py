from __future__ import annotations

import pytest
import torch
import torch_lattice
from lattice_contract import DIALECT_SCHEMA_DIGEST
from safetensors.torch import load_file
from torch import nn
from torch_lattice import nn as spnn
from torch_lattice.artifact import (
    LatticeModelArtifactOptions,
    TorchLatticeArtifactBuilder,
    save_lattice_model_artifact,
)

pytestmark = pytest.mark.artifact


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


class BinarySparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = spnn.Conv3d(2, 2, kernel_size=1)
        self.right = spnn.Conv3d(2, 2, kernel_size=1)

    def forward(self, x):
        return torch_lattice.sparse_sub(
            self.left(x),
            self.right(x),
            join="left",
            rhs_fill=1.5,
        )


class MulSparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = spnn.Conv3d(2, 2, kernel_size=1)
        self.right = spnn.Conv3d(2, 2, kernel_size=1)

    def forward(self, x):
        return self.left(x) * self.right(x)


class CatSparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = spnn.Conv3d(2, 3, kernel_size=1)
        self.right = spnn.Conv3d(2, 4, kernel_size=1)

    def forward(self, x):
        return torch_lattice.cat([self.left(x), self.right(x)])


class MultiInputOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = spnn.Conv3d(2, 3, kernel_size=1, bias=True)
        self.pool = spnn.GlobalSumPool()

    def forward(self, x, target):
        sparse = self.conv(x, coordinates=target)
        return sparse, self.pool(sparse)


class ReindexSparseModel(nn.Module):
    def forward(self, x, target):
        return torch_lattice.reindex_sparse(x, target, fill=-1.25)


def test_artifact_fx_tiny_sparse_pool_linear_artifact(tmp_path):
    torch.manual_seed(0)
    model = TinySparseModel().eval()
    sample = _sample_sparse_tensor()

    report = save_lattice_model_artifact(
        model,
        tmp_path / "tiny_sparse.lattice",
        example_inputs=(sample,),
    )

    assert sorted(path.name for path in report.artifact_dir.iterdir()) == [
        "graph.mlir",
        "weights.safetensors",
    ]
    graph = report.graph_path.read_text(encoding="utf-8")
    assert "lattice.ir_version = 0" in graph
    assert f'lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}"' in graph
    assert 'lattice.input_names = ["x_coords", "x_features", "x_active"]' in graph
    assert "%x_active: tensor<1xi32>" in graph
    assert 'lattice.output_names = ["output"]' in graph
    assert "lattice.sparse.make" in graph
    assert "lattice.conv3d" in graph
    assert "lattice.subm_conv3d" not in graph
    assert "lattice.activation" in graph
    assert "lattice.global_pool" in graph
    assert "lattice.linear" in graph
    assert "manifest.json" not in {path.name for path in report.artifact_dir.iterdir()}

    weights = load_file(report.weights_path)
    assert weights["stem.weight"].shape == (3, 1, 1, 1, 2)
    assert weights["stem.bias"].shape == (3,)
    assert weights["head.weight"].shape == (2, 3)
    assert weights["head.bias"].shape == (2,)


def test_artifact_fx_branch_add_artifact(tmp_path):
    model = SkipAddSparseModel().eval()

    report = save_lattice_model_artifact(
        model,
        tmp_path / "skip_add.lattice",
        example_inputs=(_sample_sparse_tensor(),),
        options=LatticeModelArtifactOptions(batch_size=2),
    )
    graph = report.graph_path.read_text(encoding="utf-8")

    assert "lattice.sparse.binary" in graph
    assert "op = #lattice.binary_op<add>" in graph
    assert "join = #lattice.join<outer>" in graph
    assert "lattice.activation" in graph


def test_artifact_fx_branch_cat_artifact(tmp_path):
    model = CatSparseModel().eval()

    report = save_lattice_model_artifact(
        model,
        tmp_path / "cat.lattice",
        example_inputs=(_sample_sparse_tensor(),),
        options=LatticeModelArtifactOptions(batch_size=2),
    )
    graph = report.graph_path.read_text(encoding="utf-8")
    weights = load_file(report.weights_path)

    assert "lattice.sparse.cat" in graph
    assert "join = #lattice.join<inner>" in graph
    assert weights["left.weight"].shape == (3, 1, 1, 1, 2)
    assert weights["right.weight"].shape == (4, 1, 1, 1, 2)


def test_artifact_fx_sparse_binary_join_and_fill_artifact(tmp_path):
    model = BinarySparseModel().eval()

    report = save_lattice_model_artifact(
        model,
        tmp_path / "binary.lattice",
        example_inputs=(_sample_sparse_tensor(),),
        options=LatticeModelArtifactOptions(batch_size=2),
    )
    graph = report.graph_path.read_text(encoding="utf-8")

    assert "lattice.sparse.binary" in graph
    assert "op = #lattice.binary_op<sub>" in graph
    assert "join = #lattice.join<left>" in graph
    assert "lhs_fill = 0.0 : f32" in graph
    assert "rhs_fill = 1.5 : f32" in graph


def test_artifact_fx_operator_mul_uses_inner_sparse_binary_artifact(
    tmp_path,
):
    model = MulSparseModel().eval()

    report = save_lattice_model_artifact(
        model,
        tmp_path / "mul.lattice",
        example_inputs=(_sample_sparse_tensor(),),
        options=LatticeModelArtifactOptions(batch_size=2),
    )
    graph = report.graph_path.read_text(encoding="utf-8")

    assert "lattice.sparse.binary" in graph
    assert "op = #lattice.binary_op<mul>" in graph
    assert "join = #lattice.join<inner>" in graph


def test_artifact_multi_input_output_uses_signature_abi(tmp_path):
    model = MultiInputOutputModel().eval()
    sample = _sample_sparse_tensor()
    target = torch_lattice.SparseTensor(
        torch.empty((2, 1), dtype=torch.float32),
        sample.coords[[0, 2]].clone(),
        spatial_range=sample.spatial_range,
        batch_counts=(1, 1),
    )

    report = save_lattice_model_artifact(
        model,
        tmp_path / "multi.lattice",
        example_inputs=(sample, target),
        output_names=("features", "summary"),
    )
    graph = report.graph_path.read_text(encoding="utf-8")

    assert (
        'lattice.input_names = ["x_coords", "x_features", "x_active", "target_coords", "target_features", "target_active"]'
        in graph
    )
    assert 'lattice.output_names = ["features", "summary"]' in graph
    assert "lattice.target_conv3d" in graph
    assert "lattice.global_pool" in graph


def test_artifact_fx_sparse_reindex_preserves_target_contract(tmp_path):
    sample = _sample_sparse_tensor()
    target = torch_lattice.SparseTensor(
        torch.empty((2, 1), dtype=torch.float32),
        sample.coords[[3, 0]].clone(),
        spatial_range=sample.spatial_range,
        batch_counts=(1, 1),
    )

    report = save_lattice_model_artifact(
        ReindexSparseModel().eval(),
        tmp_path / "reindex.lattice",
        example_inputs=(sample, target),
    )
    graph = report.graph_path.read_text(encoding="utf-8")

    assert "lattice.sparse.reindex" in graph
    assert "fill = -1.25 : f32" in graph


def test_artifact_validation_option_controls_writer_validation(tmp_path, monkeypatch):
    calls = []

    def record_validation(graph, weights_path):
        calls.append((graph, weights_path))

    monkeypatch.setattr(
        "torch_lattice.artifact.io._validate_payload", record_validation
    )
    model = spnn.Conv3d(2, 3, kernel_size=1).eval()
    sample = _sample_sparse_tensor()
    save_lattice_model_artifact(
        model,
        tmp_path / "unchecked.lattice",
        example_inputs=(sample,),
        options=LatticeModelArtifactOptions(validate=False),
    )
    assert not calls

    save_lattice_model_artifact(
        model,
        tmp_path / "checked.lattice",
        example_inputs=(sample,),
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("module", "expected_op"),
    [
        (spnn.Conv3d(2, 3, kernel_size=1, bias=False), "lattice.conv3d"),
        (
            spnn.SubmConv3d(2, 3, kernel_size=3, bias=False),
            "lattice.subm_conv3d",
        ),
        (
            spnn.ConvTranspose3d(2, 3, kernel_size=2, stride=2, bias=False),
            "lattice.conv_transpose3d",
        ),
        (
            spnn.GenerativeConvTranspose3d(2, 3, kernel_size=2, stride=2, bias=False),
            "lattice.generative_conv_transpose3d",
        ),
    ],
)
def test_artifact_conv3d_module_identity_selects_mlir_op(tmp_path, module, expected_op):
    report = save_lattice_model_artifact(
        module.eval(),
        tmp_path / "identity.lattice",
        example_inputs=(_sample_sparse_tensor(),),
        options=LatticeModelArtifactOptions(batch_size=2),
    )
    graph = report.graph_path.read_text(encoding="utf-8")

    assert expected_op in graph


def test_artifact_conv3d_weight_layout(tmp_path):
    conv = spnn.Conv3d(2, 3, kernel_size=(2, 1, 2), stride=(2, 1, 2), bias=False)
    with torch.no_grad():
        conv.kernel.copy_(torch.arange(conv.kernel.numel()).reshape_as(conv.kernel))

    report = save_lattice_model_artifact(
        conv.eval(),
        tmp_path / "conv.lattice",
        example_inputs=(_sample_sparse_tensor(),),
        options=LatticeModelArtifactOptions(batch_size=2),
    )
    weights = load_file(report.weights_path)

    exported = weights["conv3d.weight"]
    expected = conv.kernel.detach().reshape(2, 1, 2, 2, 3).permute(4, 0, 1, 2, 3)
    torch.testing.assert_close(exported, expected)


def test_artifact_reorders_native_odd_kernel_layout(tmp_path):
    conv = spnn.SubmConv3d(1, 1, kernel_size=(3, 3, 3), bias=False)
    sample = _sample_sparse_tensor()
    with torch.no_grad():
        conv.kernel.copy_(torch.arange(conv.kernel.numel()).reshape_as(conv.kernel))

    report = save_lattice_model_artifact(
        conv.eval(),
        tmp_path / "subm.lattice",
        example_inputs=(sample.replace(feats=sample.feats[:, :1]),),
    )
    exported = load_file(report.weights_path)["submconv3d.weight"]
    expected = (
        conv.kernel.detach()
        .reshape(3, 3, 3, 1, 1)
        .permute(2, 1, 0, 3, 4)
        .permute(4, 0, 1, 2, 3)
    )

    torch.testing.assert_close(exported, expected)


def test_explicit_builder_exports_same_artifact_shape(tmp_path):
    model = TinySparseModel().eval()
    builder = TorchLatticeArtifactBuilder(batch_size=2)
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
def test_artifact_rejects_norms_without_mlx_mlir_semantics(tmp_path, module):
    model = nn.Sequential(spnn.Conv3d(2, 3, kernel_size=1), module).eval()

    with pytest.raises(ValueError, match="not supported"):
        save_lattice_model_artifact(
            model,
            tmp_path / "unsupported.lattice",
            example_inputs=(_sample_sparse_tensor(),),
            options=LatticeModelArtifactOptions(batch_size=2),
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
        batch_counts=(2, 2),
    )
