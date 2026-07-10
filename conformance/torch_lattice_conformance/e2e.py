from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch_lattice
from safetensors.torch import save_file
from torch import nn
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.artifact import (
    LatticeModelArtifactOptions,
    TorchLatticeArtifactBuilder,
    lower_fx_artifact,
    save_lattice_model_artifact,
)
from torch_lattice.nn.functional.conv import Dataflow, conv_config

ROOT = Path("/tmp/torch_lattice_e2e_fixtures")


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    torch.manual_seed(7)
    cases = [
        "sparse_classifier",
        "target_branch",
        "point_voxel",
        "quantized_classifier_int8",
        "quantized_classifier_int4",
        "transpose_convolution",
        "generative_transpose_convolution",
        "normalized_convolution",
        "target_transpose_convolution",
        "pool_transpose",
    ]
    _sparse_classifier(ROOT / "sparse_classifier")
    _target_branch(ROOT / "target_branch")
    _point_voxel(ROOT / "point_voxel")
    _quantized_classifier(ROOT / "quantized_classifier_int8", bits=8)
    _quantized_classifier(ROOT / "quantized_classifier_int4", bits=4)
    _transpose_convolution(ROOT / "transpose_convolution")
    _generative_transpose_convolution(ROOT / "generative_transpose_convolution")
    _normalized_convolution(ROOT / "normalized_convolution")
    _target_transpose_convolution(ROOT / "target_transpose_convolution")
    _pool_transpose(ROOT / "pool_transpose")
    (ROOT / "manifest.json").write_text(
        json.dumps({"cases": cases}, indent=2),
        encoding="utf-8",
    )
    print(ROOT)


@contextmanager
def _conv_dataflow(
    dataflow: Dataflow,
    *,
    kmap_mode: str | None = None,
) -> Iterator[None]:
    previous = conv_config.get_global_conv_config()
    config = conv_config.get_default_conv_config()
    config.dataflow = dataflow
    config.ifsort = False
    if kmap_mode is not None:
        config.kmap_mode = kmap_mode
    conv_config.set_global_conv_config(config)
    try:
        yield
    finally:
        if previous is None:
            conv_config.clear_global_conv_config()
        else:
            conv_config.set_global_conv_config(previous)


class SparseClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = spnn.Conv3d(3, 4, kernel_size=1, bias=True)
        self.norm = spnn.BatchNorm(4)
        self.act = spnn.ReLU()
        self.pool = spnn.AvgPool3d(kernel_size=1, stride=1)
        self.global_pool = spnn.GlobalAvgPool()
        self.head = nn.Linear(4, 2)

    def forward(self, x: SparseTensor) -> torch.Tensor:
        return self.head(self.global_pool(self.pool(self.act(self.norm(self.stem(x))))))


class QuantizedClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = spnn.Conv3d(3, 4, kernel_size=1, bias=True)
        self.act = spnn.SiLU()
        self.global_pool = spnn.GlobalAvgPool()
        self.head = nn.Linear(4, 2)

    def forward(self, x: SparseTensor) -> torch.Tensor:
        return self.head(self.global_pool(self.act(self.stem(x))))


class TargetBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = spnn.Conv3d(2, 3, kernel_size=1, bias=True)
        self.right = spnn.Conv3d(2, 3, kernel_size=1, bias=False)
        self.target_conv = spnn.Conv3d(3, 2, kernel_size=1, bias=True)

    def forward(self, x: SparseTensor, target: SparseTensor) -> SparseTensor:
        merged = torch_lattice.sparse_add(self.left(x), self.right(x), join="outer")
        sampled = self.target_conv(merged, coordinates=target)
        return torch_lattice.cat([merged, sampled], join="inner")


class PointVoxel(nn.Module):
    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        batch_indices: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> torch.Tensor:
        voxels = torch_lattice.voxelize(
            points,
            features,
            batch_indices=batch_indices,
            active_rows=active_rows,
            voxel_size=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            reduction="mean",
        )
        return torch_lattice.devoxelize(
            points,
            voxels,
            batch_indices=batch_indices,
            point_active_rows=active_rows,
            voxel_size=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            interpolation="nearest",
        )


class TransposeConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = spnn.Conv3d(
            2, 3, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False
        )
        self.up = spnn.ConvTranspose3d(
            3, 2, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=True
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.up(self.down(x))


class GenerativeTransposeConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = spnn.GenerativeConvTranspose3d(
            2, 3, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=True
        )
        self.act = spnn.Tanh()

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.act(self.up(x))


class NormalizedConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = spnn.NormalizedSubmConv3d(2, 3, kernel_size=(3, 1, 1), bias=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.conv(x)


class PoolTranspose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = spnn.PoolTranspose3d(
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )

    def forward(
        self,
        source: SparseTensor,
        target: SparseTensor,
    ) -> SparseTensor:
        return self.up(source, target)


class TargetTransposeConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = spnn.NormalizedGenerativeConvTranspose3d(
            2,
            3,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            bias=True,
        )

    def forward(
        self,
        source: SparseTensor,
        target: SparseTensor,
    ) -> SparseTensor:
        return self.up(source, target)


def _sparse_classifier(case_dir: Path) -> None:
    case_dir.mkdir()
    model = SparseClassifier()
    x = _classifier_input()
    target = torch.tensor([[0.25, -0.5], [-0.1, 0.4]], dtype=torch.float32)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(6):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
    model.eval()
    expected = model(x).detach()
    save_lattice_model_artifact(
        model,
        case_dir,
        example_inputs=(x,),
        options=LatticeModelArtifactOptions(batch_size=2),
    )
    _save_sparse_inputs(case_dir, "x", x)
    save_file({"output": expected}, case_dir / "expected.safetensors")


def _quantized_classifier(case_dir: Path, *, bits: int) -> None:
    case_dir.mkdir()
    model = QuantizedClassifier().eval()
    x = _classifier_input()
    with torch.no_grad():
        model.stem.kernel.copy_(
            torch.tensor(
                [
                    [0.20, -0.10, 0.15, 0.05],
                    [-0.25, 0.30, 0.10, -0.20],
                    [0.40, 0.05, -0.30, 0.25],
                ],
                dtype=torch.float32,
            )
        )
        model.stem.bias.copy_(torch.tensor([0.02, -0.03, 0.04, 0.01]))
        model.head.weight.copy_(
            torch.tensor([[0.30, -0.20, 0.10, 0.05], [-0.15, 0.25, -0.05, 0.35]])
        )
        model.head.bias.copy_(torch.tensor([0.01, -0.02]))
    expected = model(x).detach()
    save_lattice_model_artifact(
        model,
        case_dir,
        example_inputs=(x,),
        options=LatticeModelArtifactOptions(
            batch_size=2,
            quantize_bits=bits,
            quantize_group_size=32,
        ),
    )
    _save_sparse_inputs(case_dir, "x", x)
    save_file({"output": expected}, case_dir / "expected.safetensors")


def _target_branch(case_dir: Path) -> None:
    case_dir.mkdir()
    model = TargetBranch()
    x = SparseTensor(
        feats=torch.tensor(
            [[0.3, -0.4], [0.7, 0.2], [-0.5, 0.8], [1.0, -0.6]],
            dtype=torch.float32,
        ),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )
    target = SparseTensor(
        feats=torch.zeros((2, 1), dtype=torch.float32),
        coords=torch.tensor([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 4, 1, 1),
    )
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    for _ in range(4):
        optimizer.zero_grad()
        out = model(x, target)
        loss = out.feats.square().mean()
        loss.backward()
        optimizer.step()
    model.eval()
    expected = model(x, target)
    builder = TorchLatticeArtifactBuilder(input_dtype="f32")
    target_value = builder.sparse_argument("target", channels=1)
    lower_fx_artifact(builder, model, inputs=(builder.current, target_value))
    builder.save(case_dir)
    _save_sparse_inputs(case_dir, "x", x, extra={"target": target})
    _save_sparse_expected(case_dir, expected)


def _point_voxel(case_dir: Path) -> None:
    case_dir.mkdir()
    model = PointVoxel().eval()
    points = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.4, 0.2, 0.2],
            [1.2, 0.1, 0.1],
            [1.6, 0.3, 0.2],
            [2.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    features = torch.tensor(
        [[1.0, -1.0], [3.0, 1.0], [5.0, 2.0], [7.0, 4.0], [9.0, 8.0]],
        dtype=torch.float32,
    )
    batch_indices = torch.zeros((5,), dtype=torch.int32)
    active_rows = torch.tensor([5], dtype=torch.int32)
    expected = model(points, features, batch_indices, active_rows).detach()
    builder = TorchLatticeArtifactBuilder(input_dtype="f32", create_default_input=False)
    points_value = builder.dense_argument("points", "tensor<?x3xf32>")
    features_value = builder.dense_argument("features", "tensor<?x2xf32>", channels=2)
    batch_value = builder.dense_argument("batch_indices", "tensor<?xi32>")
    active_value = builder.dense_argument("active_rows", "tensor<1xi32>")
    lower_fx_artifact(
        builder,
        model,
        inputs=(points_value, features_value, batch_value, active_value),
    )
    builder.save(case_dir)
    save_file(
        {
            "points": points,
            "features": features,
            "batch_indices": batch_indices,
            "active_rows": active_rows,
        },
        case_dir / "inputs.safetensors",
    )
    save_file({"output": expected}, case_dir / "expected.safetensors")


def _transpose_convolution(case_dir: Path) -> None:
    case_dir.mkdir()
    model = TransposeConvolution().eval()
    x = _transpose_input()
    model, x_eval = _cuda_eval_pair(model, x)
    with _conv_dataflow(Dataflow.GatherScatter):
        expected = model(x_eval).cpu()
        save_lattice_model_artifact(model, case_dir, example_inputs=(x,))
    _save_sparse_inputs(case_dir, "x", x)
    _save_sparse_expected(case_dir, expected)


def _generative_transpose_convolution(case_dir: Path) -> None:
    case_dir.mkdir()
    model = GenerativeTransposeConvolution().eval()
    x = SparseTensor(
        feats=torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float32),
        coords=torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 4, 1, 1),
        stride=(2, 1, 1),
    )
    expected = _generative_transpose_reference(model, x)
    save_lattice_model_artifact(model, case_dir, example_inputs=(x,))
    _save_sparse_inputs(case_dir, "x", x)
    _save_sparse_expected(case_dir, expected)


def _normalized_convolution(case_dir: Path) -> None:
    case_dir.mkdir()
    model = NormalizedConvolution().eval()
    x = _transpose_input()
    with torch.no_grad():
        model.conv.kernel.copy_(
            torch.tensor(
                [
                    [[0.2, -0.4, 0.1], [0.3, 0.5, -0.2]],
                    [[-0.1, 0.6, 0.4], [0.7, -0.3, 0.2]],
                    [[0.5, 0.2, -0.6], [-0.4, 0.1, 0.8]],
                ]
            )
        )
        model.conv.bias.copy_(torch.tensor([0.05, -0.02, 0.03]))
    model, x_eval = _cuda_eval_pair(model, x)
    with _conv_dataflow(Dataflow.GatherScatter):
        expected = model(x_eval).cpu()
    save_lattice_model_artifact(model, case_dir, example_inputs=(x,))
    _save_sparse_inputs(case_dir, "x", x)
    _save_sparse_expected(case_dir, expected)


def _pool_transpose(case_dir: Path) -> None:
    case_dir.mkdir()
    model = PoolTranspose().eval()
    source = SparseTensor(
        feats=torch.tensor([[0.25, -0.5], [0.75, 0.4]], dtype=torch.float32),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 2, 1, 1),
        stride=(2, 1, 1),
    )
    target = SparseTensor(
        feats=torch.zeros((4, 1), dtype=torch.float32),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 5, 1, 1),
    )
    expected = model(source, target)
    builder = TorchLatticeArtifactBuilder(input_dtype="f32", create_default_input=False)
    source_value = builder.sparse_argument("source", channels=2, stride=(2, 1, 1))
    target_value = builder.sparse_argument("target", channels=1)
    lower_fx_artifact(builder, model, inputs=(source_value, target_value))
    builder.save(case_dir)
    _save_sparse_inputs(case_dir, "source", source, extra={"target": target})
    _save_sparse_expected(case_dir, expected)


def _target_transpose_convolution(case_dir: Path) -> None:
    case_dir.mkdir()
    model = TargetTransposeConvolution().eval()
    source = SparseTensor(
        feats=torch.tensor([[0.25, -0.5], [0.75, 0.4]], dtype=torch.float32),
        coords=torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32),
        spatial_range=(1, 2, 1, 1),
        stride=(2, 1, 1),
    )
    target = SparseTensor(
        feats=torch.zeros((4, 1), dtype=torch.float32),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 5, 1, 1),
    )
    with torch.no_grad():
        model.up.kernel.copy_(
            torch.tensor(
                [
                    [[0.2, -0.4, 0.1], [0.3, 0.5, -0.2]],
                    [[-0.1, 0.6, 0.4], [0.7, -0.3, 0.2]],
                    [[0.5, 0.2, -0.6], [-0.4, 0.1, 0.8]],
                ]
            )
        )
        model.up.bias.copy_(torch.tensor([0.05, -0.02, 0.03]))
    model, source_eval = _cuda_eval_pair(model, source)
    if source_eval.feats.is_cuda:
        target_eval = SparseTensor(
            feats=target.feats.cuda(),
            coords=target.coords.cuda(),
            stride=target.stride,
            spatial_range=target.spatial_range,
        )
    else:
        target_eval = target
    expected = model(source_eval, target_eval).cpu()
    builder = TorchLatticeArtifactBuilder(input_dtype="f32", create_default_input=False)
    source_value = builder.sparse_argument("source", channels=2, stride=(2, 1, 1))
    target_value = builder.sparse_argument("target", channels=1)
    lower_fx_artifact(builder, model.cpu(), inputs=(source_value, target_value))
    builder.save(case_dir)
    _save_sparse_inputs(case_dir, "source", source, extra={"target": target})
    _save_sparse_expected(case_dir, expected)


def _generative_transpose_reference(
    model: GenerativeTransposeConvolution,
    tensor: SparseTensor,
) -> SparseTensor:
    kernel_size = model.up.kernel_size
    stride = model.up.stride
    offsets = [
        (x, y, z)
        for x in range(kernel_size[0])
        for y in range(kernel_size[1])
        for z in range(kernel_size[2])
    ]
    rows: dict[tuple[int, int, int, int], torch.Tensor] = {}
    for coord, feat in zip(tensor.coords, tensor.feats, strict=True):
        base = coord.clone()
        for kernel_id, offset in enumerate(offsets):
            out_coord = (
                int(base[0]),
                int(base[1]) * stride[0] + offset[0],
                int(base[2]) * stride[1] + offset[1],
                int(base[3]) * stride[2] + offset[2],
            )
            value = feat @ model.up.kernel[kernel_id]
            rows[out_coord] = rows.get(out_coord, torch.zeros_like(value)) + value
    coords = torch.tensor(sorted(rows), dtype=torch.int32)
    feats = torch.stack([rows[tuple(coord.tolist())] for coord in coords])
    if model.up.bias is not None:
        feats = feats + model.up.bias
    return SparseTensor(
        feats=torch.tanh(feats),
        coords=coords,
        stride=tuple(
            int(tensor.stride[index]) // int(stride[index]) for index in range(3)
        ),
        spatial_range=tensor.spatial_range,
    )


def _cuda_eval_pair(
    model: nn.Module,
    tensor: SparseTensor,
) -> tuple[nn.Module, SparseTensor]:
    if not torch.cuda.is_available():
        return model, tensor
    return model.cuda(), SparseTensor(
        feats=tensor.feats.cuda(),
        coords=tensor.coords.cuda(),
        stride=tensor.stride,
        spatial_range=tensor.spatial_range,
    )


def _classifier_input() -> SparseTensor:
    return SparseTensor(
        feats=torch.tensor(
            [
                [0.2, -0.1, 0.4],
                [0.5, 0.3, -0.2],
                [-0.4, 0.7, 0.1],
                [0.9, -0.8, 0.6],
                [0.1, 0.2, 0.3],
            ],
            dtype=torch.float32,
        ),
        coords=torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 2, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.int32,
        ),
        spatial_range=(2, 3, 1, 1),
    )


def _transpose_input() -> SparseTensor:
    return SparseTensor(
        feats=torch.tensor(
            [[0.2, -0.3], [0.5, 0.1], [-0.4, 0.6], [0.8, -0.2]],
            dtype=torch.float32,
        ),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=torch.int32,
        ),
        spatial_range=(1, 4, 1, 1),
    )


def _save_sparse_inputs(
    case_dir: Path,
    name: str,
    tensor: SparseTensor,
    *,
    extra: dict[str, SparseTensor] | None = None,
) -> None:
    values = {
        f"{name}_coords": tensor.coords,
        f"{name}_features": tensor.feats,
        f"{name}_active": _active_rows(tensor),
    }
    for name, sparse in (extra or {}).items():
        values[f"{name}_coords"] = sparse.coords
        values[f"{name}_features"] = sparse.feats
        values[f"{name}_active"] = _active_rows(sparse)
    save_file(values, case_dir / "inputs.safetensors")


def _save_sparse_expected(case_dir: Path, expected: SparseTensor) -> None:
    save_file(
        {
            "output.coords": expected.coords,
            "output.features": expected.feats,
            "output.active": _active_rows(expected),
        },
        case_dir / "expected.safetensors",
    )


def _active_rows(tensor: SparseTensor) -> torch.Tensor:
    return torch.tensor([tensor.feats.shape[0]], dtype=torch.int32)


if __name__ == "__main__":
    main()
