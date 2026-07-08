from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.export import (
    LatticeExportOptions,
    TorchLatticeExportBuilder,
    export_lattice_artifact,
    lower_fx_module,
)

ROOT = Path('/tmp/torch_lattice_e2e_fixtures')


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    torch.manual_seed(7)
    _sparse_classifier(ROOT / 'sparse_classifier')
    _target_branch(ROOT / 'target_branch')
    _point_voxel(ROOT / 'point_voxel')
    (ROOT / 'manifest.json').write_text(
        json.dumps(
            {
                'cases': [
                    'sparse_classifier',
                    'target_branch',
                    'point_voxel',
                ]
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(ROOT)


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


class TargetBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = spnn.Conv3d(2, 3, kernel_size=1, bias=True)
        self.right = spnn.Conv3d(2, 3, kernel_size=1, bias=False)
        self.target_conv = spnn.TargetConv3d(3, 2, kernel_size=1, bias=True)

    def forward(self, x: SparseTensor, target: SparseTensor) -> SparseTensor:
        merged = torch_lattice.sparse_add(self.left(x), self.right(x), join='outer')
        sampled = self.target_conv(merged, target)
        return torch_lattice.cat([merged, sampled], join='inner')


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
            reduction='mean',
        )
        return torch_lattice.devoxelize(
            points,
            voxels,
            batch_indices=batch_indices,
            point_active_rows=active_rows,
            voxel_size=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            interpolation='nearest',
        )


def _sparse_classifier(case_dir: Path) -> None:
    case_dir.mkdir()
    model = SparseClassifier()
    x = SparseTensor(
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
    export_lattice_artifact(
        model,
        case_dir,
        sample_input=x,
        options=LatticeExportOptions(batch_size=2),
    )
    _save_sparse_inputs(case_dir, '', x)
    save_file({'output': expected}, case_dir / 'expected.safetensors')


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
    builder = TorchLatticeExportBuilder(input_dtype='f32')
    target_value = builder.sparse_argument('target', channels=1)
    lower_fx_module(builder, model, inputs=(builder.current, target_value))
    builder.save(case_dir)
    _save_sparse_inputs(case_dir, '', x, extra={'target': target})
    save_file(
        {
            'output.coords': expected.coords,
            'output.features': expected.feats,
            'output.active': _active_rows(expected),
        },
        case_dir / 'expected.safetensors',
    )


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
    builder = TorchLatticeExportBuilder(input_dtype='f32', create_default_input=False)
    points_value = builder.dense_argument('points', 'tensor<?x3xf32>')
    features_value = builder.dense_argument('features', 'tensor<?x2xf32>', channels=2)
    batch_value = builder.dense_argument('batch_indices', 'tensor<?xi32>')
    active_value = builder.dense_argument('active_rows', 'tensor<1xi32>')
    lower_fx_module(
        builder,
        model,
        inputs=(points_value, features_value, batch_value, active_value),
    )
    builder.save(case_dir)
    save_file(
        {
            'points': points,
            'features': features,
            'batch_indices': batch_indices,
            'active_rows': active_rows,
        },
        case_dir / 'inputs.safetensors',
    )
    save_file({'output': expected}, case_dir / 'expected.safetensors')


def _save_sparse_inputs(
    case_dir: Path,
    prefix: str,
    tensor: SparseTensor,
    *,
    extra: dict[str, SparseTensor] | None = None,
) -> None:
    values = {
        f'{prefix}coords': tensor.coords,
        f'{prefix}features': tensor.feats,
        f'{prefix}active': _active_rows(tensor),
    }
    for name, sparse in (extra or {}).items():
        values[f'{name}_coords'] = sparse.coords
        values[f'{name}_features'] = sparse.feats
        values[f'{name}_active'] = _active_rows(sparse)
    save_file(values, case_dir / 'inputs.safetensors')


def _active_rows(tensor: SparseTensor) -> torch.Tensor:
    return torch.tensor([tensor.feats.shape[0]], dtype=torch.int32)


if __name__ == '__main__':
    main()
