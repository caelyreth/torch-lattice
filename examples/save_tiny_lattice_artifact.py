from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

import torch_lattice
from torch_lattice import nn as spnn
from torch_lattice.artifact import save_lattice_model_artifact


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(
        spnn.Conv3d(2, 4, kernel_size=1, bias=True),
        spnn.BatchNorm(4),
        spnn.ReLU(),
        spnn.GlobalAvgPool(),
        nn.Linear(4, 2),
    ).eval()
    sample = torch_lattice.SparseTensor(
        feats=torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            dtype=torch.float32,
        ),
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

    artifact_dir = Path("artifacts/tiny_sparse_pool_linear.lattice")
    report = save_lattice_model_artifact(model, artifact_dir, sample_input=sample)
    with torch.no_grad():
        expected = model(sample)
    save_file(
        {
            "coords": sample.coords.cpu(),
            "features": sample.feats.cpu(),
            "active": torch.tensor([sample.coords.shape[0]], dtype=torch.int32),
            "expected": expected.detach().cpu(),
        },
        artifact_dir.with_suffix(".check.safetensors"),
        metadata={"format": "torch"},
    )

    print(f"graph: {report.graph_path}")
    print(f"weights: {report.weights_path}")
    print(f"weight keys: {', '.join(report.weight_keys)}")


if __name__ == "__main__":
    main()
