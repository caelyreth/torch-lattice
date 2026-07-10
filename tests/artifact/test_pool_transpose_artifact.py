from __future__ import annotations

from torch_lattice import nn as spnn
from torch_lattice.artifact import TorchLatticeArtifactBuilder


def test_pool_transpose_exports_optional_explicit_target() -> None:
    builder = TorchLatticeArtifactBuilder(create_default_input=False)
    source = builder.sparse_argument("source", channels=3, stride=2)
    target = builder.sparse_argument("target", channels=3)
    module = spnn.PoolTranspose3d(kernel_size=2, stride=2)

    output = builder.lower_module("up", module, source, target)
    builder.output(output)
    graph = builder.to_mlir()

    assert "lattice.pool_transpose3d %source, %target" in graph


def test_pool_transpose_exports_generated_support() -> None:
    builder = TorchLatticeArtifactBuilder(input_stride=2)
    builder.module("up", spnn.PoolTranspose3d(kernel_size=2, stride=2))

    graph = builder.to_mlir()

    assert "lattice.pool_transpose3d %input" in graph


def test_trilinear_upsample_exports_target_support() -> None:
    builder = TorchLatticeArtifactBuilder(create_default_input=False)
    source = builder.sparse_argument("source", channels=3, stride=2)
    target = builder.sparse_argument("target", channels=3)

    output = builder.lower_module(
        "up", spnn.TrilinearUpsample3d(stride=2), source, target
    )
    builder.output(output)
    graph = builder.to_mlir()

    assert "lattice.trilinear_upsample3d %source, %target" in graph
