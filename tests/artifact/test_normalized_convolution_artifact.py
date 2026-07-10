from __future__ import annotations

import pytest
from torch_lattice import nn as spnn
from torch_lattice.artifact import TorchLatticeArtifactBuilder


@pytest.mark.parametrize(
    ('module', 'op'),
    [
        (
            spnn.NormalizedSubmConv3d(2, 3, kernel_size=3),
            'lattice.normalized_subm_conv3d',
        ),
        (
            spnn.NormalizedConvTranspose3d(2, 3, kernel_size=2, stride=2),
            'lattice.normalized_conv_transpose3d',
        ),
        (
            spnn.NormalizedGenerativeConvTranspose3d(
                2, 3, kernel_size=2, stride=2
            ),
            'lattice.normalized_generative_conv_transpose3d',
        ),
    ],
)
def test_normalized_module_exports_first_class_mlir_op(module, op) -> None:
    builder = TorchLatticeArtifactBuilder(input_stride=(2, 2, 2))

    builder.module('normalized', module.eval())
    graph = builder.to_mlir()

    assert op in graph
    assert 'eps = 0.00000001 : f32' in graph


def test_normalized_artifact_rejects_packed_weight_export() -> None:
    builder = TorchLatticeArtifactBuilder(quantize_bits=8)

    with pytest.raises(ValueError, match='weight squares'):
        builder.module('normalized', spnn.NormalizedSubmConv3d(2, 3))
