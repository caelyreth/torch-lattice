from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file
from torch_lattice_conformance.checkpoint import (
    convert_torchsparse_checkpoint,
    migrate_sparse_state_dict,
)


def test_checkpoint_conversion_requires_explicit_kernel_metadata(tmp_path) -> None:
    source = tmp_path / 'legacy.pt'
    torch.save({'state_dict': {'block.kernel': torch.ones((4, 1, 1))}}, source)

    with pytest.raises(ValueError, match='no kernel specification'):
        convert_torchsparse_checkpoint(
            source, tmp_path / 'weights.safetensors', kernel_specs={}
        )


def test_checkpoint_conversion_records_exact_odd_kernel_row_permutation(tmp_path) -> None:
    source = tmp_path / 'legacy.pt'
    kernel = torch.arange(27, dtype=torch.float32).reshape(27, 1, 1)
    bias = torch.tensor([0.25, -0.5])
    torch.save({'state_dict': {'block.kernel': kernel, 'block.bias': bias}}, source)

    result = convert_torchsparse_checkpoint(
        source,
        tmp_path / 'weights.safetensors',
        kernel_specs={'block.kernel': (3, 3, 3)},
    )

    converted = load_file(result.output)
    torch.testing.assert_close(
        converted['block.weight'],
        kernel[[0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4, 13, 22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26]],
    )
    torch.testing.assert_close(converted['block.bias'], bias)
    manifest = json.loads(result.manifest.read_text(encoding='utf-8'))
    assert manifest['source_kernel_layout_policy'] == 'torchsparse_k_i_o_volume_parity'
    assert manifest['target_kernel_layout'] == 'lattice_k_i_o_z_fastest'
    assert manifest['kernels'][0]['source_layout'] == 'torchsparse_k_i_o_x_fastest'
    assert manifest['kernels'][0]['row_permutation'] == [
        0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4, 13, 22, 7, 16, 25,
        2, 11, 20, 5, 14, 23, 8, 17, 26,
    ]


def test_checkpoint_conversion_preserves_even_torchsparse_row_order(tmp_path) -> None:
    source = tmp_path / 'legacy.pt'
    kernel = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
    torch.save({'state_dict': {'block.kernel': kernel}}, source)

    result = convert_torchsparse_checkpoint(
        source,
        tmp_path / 'weights.safetensors',
        kernel_specs={'block.kernel': (2, 1, 2)},
    )

    converted = load_file(result.output)
    torch.testing.assert_close(converted['block.weight'], kernel)
    manifest = json.loads(result.manifest.read_text(encoding='utf-8'))
    assert manifest['kernels'][0]['source_layout'] == 'torchsparse_k_i_o_z_fastest'
    assert manifest['kernels'][0]['row_permutation'] == [0, 1, 2, 3]


def test_model_state_migration_renames_kernels_and_adds_positions() -> None:
    source = {
        'block.kernel': torch.arange(27, dtype=torch.float32).reshape(27, 1, 1),
        'block.bias': torch.tensor([0.25]),
    }
    target = {
        'block.weight': torch.zeros((27, 1, 1)),
        'block.bias': torch.zeros(1),
        'block.kernel_positions': torch.arange(81, dtype=torch.int32).reshape(27, 3),
    }

    result = migrate_sparse_state_dict(
        source,
        target,
        kernel_sizes={'block': (3, 3, 3)},
        source_layout='torchsparse',
    )

    torch.testing.assert_close(
        result.state_dict['block.weight'],
        source['block.kernel'][
            [
                0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4, 13,
                22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26,
            ]
        ],
    )
    torch.testing.assert_close(result.state_dict['block.bias'], source['block.bias'])
    torch.testing.assert_close(
        result.state_dict['block.kernel_positions'], target['block.kernel_positions']
    )
    assert result.kernels[0].target_key == 'block.weight'


def test_model_state_migration_uses_me_hypercube_order_for_even_kernels() -> None:
    source = {'block.kernel': torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)}
    target = {
        'block.weight': torch.zeros((8, 1, 1)),
        'block.kernel_positions': torch.zeros((8, 3), dtype=torch.int32),
    }

    result = migrate_sparse_state_dict(
        source,
        target,
        kernel_sizes={'block': (2, 2, 2)},
        source_layout='minkowski_engine',
    )

    torch.testing.assert_close(
        result.state_dict['block.weight'],
        source['block.kernel'][[0, 4, 2, 6, 1, 5, 3, 7]],
    )
    assert result.kernels[0].source_layout == (
        'minkowski_engine_hypercube_k_i_o_x_fastest'
    )
