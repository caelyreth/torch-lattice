from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from safetensors.torch import load_file

pytestmark = pytest.mark.conformance


def test_fuzz_fixture_generator_writes_replayable_case_tree(tmp_path: Path) -> None:
    output = tmp_path / 'fuzz'
    archive = tmp_path / 'fuzz.tar.gz'
    subprocess.run(
        [
            sys.executable,
            '-m',
            'torch_lattice_conformance.generate',
            '--cases',
            '3',
            '--seed',
            '17',
            '--train-steps',
            '0',
            '--families',
            'sparse_branch,target_branch,point_voxel',
            '--output',
            str(output),
            '--archive',
            str(archive),
        ],
        cwd=tmp_path,
        check=True,
    )

    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['schema'] == 'torch_lattice_fuzz_fixtures.v1'
    assert manifest['case_count'] == 3
    assert archive.exists()

    for item in manifest['cases']:
        case = output / item['name']
        assert (case / 'case.json').exists()
        assert (case / 'graph.mlir').exists()
        assert (case / 'weights.safetensors').exists()
        assert (case / 'inputs.safetensors').exists()
        assert (case / 'expected.safetensors').exists()
        inputs = load_file(case / 'inputs.safetensors')
        if item['family'] != 'point_voxel':
            assert {'x_coords', 'x_features', 'x_active'} <= inputs.keys()

    with tarfile.open(archive, 'r:gz') as handle:
        names = set(handle.getnames())
    assert f'{output.name}/manifest.json' in names


def test_torchsparse_migration_compatibility_tool_smoke(tmp_path: Path) -> None:
    available = subprocess.run(
        [sys.executable, '-c', 'import torchsparse'],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if available.returncode != 0:
        return

    output = tmp_path / 'compat'
    subprocess.run(
        [
            sys.executable,
            '-m',
            'torch_lattice_conformance.migration',
            'all',
            '--cases',
            '7',
            '--seed',
            '20260709',
            '--device',
            'cpu',
            '--output',
            str(output),
        ],
        cwd=tmp_path,
        check=True,
    )
    report = json.loads((output / 'report.json').read_text())
    assert report['summary']['failed'] == 0
    assert set(report['summary']['families']) == {
        'batchnorm_chain',
        'branch_add',
        'branch_cat',
        'global_pool',
        'pointwise_chain',
        'spatial_subm_mapping',
        'stride2_forward',
    }
