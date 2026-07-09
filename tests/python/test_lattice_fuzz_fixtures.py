from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path


def test_fuzz_fixture_generator_writes_replayable_case_tree(tmp_path: Path) -> None:
    output = tmp_path / 'fuzz'
    archive = tmp_path / 'fuzz.tar.gz'
    subprocess.run(
        [
            sys.executable,
            'tools/build_lattice_fuzz_fixtures.py',
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
        cwd=Path(__file__).resolve().parents[2],
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

    with tarfile.open(archive, 'r:gz') as handle:
        names = set(handle.getnames())
    assert f'{output.name}/manifest.json' in names
