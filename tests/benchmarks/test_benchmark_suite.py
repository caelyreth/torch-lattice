from __future__ import annotations

import pytest
import torch

from torch_lattice_bench.cases import all_cases

pytestmark = pytest.mark.benchmark

def test_benchmark_catalog_exposes_operator_and_module_groups() -> None:
    cases = all_cases(
        'smoke',
        groups=('tensor', 'hash', 'dense', 'kmap', 'conv', 'nn', 'train'),
        n_values=(16,),
        channels=(4,),
        layouts=('block2',),
        dtype='fp32',
        device=torch.device('cpu'),
    )
    names = {f'{case.group}/{case.name}' for case in cases}

    assert 'tensor/relu' in names
    assert 'hash/sphash' in names
    assert 'dense/spvoxelize_forward' in names
    assert 'kmap/build_kmap_igemm_unsorted_subm_k3' in names
    assert 'conv/subm3_implicit_gemm_unsorted' in names
    assert 'nn/sparse_classifier_module' in names
    assert 'train/conv3_implicit_unsorted_forward_backward' in names


def test_benchmark_catalog_train_cases_are_backward_only() -> None:
    cases = all_cases(
        'smoke',
        groups=('train',),
        n_values=(16,),
        channels=(4,),
        layouts=('block2',),
        dtype='fp32',
        device=torch.device('cpu'),
    )

    assert cases
    assert all(case.supports('backward') for case in cases)
    assert not any(case.supports('hot_op') for case in cases)
