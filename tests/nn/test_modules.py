from __future__ import annotations

import pytest
import torch

from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F
from tests.support import (
    assert_same_sparse_coords,
    line_sparse_case,
    two_batch_sparse_case,
)

pytestmark = pytest.mark.nn


def test_activation_modules_delegate_to_feature_tensor() -> None:
    x = line_sparse_case(channels=2, rows=2).tensor()
    x.feats = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

    out = spnn.ReLU()(x)

    torch.testing.assert_close(out.feats, F.relu(x, inplace=False).feats)
    assert_same_sparse_coords(out, x)


def test_norm_modules_preserve_sparse_coordinates() -> None:
    x = line_sparse_case(channels=4, rows=4).tensor()

    modules = (
        spnn.BatchNorm(4).eval(),
        spnn.LayerNorm(4),
        spnn.RMSNorm(4),
        spnn.GroupNorm(2, 4),
    )
    for module in modules:
        out = module(x)
        assert out.feats.shape == x.feats.shape
        assert_same_sparse_coords(out, x)


def test_group_norm_multi_batch_matches_per_sample_reference() -> None:
    x = two_batch_sparse_case().tensor()
    norm = spnn.GroupNorm(num_groups=1, num_channels=2, affine=False)

    out = norm(x)
    ref = torch.cat(
        [
            torch.nn.functional.group_norm(
                x.feats[x.coords[:, 0] == batch].t().reshape(1, 2, -1),
                num_groups=1,
                weight=None,
                bias=None,
                eps=norm.eps,
            )
            .reshape(2, -1)
            .t()
            for batch in (0, 1)
        ],
        dim=0,
    )

    torch.testing.assert_close(out.feats, ref)
    assert_same_sparse_coords(out, x)


def test_conv_modules_use_pointwise_cpu_safe_path() -> None:
    x = line_sparse_case(channels=1, rows=3).tensor()
    conv = spnn.Conv3d(1, 1, kernel_size=1, bias=False)
    subm = spnn.SubmConv3d(1, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(2.0)
        subm.weight.fill_(2.0)

    forward = conv(x)
    support_preserving = subm(x)

    torch.testing.assert_close(forward.coords, x.coords)
    torch.testing.assert_close(support_preserving.coords, x.coords)
    torch.testing.assert_close(forward.feats, x.feats * 2)
    torch.testing.assert_close(support_preserving.feats, x.feats * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_conv_modules_cuda_forward_and_subm_use_distinct_support_semantics() -> None:
    x = line_sparse_case(channels=1, rows=3).tensor(device="cuda")
    conv = spnn.Conv3d(1, 1, kernel_size=(3, 1, 1), bias=False).cuda()
    subm = spnn.SubmConv3d(1, 1, kernel_size=(3, 1, 1), bias=False).cuda()
    with torch.no_grad():
        conv.weight.fill_(1.0)
        subm.weight.fill_(1.0)

    forward = conv(x)
    support_preserving = subm(x)

    assert forward.feats.shape[0] < x.feats.shape[0]
    torch.testing.assert_close(support_preserving.coords, x.coords)
    assert support_preserving.feats.shape[0] == x.feats.shape[0]


def test_global_pool_module_matches_functional_reduction() -> None:
    x = two_batch_sparse_case().tensor()

    torch.testing.assert_close(spnn.GlobalAvgPool()(x), F.global_avg_pool(x))
    torch.testing.assert_close(spnn.GlobalMaxPool()(x), F.global_max_pool(x))
