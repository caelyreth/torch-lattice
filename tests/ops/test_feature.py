from __future__ import annotations

import pytest
import torch

from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F
from torch_lattice.nn.utils import fapply
from tests.support import assert_same_sparse_coords, line_sparse_case

pytestmark = [pytest.mark.ops, pytest.mark.feature]


def test_dense_feature_transform_matches_reference_and_preserves_coordinates() -> None:
    x = line_sparse_case(channels=2, rows=2).tensor()
    linear = torch.nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[2.0, 3.0], [5.0, 7.0]]))
        linear.bias.copy_(torch.tensor([1.0, -1.0]))

    out = fapply(x, linear)

    torch.testing.assert_close(out.feats, linear(x.feats))
    assert_same_sparse_coords(out, x)


def test_activation_feature_ops_keep_coordinate_contract() -> None:
    x = line_sparse_case(channels=2, rows=2).tensor()
    x.feats = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

    torch.testing.assert_close(F.relu(x, inplace=False).feats, torch.tensor([[0.0, 2.0], [3.0, 0.0]]))
    torch.testing.assert_close(F.leaky_relu(x, negative_slope=0.1, inplace=False).feats, torch.tensor([[-0.1, 2.0], [3.0, -0.4]]))
    for module in (spnn.SiLU(), spnn.GELU(), spnn.Sigmoid(), spnn.Tanh(), spnn.Softplus()):
        out = module(x)
        assert out.feats.shape == x.feats.shape
        assert_same_sparse_coords(out, x)


def test_feature_ops_support_autograd() -> None:
    x = line_sparse_case(channels=2, rows=2).tensor()
    feats = x.feats.detach().clone().requires_grad_(True)
    linear = torch.nn.Linear(2, 2)
    sparse = type(x)(feats, x.coords, spatial_range=x.spatial_range)

    loss = F.relu(fapply(sparse, linear), inplace=False).feats.sum()
    loss.backward()

    assert feats.grad is not None
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
