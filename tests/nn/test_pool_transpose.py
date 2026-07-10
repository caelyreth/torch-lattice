from __future__ import annotations

import torch

from torch_lattice import SparseTensor
from torch_lattice import nn as spnn


def _case(device: torch.device) -> tuple[SparseTensor, SparseTensor]:
    coarse = SparseTensor(
        feats=torch.tensor([[2.0], [6.0]], device=device),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
            device=device,
        ),
        stride=2,
    )
    target = SparseTensor(
        feats=torch.zeros((3, 1), device=device),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=torch.int32,
            device=device,
        ),
    )
    return coarse, target


def test_pool_transpose_targets_support_and_averages_contributors(
    selected_device: torch.device,
) -> None:
    coarse, target = _case(selected_device)
    module = spnn.PoolTranspose3d(kernel_size=(3, 1, 1), stride=2, padding=(1, 0, 0))

    out = module(coarse, target)

    assert torch.equal(out.coords, target.coords)
    torch.testing.assert_close(
        out.feats, torch.tensor([[2.0], [4.0], [6.0]], device=selected_device)
    )


def test_pool_transpose_generates_support(selected_device: torch.device) -> None:
    coarse = SparseTensor(
        feats=torch.tensor([[3.0]], device=selected_device),
        coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=selected_device),
        stride=2,
    )

    out = spnn.PoolTranspose3d(kernel_size=(2, 1, 1), stride=2)(coarse)

    assert out.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    torch.testing.assert_close(
        out.feats, torch.tensor([[3.0], [3.0]], device=selected_device)
    )


def test_trilinear_upsample_targets_support(selected_device: torch.device) -> None:
    coarse = SparseTensor(
        feats=torch.tensor([[2.0], [6.0]], device=selected_device, requires_grad=True),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
            device=selected_device,
        ),
        stride=(2, 1, 1),
    )
    target = SparseTensor(
        feats=torch.zeros((4, 1), device=selected_device),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=torch.int32,
            device=selected_device,
        ),
    )

    out = spnn.TrilinearUpsample3d(stride=(2, 1, 1))(coarse, target)

    torch.testing.assert_close(
        out.feats,
        torch.tensor([[2.0], [4.0], [6.0], [6.0]], device=selected_device),
    )
    out.feats.sum().backward()
    assert coarse.feats.grad is not None
