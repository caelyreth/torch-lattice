from __future__ import annotations

import pytest
import torch
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.nn import functional as F


def _line(
    *,
    device: torch.device | str = "cpu",
    stride: int | tuple[int, int, int] = 1,
    requires_grad: bool = False,
) -> SparseTensor:
    return SparseTensor(
        feats=torch.tensor(
            [[1.0], [2.0], [3.0]],
            device=device,
            requires_grad=requires_grad,
        ),
        coords=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=torch.int32,
            device=device,
        ),
        stride=stride,
        spatial_range=(1, 3, 1, 1),
    )


def test_normalized_subm_matches_weight_norm_formula(cuda_device) -> None:
    x = _line(device=cuda_device)
    weight = torch.tensor([1.0, 2.0, 3.0], device=cuda_device).reshape(3, 1, 1)
    bias = torch.tensor([0.25], device=cuda_device)

    actual = F.normalized_conv3d(
        x,
        weight,
        kernel_size=(3, 1, 1),
        bias=bias,
        subm=True,
    )
    numerator = F.conv3d(x, weight, kernel_size=(3, 1, 1), subm=True)
    denominator = F.conv3d(
        x.replace(feats=torch.ones_like(x.feats)),
        weight.square(),
        kernel_size=(3, 1, 1),
        subm=True,
    )
    expected = numerator.feats / torch.sqrt(denominator.feats + 1e-8) + bias

    assert actual.coord_key == x.coord_key
    torch.testing.assert_close(actual.feats, expected)


def test_normalized_pointwise_bypasses_weight_norm() -> None:
    x = SparseTensor(
        feats=torch.tensor([[2.0, 3.0]]),
        coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
    )
    module = spnn.NormalizedSubmConv3d(2, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[[4.0], [5.0]]]))

    out = module(x)

    torch.testing.assert_close(out.feats, torch.tensor([[23.0]]))


def test_normalized_subm_preserves_weight_dependent_gradients(
    cuda_device,
) -> None:
    x = _line(device=cuda_device, requires_grad=True)
    weight = torch.tensor(
        [[[1.0]], [[2.0]], [[3.0]]],
        device=cuda_device,
        requires_grad=True,
    )

    out = F.normalized_conv3d(
        x, weight, kernel_size=(3, 1, 1), subm=True, training=True
    )
    out.feats.sum().backward()

    assert x.feats.grad is not None
    assert weight.grad is not None
    assert torch.count_nonzero(weight.grad).item() > 0


def test_normalized_generative_transpose_uses_one_support(
    cuda_device,
) -> None:
    x = SparseTensor(
        feats=torch.tensor([[2.0]], device=cuda_device),
        coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=cuda_device),
        stride=2,
        spatial_range=(1, 2, 2, 2),
    )
    config = F.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap"
    module = spnn.NormalizedGenerativeConvTranspose3d(
        1,
        1,
        kernel_size=2,
        stride=2,
        bias=False,
        config=config,
    ).to(cuda_device)
    with torch.no_grad():
        module.weight.copy_(torch.arange(1.0, 9.0, device=cuda_device).reshape(8, 1, 1))

    out = module(x)
    numerator = F.conv3d(
        x,
        module.weight,
        kernel_size=module.kernel_size,
        stride=module.stride,
        transposed=True,
        generative=True,
        config=config,
    )
    denominator = F.conv3d(
        x.replace(feats=torch.ones_like(x.feats)),
        module.weight.square(),
        kernel_size=module.kernel_size,
        stride=module.stride,
        transposed=True,
        generative=True,
        config=config,
    )
    expected = numerator.feats / torch.sqrt(denominator.feats + module.eps)

    assert torch.equal(out.coords, numerator.coords)
    torch.testing.assert_close(out.feats, expected)


def test_normalized_transpose_reuses_forward_inverse_relation(
    cuda_device,
) -> None:
    x = _line(device=cuda_device)
    down = spnn.Conv3d(1, 1, kernel_size=(2, 1, 1), stride=2, bias=False).to(
        cuda_device
    )
    up = spnn.NormalizedConvTranspose3d(
        1, 1, kernel_size=(2, 1, 1), stride=2, bias=False
    ).to(cuda_device)
    with torch.no_grad():
        down.weight.fill_(1.0)
        up.weight.fill_(1.0)

    reduced = down(x)
    restored = up(reduced)

    assert restored.coord_key == x.coord_key
    assert torch.equal(restored.coords, x.coords)


def test_target_transpose_matches_indexed_geometry(cuda_device) -> None:
    source = _line(device=cuda_device, stride=(2, 1, 1))
    target = SparseTensor(
        feats=torch.zeros((5, 1), device=cuda_device),
        coords=torch.tensor(
            [[0, index, 0, 0] for index in range(5)],
            dtype=torch.int32,
            device=cuda_device,
        ),
        stride=1,
        spatial_range=(1, 5, 1, 1),
    )
    module = spnn.ConvTranspose3d(
        1,
        1,
        kernel_size=(3, 1, 1),
        stride=(2, 1, 1),
        padding=(1, 0, 0),
        bias=False,
    ).to(cuda_device)
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor([1.0, 2.0, 3.0], device=cuda_device).reshape(3, 1, 1)
        )

    out = module(source, target)

    assert torch.equal(out.coords, target.coords)
    torch.testing.assert_close(
        out.feats,
        torch.tensor([[2.0], [5.0], [4.0], [9.0], [6.0]], device=cuda_device),
    )


def test_normalized_target_transpose_matches_weight_norm_and_gradients(
    cuda_device,
) -> None:
    source = _line(
        device=cuda_device, stride=(2, 1, 1), requires_grad=True
    )
    target = SparseTensor(
        feats=torch.zeros((5, 1), device=cuda_device),
        coords=torch.tensor(
            [[0, index, 0, 0] for index in range(5)],
            dtype=torch.int32,
            device=cuda_device,
        ),
        stride=1,
        spatial_range=(1, 5, 1, 1),
    )
    weight = torch.tensor(
        [[[1.0]], [[2.0]], [[3.0]]],
        device=cuda_device,
        requires_grad=True,
    )
    kwargs = {
        "kernel_size": (3, 1, 1),
        "stride": (2, 1, 1),
        "padding": (1, 0, 0),
        "transposed": True,
        "coordinates": target,
        "training": True,
    }

    actual = F.normalized_conv3d(source, weight, **kwargs)
    numerator = F.conv3d(source, weight, **kwargs)
    denominator = F.conv3d(
        source.replace(feats=torch.ones_like(source.feats)),
        weight.square(),
        **kwargs,
    )
    expected = numerator.feats / torch.sqrt(denominator.feats + 1e-8)
    torch.testing.assert_close(actual.feats, expected)

    actual.feats.sum().backward()
    assert source.feats.grad is not None
    assert weight.grad is not None
    assert torch.count_nonzero(weight.grad).item() > 0


def test_normalized_convolution_rejects_nonpositive_eps() -> None:
    with pytest.raises(ValueError, match="eps must be positive"):
        spnn.NormalizedSubmConv3d(1, 1, eps=0.0)
