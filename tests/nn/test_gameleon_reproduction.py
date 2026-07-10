from __future__ import annotations

import torch

from torch_lattice.nn.functional.conv import Dataflow, conv_config
from torch_lattice_conformance.e2e import (
    GameleonReproductionBlock,
    _gameleon_input,
)


def test_gameleon_reproduction_block_trains_on_cuda(cuda_device) -> None:
    model = GameleonReproductionBlock().to(cuda_device)
    x = _gameleon_input().to(cuda_device)
    target = torch.tanh(x.feats * 0.75)
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    config = conv_config.get_default_conv_config()
    config.dataflow = Dataflow.GatherScatter
    config.ifsort = False
    previous = conv_config.get_global_conv_config()
    conv_config.set_global_conv_config(config)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output.feats, target)
        loss.backward()

        assert torch.isfinite(loss)
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

        optimizer.step()
    finally:
        if previous is None:
            conv_config.clear_global_conv_config()
        else:
            conv_config.set_global_conv_config(previous)

    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
