import torch

from torch_lattice import SparseTensor

__all__ = ["global_avg_pool", "global_max_pool"]


def global_avg_pool(inputs: SparseTensor) -> torch.Tensor:
    if (
        inputs.spatial_range is not None
        and len(inputs.spatial_range) > 0
        and inputs.spatial_range[0] == 1
    ):
        return torch.mean(inputs.feats, dim=0, keepdim=True)

    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.mean(input, dim=0)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def global_max_pool(inputs: SparseTensor) -> torch.Tensor:
    if (
        inputs.spatial_range is not None
        and len(inputs.spatial_range) > 0
        and inputs.spatial_range[0] == 1
    ):
        return torch.max(inputs.feats, dim=0, keepdim=True)[0]

    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.max(input, dim=0)[0]
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs
