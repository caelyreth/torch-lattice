from typing import Dict

import torch
from torch.autograd import Function

import torch_lattice
import torch_lattice.backend

__all__ = ["GatherScatterConvolutionFuntion", "gather_scatter_forward_no_grad"]


def _select_active_weight(weight: torch.Tensor, kmap: Dict) -> torch.Tensor:
    active_kernel_offsets = kmap.get("active_kernel_offsets")
    if active_kernel_offsets is None:
        return weight
    active_kernel_offsets = active_kernel_offsets.to(
        device=weight.device, dtype=torch.long
    )
    weight_cache_key = (
        int(weight.data_ptr()),
        int(getattr(weight, "_version", 0)),
        weight.device,
        weight.dtype,
    )
    weight_cache = kmap.get("_active_weight_cache")
    if weight_cache is not None and weight_cache[0] == weight_cache_key:
        return weight_cache[1]
    weight_runtime = weight.index_select(0, active_kernel_offsets).contiguous()
    kmap["_active_weight_cache"] = (weight_cache_key, weight_runtime)
    return weight_runtime


def _gather_scatter_forward_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    kmap: Dict,
    config: Dict,
    transposed: bool = False,
    *,
    return_context: bool = True,
) -> tuple[torch.Tensor, tuple]:
    nbmaps = kmap["nbmaps"]
    nbsizes = kmap["nbsizes"]
    nbsizes_cpu = kmap.get("nbsizes_cpu")
    if nbsizes_cpu is None:
        nbsizes_cpu = nbsizes.int().cpu().contiguous()
    sizes = kmap["sizes"]
    input_mask = kmap["input_mask"]
    output_mask = kmap["output_mask"]
    epsilon = config["epsilon"]
    mm_thresh = config["mm_thresh"]

    conv_mode = 0
    buffer = input.new_empty((0,))
    if torch_lattice.backends.benchmark:  # type: ignore
        conv_mode = 1 if (epsilon == 0.0 and mm_thresh == 0) else 2
        required = int(nbsizes_cpu.sum().item()) * (
            int(input.shape[1]) + int(weight.shape[-1])
        )
        buffer = input.new_empty((required,))

    input = input.contiguous()
    weight = weight.contiguous()
    if not return_context:
        weight = _select_active_weight(weight, kmap)
    nbmaps = nbmaps.int().contiguous()

    if input.device.type == "cuda":
        if torch.float16 in [input.dtype, weight.dtype]:
            input = input.to(torch.float16)
            weight = weight.to(torch.float16)

        output = torch_lattice.backend.conv_forward_gather_scatter_cuda(
            input,
            weight,
            nbmaps,
            nbsizes_cpu,
            input_mask,
            output_mask,
            sizes[1] if not transposed else sizes[0],
            epsilon,
            int(mm_thresh),
            conv_mode,
            transposed,
            buffer,
        )
    else:
        output_size = sizes[1] if not transposed else sizes[0]
        output = torch.zeros(
            output_size,
            weight.size(-1),
            dtype=input.dtype,
            device=input.device,
        )
        if input.device.type == "cpu":
            torch_lattice.backend.conv_forward_gather_scatter_cpu(
                input, output, weight, nbmaps, nbsizes_cpu, transposed
            )
        else:
            cur_st = 0
            for kernel_idx in range(weight.shape[0]):
                cur_ed = cur_st + nbsizes_cpu[kernel_idx]
                in_map = nbmaps[cur_st:cur_ed, 0].long()
                out_map = nbmaps[cur_st:cur_ed, 1].long()
                cur_st += nbsizes_cpu[kernel_idx]

                if transposed:
                    in_map, out_map = out_map, in_map

                cur_feat = input[in_map]
                cur_feat = torch.mm(cur_feat, weight[kernel_idx])
                output[out_map] += cur_feat
    if not return_context:
        return output.to(weight.dtype), ()

    return output.to(weight.dtype), (input, weight, nbmaps, nbsizes_cpu, transposed)


def gather_scatter_forward_no_grad(
    input: torch.Tensor,
    weight: torch.Tensor,
    kmap: Dict,
    config: Dict,
    transposed: bool = False,
) -> torch.Tensor:
    output, _ = _gather_scatter_forward_impl(
        input,
        weight,
        kmap,
        config,
        transposed,
        return_context=False,
    )
    return output


class GatherScatterConvolutionFuntion(Function):  # TorchLattice_v2
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        kmap: Dict,
        config: Dict,
        transposed: bool = False,
    ) -> torch.Tensor:
        output, ctx.for_backwards = _gather_scatter_forward_impl(
            input,
            weight,
            kmap,
            config,
            transposed,
        )
        return output.to(weight.dtype)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, nbmaps, nbsizes_cpu, transposed = ctx.for_backwards

        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        if grad_output.device.type == "cuda":
            torch_lattice.backend.conv_backward_gather_scatter_cuda(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes_cpu,
                transposed,
            )
        elif grad_output.device.type == "cpu":
            torch_lattice.backend.conv_backward_gather_scatter_cpu(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes_cpu,
                transposed,
            )
        else:
            raise NotImplementedError(
                f"gather-scatter backward is not implemented for {grad_output.device.type}"
            )
        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
        )
