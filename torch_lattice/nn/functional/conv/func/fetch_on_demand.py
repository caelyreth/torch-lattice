from typing import Dict

import torch
from torch.autograd import Function

import torch_lattice
import torch_lattice.backend
from torch_lattice.nn.functional.conv.kmap.layout import (
    fod_neighbor_map,
    neighbor_pairs,
)

__all__ = ["FetchOnDemandConvolutionFuntion", "fetch_on_demand_forward_no_grad"]

_FOD_FUSED_MAX_QMAPSIZE = 65535 * 64


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


def _fetch_on_demand_forward_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    kmap: Dict,
    config: Dict,
    transposed: bool = False,
    *,
    return_context: bool = True,
) -> tuple[torch.Tensor, tuple]:
    fod_map = fod_neighbor_map(kmap)
    pairs = neighbor_pairs(kmap)
    nbsizes = kmap["nbsizes"]
    nbsizes_cpu = kmap.get("nbsizes_cpu")
    sizes = kmap["sizes"]

    mapsize = fod_map.size(1)

    input = input.contiguous()
    weight = weight.contiguous()
    if not return_context:
        weight = _select_active_weight(weight, kmap)
    nbsizes_cpu = nbsizes_cpu if nbsizes_cpu is not None else nbsizes.cpu()

    if input.device.type != "cuda":
        raise NotImplementedError("fetch-on-demand convolution requires CUDA")

    if torch.float16 in [input.dtype, weight.dtype]:
        input = input.to(torch.float16)
        weight = weight.to(torch.float16)

    kernel_volume = weight.size(0)
    mid_kernel = kernel_volume // 2
    output_size = sizes[1] if not transposed else sizes[0]
    center_only = kmap.get("FOD_center_only")
    if center_only is None:
        center_only = (
            not transposed
            and kernel_volume % 2 == 1
            and input.size(0) == output_size
            and int(nbsizes_cpu[mid_kernel]) == mapsize
            and int(nbsizes_cpu.sum()) == mapsize
        )
    if (
        not transposed
        and kernel_volume % 2 == 1
        and input.size(0) == output_size
        and center_only
    ):
        output = input.matmul(weight[mid_kernel])
        if not return_context:
            return output.to(weight.dtype), ()
        return output.to(weight.dtype), (
            input,
            weight,
            pairs,
            nbsizes_cpu,
            transposed,
            True,
        )

    qmapsize = kmap.get("qmapsize")
    qmapsize_int = (
        (int(qmapsize.item()) if hasattr(qmapsize, "item") else int(qmapsize))
        if qmapsize is not None
        else 0
    )
    use_fusion = (
        config["FOD_fusion"]
        and qmapsize_int > 0
        and qmapsize_int <= _FOD_FUSED_MAX_QMAPSIZE
    )

    if use_fusion:
        nbaddrs = kmap["nbaddrs"]
        qnbaddrs = kmap["qnbaddrs"]
        output = torch_lattice.backend.conv_forward_fetch_on_demand_cuda(
            input,
            weight,
            fod_map,
            mapsize,
            nbaddrs,
            qnbaddrs,
            output_size,
            qmapsize_int,
            transposed,
            torch_lattice.backends.allow_tf32,
            torch_lattice.backends.allow_fp16,
        )
    else:
        output = torch_lattice.backend.conv_forward_fetch_on_demand_no_fusion_cuda(
            input,
            weight,
            fod_map,
            nbsizes_cpu,
            mapsize,
            output_size,
            transposed,
            torch_lattice.backends.allow_tf32,
            torch_lattice.backends.allow_fp16,
        )

    if not return_context:
        return output.to(weight.dtype), ()

    return output.to(weight.dtype), (
        input,
        weight,
        pairs,
        nbsizes_cpu,
        transposed,
        False,
    )


def fetch_on_demand_forward_no_grad(
    input: torch.Tensor,
    weight: torch.Tensor,
    kmap: Dict,
    config: Dict,
    transposed: bool = False,
) -> torch.Tensor:
    output, _ = _fetch_on_demand_forward_impl(
        input,
        weight,
        kmap,
        config,
        transposed,
        return_context=False,
    )
    return output


class FetchOnDemandConvolutionFuntion(Function):
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
        output, ctx.for_backwards = _fetch_on_demand_forward_impl(
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
        input, weight, pairs, nbsizes_cpu, transposed, center_only = ctx.for_backwards

        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        if center_only:
            grad_output = grad_output.contiguous()
            mid_kernel = weight.size(0) // 2
            grad_input = (
                grad_output.matmul(weight[mid_kernel].transpose(0, 1))
                if ctx.needs_input_grad[0]
                else None
            )
            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
                grad_weight[mid_kernel] = input.transpose(0, 1).matmul(grad_output)
            else:
                grad_weight = None
            return (grad_input, grad_weight, None, None, None, None)

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        if grad_output.device.type == "cuda":
            torch_lattice.backend.conv_backward_gather_scatter_cuda(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                pairs,
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
                pairs,
                nbsizes_cpu,
                transposed,
            )
        else:
            raise NotImplementedError(
                f"fetch-on-demand backward is not implemented for {grad_output.device.type}"
            )
        return (grad_input, grad_weight, None, None, None, None)
