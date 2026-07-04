from typing import Dict

import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd

import torch_lattice
import torch_lattice.backend
import torch_lattice.backends

__all__ = ["ImplicitGEMMConvolutionFuntion", "implicit_gemm_forward_no_grad"]


def _resolve_wgrad_split_k(config: Dict, kernel_volume: int, ifsort: bool = False) -> int:
    value = config.get("wgrad_split_k", "auto")
    if value == "auto":
        if ifsort:
            return 32
        if kernel_volume <= 3:
            return 64
        if kernel_volume <= 9:
            return 16
        return 8
    return int(value)


def _implicit_gemm_forward_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    kmap: Dict,
    config: Dict,
    transposed: bool = False,
    *,
    return_context: bool = True,
) -> tuple[torch.Tensor, tuple]:
    sizes = kmap["sizes"]
    if not transposed:
        out_in_map = kmap["out_in_map"]
        reorder_out_in_map = kmap["reorder_out_in_map"]
        reduced_sorted_mask = kmap["reduced_sorted_mask"]
        reorder_loc = kmap["reorder_loc"]
        if return_context:
            out_in_map_bwd = kmap["out_in_map_bwd"]
            reorder_out_in_map_bwd = kmap["reorder_out_in_map_bwd"]
            reduced_sorted_mask_bwd_wgrad = kmap["reduced_sorted_mask_bwd_wgrad"]
            reduced_sorted_mask_bwd_dgrad = kmap["reduced_sorted_mask_bwd_dgrad"]
            reorder_loc_bwd = kmap["reorder_loc_bwd"]
    else:
        out_in_map = kmap["out_in_map_t"]
        reorder_out_in_map = kmap["reorder_out_in_map_t"]
        reduced_sorted_mask = kmap["reduced_sorted_mask_t"]
        reorder_loc = kmap["reorder_loc_t"]
        if return_context:
            out_in_map_bwd = kmap["out_in_map_bwd_t"]
            reorder_out_in_map_bwd = kmap["reorder_out_in_map_bwd_t"]
            reduced_sorted_mask_bwd_wgrad = kmap["reduced_sorted_mask_bwd_wgrad_t"]
            reduced_sorted_mask_bwd_dgrad = kmap["reduced_sorted_mask_bwd_dgrad_t"]
            reorder_loc_bwd = kmap["reorder_loc_bwd_t"]

    ifsort = config["ifsort"]

    input = input.contiguous()
    weight = weight.contiguous()
    active_kernel_offsets = kmap.get("active_kernel_offsets")
    full_kernel_volume = weight.size(0)
    if active_kernel_offsets is not None:
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
            weight_runtime = weight_cache[1]
        else:
            weight_runtime = weight.index_select(0, active_kernel_offsets).contiguous()
            kmap["_active_weight_cache"] = (weight_cache_key, weight_runtime)
    else:
        weight_runtime = weight

    center_only = (
        not transposed
        and kmap.get("IGEMM_center_only", False)
        and weight.size(0) % 2 == 1
        and sizes[0] == sizes[1]
    )
    if center_only:
        mid_kernel = weight.size(0) // 2
        output = input.matmul(weight[mid_kernel])
        if not return_context:
            return output.to(weight.dtype), ()
        wgrad_split_k = _resolve_wgrad_split_k(config, weight_runtime.size(0), ifsort)
        return output.to(weight.dtype), (
            input,
            weight_runtime,
            out_in_map_bwd,
            reorder_out_in_map_bwd,
            reduced_sorted_mask_bwd_wgrad,
            reduced_sorted_mask_bwd_dgrad,
            reorder_loc_bwd,
            transposed,
            wgrad_split_k,
            True,
            active_kernel_offsets,
            full_kernel_volume,
            ifsort,
        )

    if input.device.type != "cuda":
        raise NotImplementedError

    if torch.float16 in [input.dtype, weight.dtype]:
        input = input.to(torch.float16)
        weight_runtime = weight_runtime.to(torch.float16)

    num_out_feats = sizes[1] if not transposed else sizes[0]
    num_out_channels = weight_runtime.shape[-1]

    if not ifsort:
        output = torch_lattice.backend.conv_forward_implicit_gemm_cuda(
            input,
            weight_runtime,
            out_in_map,
            num_out_feats,
            num_out_channels,
            torch_lattice.backends.allow_tf32,
            torch_lattice.backends.allow_fp16,
        )
    else:
        output = torch_lattice.backend.conv_forward_implicit_gemm_sorted_cuda(
            input,
            weight_runtime,
            reorder_out_in_map,
            reduced_sorted_mask,
            reorder_loc,
            num_out_feats,
            num_out_channels,
            torch_lattice.backends.allow_tf32,
            torch_lattice.backends.allow_fp16,
        )
    if not return_context:
        return output.to(weight.dtype), ()

    wgrad_split_k = _resolve_wgrad_split_k(config, weight_runtime.size(0), ifsort)
    return output.to(weight.dtype), (
        input,
        weight_runtime,
        out_in_map_bwd,
        reorder_out_in_map_bwd,
        reduced_sorted_mask_bwd_wgrad,
        reduced_sorted_mask_bwd_dgrad,
        reorder_loc_bwd,
        transposed,
        wgrad_split_k,
        False,
        active_kernel_offsets,
        full_kernel_volume,
        ifsort,
    )


def implicit_gemm_forward_no_grad(
    input: torch.Tensor,
    weight: torch.Tensor,
    kmap: Dict,
    config: Dict,
    transposed: bool = False,
) -> torch.Tensor:
    output, _ = _implicit_gemm_forward_impl(
        input,
        weight,
        kmap,
        config,
        transposed,
        return_context=False,
    )
    return output


class ImplicitGEMMConvolutionFuntion(Function):  # TorchLattice++
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
        output, ctx.for_backwards = _implicit_gemm_forward_impl(
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
        (
            input,
            weight,
            out_in_map_bwd,
            reorder_out_in_map_bwd,
            reduced_sorted_mask_bwd_wgrad,
            reduced_sorted_mask_bwd_dgrad,
            reorder_loc_bwd,
            transposed,
            wgrad_split_k,
            center_only,
            active_kernel_offsets,
            full_kernel_volume,
            ifsort,
        ) = ctx.for_backwards

        grad_output = grad_output.contiguous()

        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        kernel_volume, ic, oc = weight.size()

        if center_only:
            mid_kernel = kernel_volume // 2
            grad_input = (
                grad_output.matmul(weight[mid_kernel].transpose(0, 1))
                if ctx.needs_input_grad[0]
                else None
            )
            if ctx.needs_input_grad[1]:
                grad_weight_runtime = torch.zeros_like(weight)
                grad_weight_runtime[mid_kernel] = input.transpose(0, 1).matmul(
                    grad_output
                )
            else:
                grad_weight_runtime = None
            if grad_weight_runtime is not None and active_kernel_offsets is not None:
                grad_weight = torch.zeros(
                    (full_kernel_volume, weight.size(1), weight.size(2)),
                    dtype=grad_weight_runtime.dtype,
                    device=grad_weight_runtime.device,
                )
                grad_weight.index_copy_(0, active_kernel_offsets, grad_weight_runtime)
            else:
                grad_weight = grad_weight_runtime
            return (grad_input, grad_weight, None, None, None)

        if grad_output.device.type != "cuda":
            raise NotImplementedError

        if ifsort and kernel_volume < 32:  # sort mode
            # dgrad
            grad_input = torch_lattice.backend.conv_forward_implicit_gemm_sorted_cuda(
                grad_output,
                weight.transpose(2, 1).contiguous(),
                reorder_out_in_map_bwd,
                reduced_sorted_mask_bwd_dgrad,
                reorder_loc_bwd,
                input.size(0),
                input.size(1),
                torch_lattice.backends.allow_tf32,
                torch_lattice.backends.allow_fp16,
            )

            # wgrad
            grad_weight = (
                (
                    torch_lattice.backend.conv_backward_wgrad_implicit_gemm_sorted_cuda(
                        grad_output,
                        input,
                        reorder_out_in_map_bwd,
                        reduced_sorted_mask_bwd_wgrad,
                        reorder_loc_bwd,
                        wgrad_split_k,
                        torch_lattice.backends.allow_tf32,
                        torch_lattice.backends.allow_fp16,
                    )
                )
                .reshape(kernel_volume, oc, ic)
                .transpose(2, 1)
                .contiguous()
            )

        else:  # unsort mode
            # dgrad
            grad_input = torch_lattice.backend.conv_forward_implicit_gemm_cuda(
                grad_output,
                weight.transpose(2, 1).contiguous(),
                out_in_map_bwd,
                input.size(0),
                input.size(1),
                torch_lattice.backends.allow_tf32,
                torch_lattice.backends.allow_fp16,
            )

            # wgrad
            grad_weight = (
                (
                    torch_lattice.backend.conv_backward_wgrad_implicit_gemm_cuda(
                        grad_output,
                        input,
                        out_in_map_bwd,
                        wgrad_split_k,
                        torch_lattice.backends.allow_tf32,
                        torch_lattice.backends.allow_fp16,
                    )
                )
                .reshape(kernel_volume, oc, ic)
                .transpose(2, 1)
                .contiguous()
            )
        if grad_weight is not None and active_kernel_offsets is not None:
            grad_weight_runtime = grad_weight
            grad_weight = torch.zeros(
                (full_kernel_volume, grad_weight_runtime.size(1), grad_weight_runtime.size(2)),
                dtype=grad_weight_runtime.dtype,
                device=grad_weight_runtime.device,
            )
            grad_weight.index_copy_(0, active_kernel_offsets, grad_weight_runtime)
        return (grad_input, grad_weight, None, None, None)
