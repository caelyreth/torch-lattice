from typing import List, Dict, Optional, Tuple, Union

# import numpy as np
import torch

import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice.utils import make_ntuple

from .func import *
from ..relation import build_target_out_in_map, gather_scatter_kmap_from_out_in_map

__all__ = ["conv3d", "target_conv3d"]


def _make_kmap_cache_key(
    tensor_stride: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    subm: bool,
    config: Dict,
    training: bool,
) -> Tuple:
    return (
        tensor_stride,
        kernel_size,
        stride,
        padding,
        dilation,
        bool(subm),
        config.kmap_mode,
        config.downsample_mode,
        config.dataflow,
        bool(config.ifsort),
        bool(config.FOD_fusion) if getattr(config.dataflow, "name", None) == "FetchOnDemand" else None,
        bool(config.get("IGEMM_center_only", False)) if getattr(config.dataflow, "name", None) == "ImplicitGEMM" else None,
        int(config.split_mask_num),
        int(config.split_mask_num_bwd) if training else 0,
        config.get("wgrad_split_k", "auto") if training and getattr(config.dataflow, "name", None) == "ImplicitGEMM" else 0,
        bool(training),
    )


def conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    config: Dict = None,
    subm: bool = False,
    transposed: bool = False,
    generative: bool = False,
    training: bool = False,
) -> SparseTensor:
    from torch_lattice.nn import functional as F

    feats, coords = input.feats, input.coords
    kernel_size = make_ntuple(kernel_size, ndim=3)
    # kernel_volume = np.prod(kernel_size)
    stride = make_ntuple(stride, ndim=3)
    padding = make_ntuple(padding, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)
    if subm:
        if transposed or generative:
            raise ValueError("submanifold convolution cannot be transposed or generative.")
        if stride != (1, 1, 1):
            raise ValueError("submanifold convolution requires stride=1.")

    conv_mode = F.get_conv_mode()
    if config is None:
        config = F.conv_config.get_global_conv_config()
        if config is None:
            config = F.conv_config.get_default_conv_config(
                conv_mode=conv_mode, training=training
            )

    # TODO: Deal with kernel volume > 32. (Split mask or unsort)

    dataflow = config.dataflow
    kmap_mode = config.kmap_mode
    inference_no_grad = (
        not torch.is_grad_enabled()
        or (not feats.requires_grad and not weight.requires_grad)
    )

    if dataflow == F.Dataflow.ImplicitGEMM:
        ConvolutionFunction = ImplicitGEMMConvolutionFuntion
        no_grad_forward = implicit_gemm_forward_no_grad
    elif dataflow == F.Dataflow.GatherScatter:
        ConvolutionFunction = GatherScatterConvolutionFuntion
        no_grad_forward = gather_scatter_forward_no_grad
        config.ifsort = False
    elif dataflow == F.Dataflow.FetchOnDemand:
        ConvolutionFunction = FetchOnDemandConvolutionFuntion
        no_grad_forward = fetch_on_demand_forward_no_grad
        config.ifsort = False
    elif (
        dataflow == F.Dataflow.CodedCSR
    ):  # Placeholder for PCEngine integration. Mode name can be modified.
        config.ifsort = False
        assert 0, "CodedCSR has not been integrated."
    else:
        raise ValueError("unsupported dataflow: {}".format(dataflow))

    if kernel_size == (1, 1, 1) and stride == (1, 1, 1) and dilation == (1, 1, 1):
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=input.stride,
            spatial_range=input.spatial_range,
        )
    elif not transposed:
        kmap_key = _make_kmap_cache_key(
            input.stride, kernel_size, stride, padding, dilation, subm, config, training
        )
        kmap = input._caches.kmaps.get(kmap_key)

        output_stride = tuple(input.stride[k] * stride[k] for k in range(3))
        hashmap_stride = output_stride if kmap_mode == "hashmap_on_the_fly" else input.stride
        hashmap = input._caches.hashmaps.get(hashmap_stride)
        if hashmap is None:
            hashmap_keys, hashmap_vals = None, None
        else:
            hashmap_keys, hashmap_vals = hashmap

        spatial_range = input.spatial_range

        if kmap is None:
            kmap = F.build_kernel_map(
                coords,
                feats.shape[0],
                kernel_size,
                stride,
                padding,
                hashmap_keys,
                hashmap_vals,
                spatial_range,
                kmap_mode,
                dataflow,
                downsample_mode=config.downsample_mode,
                training=training,
                ifsort=config.ifsort,
                split_mask_num=config.split_mask_num,
                split_mask_num_bwd=config.split_mask_num_bwd,
                FOD_fusion=config.FOD_fusion,
                IGEMM_center_only=config.get("IGEMM_center_only", False),
                inference=inference_no_grad,
                subm=subm,
            )

            hashmap = [kmap["hashmap_keys"], kmap["hashmap_vals"]]

            input._caches.kmaps[kmap_key] = kmap
            input._caches.hashmaps[hashmap_stride] = hashmap

        if (
            no_grad_forward is not None
            and inference_no_grad
        ):
            feats = no_grad_forward(feats, weight, kmap, config, transposed)
        else:
            feats = ConvolutionFunction.apply(
                feats,
                weight,
                kmap,
                config,
                transposed,
            )

        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=kmap["coords"],
            feats=feats,
            stride=output_stride,
            spatial_range=kmap["spatial_range"],
        )
    else:
        tensor_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        if not generative:
            kmap = input._caches.kmaps.get(
                _make_kmap_cache_key(
                    tensor_stride,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    False,
                    config,
                    training,
                )
            )

            kmap = F.transpose_kernel_map(
                kmap,
                config.ifsort,
                training=training,
                split_mask_num=config.split_mask_num,
                split_mask_num_bwd=config.split_mask_num_bwd,
            )

            feats = ConvolutionFunction.apply(
                feats,
                weight,
                kmap,
                config,
                transposed,
            )

            if bias is not None:
                feats += bias
            output = SparseTensor(
                coords=input._caches.cmaps[tensor_stride][0],
                feats=feats,
                stride=tensor_stride,
                spatial_range=input._caches.cmaps[tensor_stride][1],
            )
        else:
            hashmap_keys, hashmap_vals = None, None

            spatial_range = input.spatial_range
            kmap = F.build_kernel_map(
                coords,
                feats.shape[0],
                kernel_size,
                stride,
                padding,
                hashmap_keys,
                hashmap_vals,
                spatial_range,
                kmap_mode,
                dataflow,
                downsample_mode=config.downsample_mode,
                training=training,
                ifsort=config.ifsort,
                generative=generative,
                FOD_fusion=config.FOD_fusion,
                IGEMM_center_only=config.get("IGEMM_center_only", False),
                inference=inference_no_grad,
                subm=False,
            )
            # generate output: logically forced to be not transposed
            feats = ConvolutionFunction.apply(
                feats,
                weight,
                kmap,
                config,
                False,
            )
            if bias is not None:
                feats += bias
            input._caches.cmaps[tensor_stride] = (
                kmap["coords"],
                kmap.get("spatial_range"),
            )
            output = SparseTensor(
                coords=input._caches.cmaps[tensor_stride][0],
                feats=feats,
                stride=tensor_stride,
                spatial_range=input._caches.cmaps[tensor_stride][1],
            )
            hashmap = [kmap["hashmap_keys"], kmap["hashmap_vals"]]
            input._caches.kmaps = dict()  # new_kmap
            input._caches.hashmaps = dict()

    output._caches = input._caches
    output._caches.cmaps.setdefault(
        output.stride, (output.coords, output.spatial_range)
    )
    return output



def target_conv3d(
    input: SparseTensor,
    target: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    config: Dict = None,
    training: bool = False,
) -> SparseTensor:
    """Sparse convolution evaluated only at ``target`` coordinates."""

    from torch_lattice.nn import functional as F

    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    padding = make_ntuple(padding, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)
    weight = _kernel_weight(weight, kernel_size)

    if config is None:
        config = F.conv_config.get_global_conv_config()
        if config is None:
            config = F.conv_config.get_default_conv_config(
                conv_mode=F.get_conv_mode(), training=training
            )
    config = config.copy()
    config.dataflow = F.Dataflow.GatherScatter
    config.ifsort = False

    relation = build_target_out_in_map(
        input.coords,
        target.coords,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    kmap = gather_scatter_kmap_from_out_in_map(
        relation,
        input_size=int(input.feats.shape[0]),
    )
    feats = GatherScatterConvolutionFuntion.apply(
        input.feats,
        weight,
        kmap,
        config,
        False,
    )
    if bias is not None:
        feats = feats + bias
    output = SparseTensor(
        coords=target.coords,
        feats=feats,
        stride=target.stride,
        spatial_range=target.spatial_range,
    )
    output._caches = target._caches
    return output


def _kernel_weight(
    weight: torch.Tensor,
    kernel_size: Tuple[int, int, int],
) -> torch.Tensor:
    kernel_volume = int(kernel_size[0] * kernel_size[1] * kernel_size[2])
    if weight.ndim == 2:
        if kernel_volume != 1:
            raise ValueError("2D target_conv3d weight requires kernel_size=1.")
        return weight.reshape(1, weight.shape[0], weight.shape[1]).contiguous()
    if weight.ndim != 3 or int(weight.shape[0]) != kernel_volume:
        raise ValueError(
            f"target_conv3d weight shape {tuple(weight.shape)} does not match kernel_size={kernel_size}."
        )
    return weight.contiguous()
