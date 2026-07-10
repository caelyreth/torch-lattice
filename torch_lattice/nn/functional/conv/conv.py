from __future__ import annotations

from collections.abc import Sequence

import torch

from torch_lattice import SparseTensor
from torch_lattice.core import RelationKey
from torch_lattice.utils import make_ntuple

from ..relation import (
    build_pool_output_coords,
    build_target_out_in_map,
    build_target_transposed_out_in_map,
    gather_scatter_kmap_from_out_in_map,
)
from .func.fetch_on_demand import (
    FetchOnDemandConvolutionFuntion,
    fetch_on_demand_forward_no_grad,
)
from .func.gather_scatter import (
    GatherScatterConvolutionFuntion,
    gather_scatter_forward_no_grad,
)
from .func.implicit_gemm import (
    ImplicitGEMMConvolutionFuntion,
    implicit_gemm_forward_no_grad,
)

__all__ = ["conv3d", "normalized_conv3d"]

Triple = tuple[int, int, int]


def conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: int | Sequence[int],
    bias: torch.Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    config=None,
    subm: bool = False,
    transposed: bool = False,
    generative: bool = False,
    training: bool = False,
    coordinates: SparseTensor | None = None,
) -> SparseTensor:
    """Apply sparse convolution with generated or explicit target support."""

    size = _triple(kernel_size)
    step = _triple(stride)
    pad = _triple(padding)
    spacing = _triple(dilation)
    _validate_convolution_modes(
        subm=subm,
        transposed=transposed,
        generative=generative,
        stride=step,
        coordinates=coordinates,
    )
    resolved = _resolved_config(config, training=training)

    if coordinates is not None and transposed:
        return _target_transposed_convolution(
            input,
            coordinates,
            weight,
            bias,
            kernel_size=size,
            stride=step,
            padding=pad,
            dilation=spacing,
            config=resolved,
            training=training,
        )

    if coordinates is not None:
        return _target_convolution(
            input,
            coordinates,
            weight,
            bias,
            kernel_size=size,
            stride=step,
            padding=pad,
            dilation=spacing,
            config=resolved,
            training=training,
        )

    if _is_pointwise(size, step, pad, spacing):
        feats = input.feats.matmul(weight)
        if bias is not None:
            feats = feats + bias
        return input.replace(feats=feats)

    if generative and (spacing != (1, 1, 1) or pad != (0, 0, 0)):
        raise ValueError(
            "generative transposed convolution requires dilation=1 and padding=0"
        )

    if spacing != (1, 1, 1):
        return _relation_convolution(
            input,
            weight,
            bias,
            kernel_size=size,
            stride=step,
            padding=pad,
            dilation=spacing,
            config=resolved,
            subm=subm,
            transposed=transposed,
            training=training,
        )

    convolution, no_grad_forward, resolved = _dispatch(resolved)
    execution = _execution_key(resolved, training=training)
    inference_no_grad = not torch.is_grad_enabled() or (
        not input.feats.requires_grad and not weight.requires_grad
    )

    if not transposed:
        return _forward_native_convolution(
            input,
            weight,
            bias,
            kernel_size=size,
            stride=step,
            padding=pad,
            dilation=spacing,
            config=resolved,
            execution=execution,
            subm=subm,
            training=training,
            inference_no_grad=inference_no_grad,
            convolution=convolution,
            no_grad_forward=no_grad_forward,
        )
    if generative:
        return _generative_transposed_convolution(
            input,
            weight,
            bias,
            kernel_size=size,
            stride=step,
            padding=pad,
            config=resolved,
            training=training,
            inference_no_grad=inference_no_grad,
            convolution=convolution,
        )
    return _inverse_convolution(
        input,
        weight,
        bias,
        kernel_size=size,
        stride=step,
        padding=pad,
        dilation=spacing,
        config=resolved,
        execution=execution,
        training=training,
        convolution=convolution,
    )


def normalized_conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: int | Sequence[int],
    bias: torch.Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    config=None,
    subm: bool = False,
    transposed: bool = False,
    generative: bool = False,
    training: bool = False,
    coordinates: SparseTensor | None = None,
    eps: float = 1e-8,
) -> SparseTensor:
    """Apply weight-normalized sparse convolution.

    Non-pointwise kernels compute ``conv(input, weight)`` and divide by
    ``sqrt(conv(ones, weight.square()) + eps)`` before applying bias. Both
    passes use the same coordinate manager and therefore reuse cached kernel
    relations. Pointwise kernels intentionally use ordinary matrix
    multiplication, matching the source normalized-convolution contract.
    """
    size = _triple(kernel_size)
    step = _triple(stride)
    pad = _triple(padding)
    spacing = _triple(dilation)
    if eps <= 0:
        raise ValueError("eps must be positive")
    kwargs = {
        "kernel_size": size,
        "stride": step,
        "padding": pad,
        "dilation": spacing,
        "config": config,
        "subm": subm,
        "transposed": transposed,
        "generative": generative,
        "training": training,
        "coordinates": coordinates,
    }
    if _is_pointwise(size, step, pad, spacing):
        return conv3d(input, weight, bias=bias, **kwargs)

    numerator = conv3d(input, weight, **kwargs)
    unit = input.replace(feats=torch.ones_like(input.feats))
    denominator = conv3d(unit, weight.square(), **kwargs)
    if numerator.coords.shape != denominator.coords.shape or not torch.equal(
        numerator.coords, denominator.coords
    ):
        raise RuntimeError("normalized convolution passes produced different support")
    features = numerator.feats / torch.sqrt(denominator.feats + eps)
    if bias is not None:
        features = features + bias
    return numerator.replace(feats=features)


def _forward_native_convolution(
    input: SparseTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
    config,
    execution: tuple,
    subm: bool,
    training: bool,
    inference_no_grad: bool,
    convolution,
    no_grad_forward,
) -> SparseTensor:
    from torch_lattice.nn import functional as F

    cached = input.coord_manager.forward_relation(
        input.coord_key,
        operation="subm_conv3d" if subm else "conv3d",
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        execution=execution,
    )
    if cached is None:
        hashmap = input.coord_manager.hashmap(input.coord_key, execution)
        hashmap_keys, hashmap_vals = hashmap if hashmap is not None else (None, None)
        kmap = F.build_kernel_map(
            input.coords,
            input.feats.shape[0],
            kernel_size,
            stride,
            padding,
            hashmap_keys,
            hashmap_vals,
            input.spatial_range,
            config.kmap_mode,
            config.dataflow,
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
        output_stride = tuple(input.stride[index] * stride[index] for index in range(3))
        if subm:
            target_key = input.coord_key
        else:
            target_key = input.coord_manager.insert(
                kmap["coords"],
                output_stride,
                spatial_range=kmap["spatial_range"],
                batch_counts=_counts_from_coords(kmap["coords"], kmap["spatial_range"]),
            )
        relation_key = RelationKey(
            input.coord_key,
            target_key,
            "subm_conv3d" if subm else "conv3d",
            kernel_size,
            stride,
            padding,
            dilation,
            execution,
        )
        input.coord_manager.set_forward_relation(relation_key, kmap)
        if not subm:
            input.coord_manager.set_inverse_relation(relation_key, kmap)
        input.coord_manager.set_hashmap(
            input.coord_key,
            execution,
            (kmap["hashmap_keys"], kmap["hashmap_vals"]),
        )
    else:
        target_key, kmap = cached

    if no_grad_forward is not None and inference_no_grad:
        feats = no_grad_forward(input.feats, weight, kmap, config, False)
    else:
        feats = convolution.apply(input.feats, weight, kmap, config, False)
    if bias is not None:
        feats = feats + bias
    coordinate_map = input.coord_manager.get(target_key)
    return SparseTensor(
        feats,
        coordinate_map.coords,
        target_key.stride,
        coordinate_map.spatial_range,
        batch_counts=coordinate_map.batch_counts,
        coord_manager=input.coord_manager,
        coord_key=target_key,
    )


def _inverse_convolution(
    input: SparseTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
    config,
    execution: tuple,
    training: bool,
    convolution,
) -> SparseTensor:
    from torch_lattice.nn import functional as F

    inverse = input.coord_manager.inverse_relation(
        input.coord_key,
        operation="conv3d",
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        execution=execution,
    )
    if inverse is None:
        raise ValueError(
            "ConvTranspose3d requires a matching earlier Conv3d relation in "
            "the same coordinate manager"
        )
    target_key, forward_kmap = inverse
    kmap = F.transpose_kernel_map(
        forward_kmap,
        config.ifsort,
        training=training,
        split_mask_num=config.split_mask_num,
        split_mask_num_bwd=config.split_mask_num_bwd,
    )
    feats = convolution.apply(input.feats, weight, kmap, config, True)
    if bias is not None:
        feats = feats + bias
    coordinate_map = input.coord_manager.get(target_key)
    return SparseTensor(
        feats,
        coordinate_map.coords,
        target_key.stride,
        coordinate_map.spatial_range,
        batch_counts=coordinate_map.batch_counts,
        coord_manager=input.coord_manager,
        coord_key=target_key,
    )


def _generative_transposed_convolution(
    input: SparseTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    config,
    training: bool,
    inference_no_grad: bool,
    convolution,
) -> SparseTensor:
    from torch_lattice.nn import functional as F

    target_stride = tuple(input.stride[index] // stride[index] for index in range(3))
    if any(input.stride[index] % stride[index] for index in range(3)):
        raise ValueError("transposed stride must divide the input sparse stride")
    kmap = F.build_kernel_map(
        input.coords,
        input.feats.shape[0],
        kernel_size,
        stride,
        padding,
        None,
        None,
        input.spatial_range,
        config.kmap_mode,
        config.dataflow,
        downsample_mode=config.downsample_mode,
        training=training,
        ifsort=config.ifsort,
        generative=True,
        FOD_fusion=config.FOD_fusion,
        IGEMM_center_only=config.get("IGEMM_center_only", False),
        inference=inference_no_grad,
        subm=False,
    )
    feats = convolution.apply(input.feats, weight, kmap, config, False)
    if bias is not None:
        feats = feats + bias
    return input.with_coordinates(
        feats=feats,
        coords=kmap["coords"],
        stride=target_stride,
        spatial_range=kmap.get("spatial_range"),
    )


def _relation_convolution(
    input: SparseTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
    config,
    subm: bool,
    transposed: bool,
    training: bool,
) -> SparseTensor:
    if transposed:
        raise ValueError(
            "dilated ConvTranspose3d requires an optimized inverse relation "
            "implementation and is not currently supported"
        )
    target_coords = (
        input.coords
        if subm
        else build_pool_output_coords(
            input.coords,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            spatial_range=input.spatial_range,
        )
    )
    output_stride = tuple(input.stride[index] * stride[index] for index in range(3))
    if subm:
        target = input
    else:
        target = input.with_coordinates(
            feats=input.feats.new_empty((target_coords.shape[0], input.feats.shape[1])),
            coords=target_coords,
            stride=output_stride,
            spatial_range=_output_spatial_range(
                input.spatial_range,
                kernel_size,
                stride,
                padding,
                dilation,
            ),
            batch_counts=_counts_from_coords(
                target_coords,
                _output_spatial_range(
                    input.spatial_range,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                ),
            ),
        )
    return _target_convolution(
        input,
        target,
        weight,
        bias,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        config=config,
        training=training,
    )


def _target_convolution(
    input: SparseTensor,
    target: SparseTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
    config,
    training: bool,
) -> SparseTensor:
    from torch_lattice.nn import functional as F

    expected_stride = tuple(input.stride[index] * stride[index] for index in range(3))
    if target.stride != expected_stride:
        raise ValueError(
            f"target stride {target.stride} does not match convolution output "
            f"stride {expected_stride}"
        )
    weight = _kernel_weight(weight, kernel_size)
    target_config = config.copy()
    target_config.dataflow = F.Dataflow.GatherScatter
    target_config.ifsort = False
    execution = _execution_key(target_config, training=training)
    shared_manager = input.coord_manager is target.coord_manager
    relation_key = (
        RelationKey(
            input.coord_key,
            target.coord_key,
            "target_conv3d",
            kernel_size,
            stride,
            padding,
            dilation,
            execution,
        )
        if shared_manager
        else None
    )
    kmap = (
        input.coord_manager.relation(relation_key) if relation_key is not None else None
    )
    if kmap is None:
        relation = build_target_out_in_map(
            input.coords,
            target.coords,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        kmap = gather_scatter_kmap_from_out_in_map(
            relation, input_size=int(input.feats.shape[0])
        )
        if relation_key is not None:
            input.coord_manager.set_relation(relation_key, kmap)
    feats = GatherScatterConvolutionFuntion.apply(
        input.feats, weight, kmap, target_config, False
    )
    if bias is not None:
        feats = feats + bias
    return target.replace(feats=feats)


def _target_transposed_convolution(
    input: SparseTensor,
    target: SparseTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
    config,
    training: bool,
) -> SparseTensor:
    from torch_lattice.nn import functional as F

    expected_stride = tuple(input.stride[index] // stride[index] for index in range(3))
    if any(input.stride[index] % stride[index] for index in range(3)):
        raise ValueError("transposed stride must divide the input sparse stride")
    if target.stride != expected_stride:
        raise ValueError(
            f"target stride {target.stride} does not match transpose output "
            f"stride {expected_stride}"
        )
    weight = _kernel_weight(weight, kernel_size)
    target_config = config.copy()
    target_config.dataflow = F.Dataflow.GatherScatter
    target_config.ifsort = False
    execution = _execution_key(target_config, training=training)
    shared_manager = input.coord_manager is target.coord_manager
    relation_key = (
        RelationKey(
            input.coord_key,
            target.coord_key,
            "target_conv_transpose3d",
            kernel_size,
            stride,
            padding,
            dilation,
            execution,
        )
        if shared_manager
        else None
    )
    kmap = (
        input.coord_manager.relation(relation_key) if relation_key is not None else None
    )
    if kmap is None:
        relation = build_target_transposed_out_in_map(
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
        if relation_key is not None:
            input.coord_manager.set_relation(relation_key, kmap)
    feats = GatherScatterConvolutionFuntion.apply(
        input.feats,
        weight,
        kmap,
        target_config,
        False,
    )
    if bias is not None:
        feats = feats + bias
    return target.replace(feats=feats)


def _dispatch(config):
    from torch_lattice.nn import functional as F

    local = config.copy()
    if local.dataflow == F.Dataflow.ImplicitGEMM:
        return ImplicitGEMMConvolutionFuntion, implicit_gemm_forward_no_grad, local
    if local.dataflow == F.Dataflow.GatherScatter:
        local.ifsort = False
        return GatherScatterConvolutionFuntion, gather_scatter_forward_no_grad, local
    if local.dataflow == F.Dataflow.FetchOnDemand:
        local.ifsort = False
        return FetchOnDemandConvolutionFuntion, fetch_on_demand_forward_no_grad, local
    raise ValueError(f"unsupported convolution dataflow: {local.dataflow}")


def _resolved_config(config, *, training: bool):
    from torch_lattice.nn import functional as F

    resolved = config or F.conv_config.get_global_conv_config()
    if resolved is None:
        resolved = F.conv_config.get_default_conv_config(
            conv_mode=F.get_conv_mode(), training=training
        )
    return resolved.copy()


def _execution_key(config, *, training: bool) -> tuple:
    return (
        config.kmap_mode,
        config.downsample_mode,
        config.dataflow,
        bool(config.ifsort),
        bool(config.FOD_fusion),
        bool(config.get("IGEMM_center_only", False)),
        int(config.split_mask_num),
        int(config.split_mask_num_bwd) if training else 0,
        config.get("wgrad_split_k", "auto") if training else 0,
        bool(training),
    )


def _validate_convolution_modes(
    *,
    subm: bool,
    transposed: bool,
    generative: bool,
    stride: Triple,
    coordinates: SparseTensor | None,
) -> None:
    if subm and (transposed or generative or coordinates is not None):
        raise ValueError("submanifold convolution only supports implicit input support")
    if subm and stride != (1, 1, 1):
        raise ValueError("submanifold convolution requires stride=1")
    if generative and not transposed:
        raise ValueError("generative convolution must be transposed")
    if coordinates is not None and subm:
        raise ValueError("submanifold convolution cannot consume target support")


def _kernel_weight(weight: torch.Tensor, kernel_size: Triple) -> torch.Tensor:
    kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if weight.ndim == 2:
        if kernel_volume != 1:
            raise ValueError("2D convolution weight requires kernel_size=1")
        return weight.reshape(1, weight.shape[0], weight.shape[1]).contiguous()
    if weight.ndim != 3 or int(weight.shape[0]) != kernel_volume:
        raise ValueError(
            f"convolution weight shape {tuple(weight.shape)} does not match "
            f"kernel_size={kernel_size}"
        )
    return weight.contiguous()


def _counts_from_coords(coords: torch.Tensor, spatial_range) -> tuple[int, ...] | None:
    if spatial_range is None:
        return None
    batch_size = int(spatial_range[0])
    if coords.shape[0] == 0:
        return (0,) * batch_size
    return tuple(
        int(value)
        for value in torch.bincount(coords[:, 0].to(torch.long), minlength=batch_size)
        .cpu()
        .tolist()
    )


def _output_spatial_range(
    spatial_range,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
):
    if spatial_range is None:
        return None
    return tuple(spatial_range[:1]) + tuple(
        max(
            0,
            (
                int(spatial_range[index + 1])
                + 2 * padding[index]
                - dilation[index] * (kernel_size[index] - 1)
                - 1
            )
            // stride[index]
            + 1,
        )
        for index in range(3)
    )


def _is_pointwise(
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> bool:
    return (
        kernel_size == (1, 1, 1)
        and stride == (1, 1, 1)
        and padding == (0, 0, 0)
        and dilation == (1, 1, 1)
    )


def _triple(value) -> Triple:
    return tuple(int(item) for item in make_ntuple(value, ndim=3))
