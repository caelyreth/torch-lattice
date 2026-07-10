from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from torch_lattice import nn as spnn

try:
    from lattice_contract import (
        MLIRModuleBuilder,
        SparseTensorType,
        TensorType,
        WeightType,
        dense_packing,
        quantized_packing,
    )
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError(
        "torch_lattice.artifact requires the MLIR artifact API from "
        "lattice-contract; install a lattice-contract build that exports "
        "MLIRModuleBuilder, TensorType, SparseTensorType, WeightType, and "
        "DIALECT_SCHEMA_DIGEST."
    ) from exc

from .weights import pack_quantized_weight

ValueKind = Literal["sparse_tensor", "dense_tensor"]


@dataclass(frozen=True)
class ModuleLowering:
    fn: Callable[..., ArtifactValue]
    arities: frozenset[int]

    def lower(
        self,
        builder: TorchLatticeArtifactBuilder,
        name: str,
        module: nn.Module,
        inputs: tuple[ArtifactValue, ...],
    ) -> ArtifactValue:
        if len(inputs) not in self.arities:
            expected = " or ".join(str(value) for value in sorted(self.arities))
            raise ValueError(
                f"{type(module).__name__} artifact expects {expected} "
                "symbolic lattice values."
            )
        return self.fn(builder, name, module, *inputs)


_MODULE_LOWERINGS: dict[type[nn.Module], ModuleLowering] = {}


def module_lowering(
    *types: type[nn.Module],
    arity: int | tuple[int, ...] = 1,
) -> Callable[[Callable[..., ArtifactValue]], Callable[..., ArtifactValue]]:
    """Register a Torch module lowering method."""

    def decorator(
        fn: Callable[..., ArtifactValue],
    ) -> Callable[..., ArtifactValue]:
        arities = (arity,) if isinstance(arity, int) else arity
        lowering = ModuleLowering(fn, frozenset(arities))
        for module_type in types:
            if module_type in _MODULE_LOWERINGS:
                raise ValueError(
                    f"duplicate artifact module lowering: {module_type.__name__}"
                )
            _MODULE_LOWERINGS[module_type] = lowering
        return fn

    return decorator


_RMS_NORM_TYPES = (spnn.RMSNorm,) + ((nn.RMSNorm,) if hasattr(nn, "RMSNorm") else ())


@dataclass(frozen=True)
class ArtifactValue:
    """A value in the Torch-to-lattice artifact graph."""

    value: object
    kind: ValueKind
    channels: int | None


class TorchLatticeArtifactBuilder:
    """Explicit builder for Torch-to-lattice MLIR artifacts."""

    def __init__(
        self,
        *,
        input_name: str = "input",
        output_name: str = "output",
        input_dtype: str = "f32",
        batch_size: int | None = None,
        quantize_bits: int | None = None,
        quantize_group_size: int = 32,
        quantize_scale_dtype: str = "f16",
        create_default_input: bool = True,
        input_stride=(1, 1, 1),
    ) -> None:
        if input_dtype not in {"f16", "f32"}:
            raise ValueError("input_dtype must be 'f16' or 'f32'.")
        if batch_size is not None and batch_size < 0:
            raise ValueError("batch_size must be non-negative or None.")
        if quantize_bits is not None and quantize_bits not in (4, 8):
            raise ValueError("quantize_bits must be None, 4, or 8.")
        if quantize_group_size <= 0:
            raise ValueError("quantize_group_size must be positive.")
        if quantize_scale_dtype not in {"f16", "f32"}:
            raise ValueError("quantize_scale_dtype must be 'f16' or 'f32'.")
        self.input_name = input_name
        self.output_name = output_name
        self.input_dtype = input_dtype
        self.batch_size = batch_size
        self.quantize_bits = quantize_bits
        self.quantize_group_size = int(quantize_group_size)
        self.quantize_scale_dtype = quantize_scale_dtype
        self.input_stride = _triple(input_stride)
        self._builder = MLIRModuleBuilder()
        self._weights: dict[str, torch.Tensor] = {}
        self._value = self.sparse_input() if create_default_input else None
        self._finalized = False

    @property
    def current(self) -> ArtifactValue:
        if self._value is None:
            raise ValueError(
                "builder has no current value; pass explicit inputs or create a default sparse input."
            )
        return self._value

    @property
    def weights(self) -> dict[str, torch.Tensor]:
        return dict(self._weights)

    def sparse_input(self) -> ArtifactValue:
        coords = self._builder.argument(
            "coords",
            TensorType("tensor<?x4xi32>"),
            role="sparse_coords",
        )
        features = self._builder.argument(
            "features",
            TensorType(f"tensor<?x?x{self.input_dtype}>"),
            role="sparse_features",
        )
        active = self._builder.argument(
            "active",
            self._active_rows_type(),
            role="sparse_active",
        )
        sparse_type = SparseTensorType(dtype=self.input_dtype)
        value = self._builder.sparse_make(
            coords=coords,
            features=features,
            active=active,
            stride=self.input_stride,
            coord_order="batch_x_y_z",
            result_type=sparse_type,
            result=self.input_name,
        )
        return ArtifactValue(value=value, kind="sparse_tensor", channels=None)

    def sparse_argument(
        self,
        name: str,
        *,
        dtype: str | None = None,
        channels: int | None = None,
        stride=(1, 1, 1),
    ) -> ArtifactValue:
        safe = _safe_value_name(name)
        dtype = dtype or self.input_dtype
        coords = self._builder.argument(
            f"{safe}_coords",
            TensorType("tensor<?x4xi32>"),
            role="sparse_coords",
        )
        features = self._builder.argument(
            f"{safe}_features",
            TensorType(
                f"tensor<?x{int(channels)}x{dtype}>"
                if channels is not None
                else f"tensor<?x?x{dtype}>"
            ),
            role="sparse_features",
        )
        active = self._builder.argument(
            f"{safe}_active",
            self._active_rows_type(),
            role="sparse_active",
        )
        sparse_type = SparseTensorType(dtype=dtype)
        value = self._builder.sparse_make(
            coords=coords,
            features=features,
            active=active,
            stride=_triple(stride),
            coord_order="batch_x_y_z",
            result_type=sparse_type,
            result=safe,
        )
        return ArtifactValue(value=value, kind="sparse_tensor", channels=channels)

    def dense_argument(
        self,
        name: str,
        type: str | TensorType,
        *,
        channels: int | None = None,
    ) -> ArtifactValue:
        value = self._builder.argument(
            name, type if isinstance(type, TensorType) else TensorType(type)
        )
        return ArtifactValue(value=value, kind="dense_tensor", channels=channels)

    def module(self, name: str, module: nn.Module) -> ArtifactValue:
        value = self.lower_module(name, module, self._value)
        self._value = value
        return value

    def lower_module(
        self,
        name: str,
        module: nn.Module,
        *inputs: ArtifactValue,
    ) -> ArtifactValue:
        lowering = next(
            (
                registered
                for module_type in type(module).__mro__
                if (registered := _MODULE_LOWERINGS.get(module_type)) is not None
            ),
            None,
        )
        if lowering is not None:
            return lowering.lower(self, name, module, inputs)
        raise ValueError(
            f"unsupported module for lattice artifact: {type(module).__name__}"
        )

    def _conv_args(
        self,
        name: str,
        module: spnn.Conv3d,
        input: ArtifactValue,
        *,
        relation_order: bool = False,
    ):
        value = self._current_or(input)
        self._require_sparse(value, module)
        sparse_type = SparseTensorType(dtype=self._dtype_for_module(module))
        weight = self._weight(
            name,
            "weight",
            _conv_weight_to_mlx(module, relation_order=relation_order),
            family="conv3d",
            layout="conv3d_o_zyx_i",
        )
        bias = None
        if module.bias is not None:
            bias = self._weight(
                name, "bias", module.bias.detach(), family="bias", layout="bias_c"
            )

        return {
            "input": value.value,
            "weight": weight,
            "bias": bias,
            "kernel_size": _triple(module.kernel_size),
            "result_type": sparse_type,
            "result": _safe_value_name(name),
        }

    @module_lowering(spnn.Conv3d, arity=(1, 2))
    def conv3d(
        self,
        name: str,
        module: spnn.Conv3d,
        input: ArtifactValue,
        coordinates: ArtifactValue | None = None,
    ) -> ArtifactValue:
        relation_order = coordinates is not None or any(
            value != 1 for value in module.dilation
        )
        args = self._conv_args(
            name,
            module,
            input,
            relation_order=relation_order,
        )
        if coordinates is None:
            out = self._builder.conv3d(
                **args,
                stride=_triple(module.stride),
                padding=_triple(module.padding),
                dilation=_triple(module.dilation),
            )
        else:
            self._require_sparse(coordinates, module)
            out = self._builder.target_conv3d(
                **args,
                target=coordinates.value,
                stride=_triple(module.stride),
                padding=_triple(module.padding),
                dilation=_triple(module.dilation),
            )
        return ArtifactValue(out, "sparse_tensor", module.out_channels)

    @module_lowering(spnn.SubmConv3d)
    def subm_conv3d(
        self,
        name: str,
        module: spnn.SubmConv3d,
        input: ArtifactValue,
    ) -> ArtifactValue:
        out = self._builder.subm_conv3d(
            **self._conv_args(
                name,
                module,
                input,
                relation_order=any(value != 1 for value in module.dilation),
            ),
            dilation=_triple(module.dilation),
        )
        return ArtifactValue(out, "sparse_tensor", module.out_channels)

    @module_lowering(spnn.NormalizedSubmConv3d)
    def normalized_subm_conv3d(
        self,
        name: str,
        module: spnn.NormalizedSubmConv3d,
        input: ArtifactValue,
    ) -> ArtifactValue:
        self._require_dense_normalized_weights(module)
        out = self._builder.normalized_subm_conv3d(
            **self._conv_args(
                name,
                module,
                input,
                relation_order=any(value != 1 for value in module.dilation),
            ),
            dilation=_triple(module.dilation),
            eps=module.eps,
        )
        return ArtifactValue(out, "sparse_tensor", module.out_channels)

    @module_lowering(spnn.ConvTranspose3d)
    def conv_transpose3d(
        self,
        name: str,
        module: spnn.ConvTranspose3d,
        input: ArtifactValue,
    ) -> ArtifactValue:
        out = self._builder.conv_transpose3d(
            **self._conv_args(name, module, input),
            stride=_triple(module.stride),
            padding=_triple(module.padding),
            dilation=_triple(module.dilation),
        )
        return ArtifactValue(out, "sparse_tensor", module.out_channels)

    @module_lowering(spnn.NormalizedConvTranspose3d)
    def normalized_conv_transpose3d(
        self,
        name: str,
        module: spnn.NormalizedConvTranspose3d,
        input: ArtifactValue,
    ) -> ArtifactValue:
        self._require_dense_normalized_weights(module)
        out = self._builder.normalized_conv_transpose3d(
            **self._conv_args(name, module, input),
            stride=_triple(module.stride),
            padding=_triple(module.padding),
            dilation=_triple(module.dilation),
            eps=module.eps,
        )
        return ArtifactValue(out, "sparse_tensor", module.out_channels)

    @module_lowering(spnn.GenerativeConvTranspose3d)
    def generative_conv_transpose3d(
        self,
        name: str,
        module: spnn.GenerativeConvTranspose3d,
        input: ArtifactValue,
    ) -> ArtifactValue:
        out = self._builder.generative_conv_transpose3d(
            **self._conv_args(name, module, input),
            stride=_triple(module.stride),
        )
        return ArtifactValue(out, "sparse_tensor", module.out_channels)

    @module_lowering(spnn.NormalizedGenerativeConvTranspose3d)
    def normalized_generative_conv_transpose3d(
        self,
        name: str,
        module: spnn.NormalizedGenerativeConvTranspose3d,
        input: ArtifactValue,
    ) -> ArtifactValue:
        self._require_dense_normalized_weights(module)
        out = self._builder.normalized_generative_conv_transpose3d(
            **self._conv_args(name, module, input),
            stride=_triple(module.stride),
            eps=module.eps,
        )
        return ArtifactValue(out, "sparse_tensor", module.out_channels)

    @module_lowering(spnn.Pool3d)
    def pool3d(
        self,
        name: str,
        module: spnn.Pool3d,
        input: ArtifactValue | None = None,
    ) -> ArtifactValue:
        value = self._current_or(input)
        self._require_sparse(value, module)
        out = self._builder.pool3d(
            input=value.value,
            mode=module.mode,
            kernel_size=_triple(module.kernel_size),
            stride=_triple(module.stride),
            padding=_triple(module.padding),
            dilation=_triple(module.dilation),
            result_type=SparseTensorType(dtype=self.input_dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "sparse_tensor", value.channels)

    @module_lowering(spnn.PoolTranspose3d, arity=(1, 2))
    def pool_transpose3d(
        self,
        name: str,
        module: spnn.PoolTranspose3d,
        input: ArtifactValue,
        target: ArtifactValue | None = None,
    ) -> ArtifactValue:
        self._require_sparse(input, module)
        if target is not None:
            self._require_sparse(target, module)
        out = self._builder.pool_transpose3d(
            input=input.value,
            target=None if target is None else target.value,
            kernel_size=_triple(module.kernel_size),
            stride=_triple(module.stride),
            padding=_triple(module.padding),
            dilation=_triple(module.dilation),
            result_type=SparseTensorType(dtype=self.input_dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "sparse_tensor", input.channels)

    def voxelize(
        self,
        name: str,
        *,
        points: ArtifactValue,
        features: ArtifactValue,
        batch_indices: ArtifactValue,
        active_rows: ArtifactValue,
        voxel_size,
        origin=0.0,
        reduction: Literal["sum", "mean"] = "mean",
        stride=1,
    ) -> ArtifactValue:
        for label, value in {
            "points": points,
            "features": features,
            "batch_indices": batch_indices,
            "active_rows": active_rows,
        }.items():
            self._require_dense_name(value, label)
        out = self._builder.voxelize(
            points=points.value,
            features=features.value,
            batch_indices=batch_indices.value,
            active_rows=active_rows.value,
            voxel_size=_float_triple(voxel_size),
            origin=_float_triple(origin),
            reduction=reduction,
            stride=_triple(stride),
            result_type=SparseTensorType(dtype=self.input_dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "sparse_tensor", features.channels)

    def devoxelize(
        self,
        name: str,
        *,
        points: ArtifactValue,
        voxels: ArtifactValue,
        batch_indices: ArtifactValue,
        point_active_rows: ArtifactValue,
        voxel_size,
        origin=0.0,
        interpolation: Literal["nearest", "linear"] = "nearest",
    ) -> ArtifactValue:
        self._require_dense_name(points, "points")
        self._require_sparse_name(voxels, "voxels")
        self._require_dense_name(batch_indices, "batch_indices")
        self._require_dense_name(point_active_rows, "point_active_rows")
        out = self._builder.devoxelize(
            points=points.value,
            voxels=voxels.value,
            batch_indices=batch_indices.value,
            point_active_rows=point_active_rows.value,
            voxel_size=_float_triple(voxel_size),
            origin=_float_triple(origin),
            interpolation=interpolation,
            result_type=_feature_type(voxels.channels, self.input_dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "dense_tensor", voxels.channels)

    @module_lowering(spnn.BatchNorm)
    def batch_norm(
        self,
        name: str,
        module: spnn.BatchNorm,
        input: ArtifactValue | None = None,
    ) -> ArtifactValue:
        value = self._current_or(input)
        self._require_sparse(value, module)
        sparse, features = self._sparse_features(name, value)
        dtype = self._dtype_for_module(module)
        scale = self._optional_channel_weight(
            name, "weight", module.weight, dtype=dtype
        )
        bias = self._optional_channel_weight(
            name, "bias", module.bias, family="bias", layout="bias_c", dtype=dtype
        )
        mean = self._optional_channel_weight(
            name, "running_mean", module.running_mean, dtype=dtype
        )
        var = self._optional_channel_weight(
            name, "running_var", module.running_var, dtype=dtype
        )
        if scale is None:
            scale = self._constant_channel_weight(
                name, "weight", torch.ones(module.num_features), dtype=dtype
            )
        if bias is None:
            bias = self._constant_channel_weight(
                name,
                "bias",
                torch.zeros(module.num_features),
                family="bias",
                layout="bias_c",
                dtype=dtype,
            )
        if mean is None or var is None:
            raise ValueError(
                "BatchNorm artifact requires frozen running_mean and running_var."
            )
        out_features = self._builder.batch_norm(
            input=features,
            scale=scale,
            bias=bias,
            mean=mean,
            var=var,
            eps=float(module.eps),
            result_type=_feature_type(module.num_features, dtype),
            result=f"{_safe_value_name(name)}_features",
        )
        out = self._builder.sparse_with_features(
            input=sparse,
            features=out_features,
            result_type=SparseTensorType(dtype=dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "sparse_tensor", module.num_features)

    @module_lowering(spnn.LayerNorm, nn.LayerNorm)
    def layer_norm(
        self,
        name: str,
        module: nn.LayerNorm,
        input: ArtifactValue | None = None,
    ) -> ArtifactValue:
        value = self._current_or(input)
        sparse, features, output_kind = self._feature_input(name, value)
        channels = _single_normalized_dim(module.normalized_shape, "LayerNorm")
        dtype = self._dtype_for_module(module)
        scale = self._optional_channel_weight(
            name, "weight", module.weight, dtype=dtype
        )
        bias = self._optional_channel_weight(
            name, "bias", module.bias, family="bias", layout="bias_c", dtype=dtype
        )
        if scale is None:
            scale = self._constant_channel_weight(
                name, "weight", torch.ones(channels), dtype=dtype
            )
        if bias is None:
            bias = self._constant_channel_weight(
                name,
                "bias",
                torch.zeros(channels),
                family="bias",
                layout="bias_c",
                dtype=dtype,
            )
        out_features = self._builder.layer_norm(
            input=features,
            scale=scale,
            bias=bias,
            eps=float(module.eps),
            result_type=_feature_type(channels, dtype),
            result=f"{_safe_value_name(name)}_features",
        )
        return self._feature_output(
            name, sparse, out_features, output_kind, channels, dtype
        )

    @module_lowering(*_RMS_NORM_TYPES)
    def rms_norm(
        self,
        name: str,
        module: nn.Module,
        input: ArtifactValue | None = None,
    ) -> ArtifactValue:
        value = self._current_or(input)
        sparse, features, output_kind = self._feature_input(name, value)
        channels = _single_normalized_dim(module.normalized_shape, "RMSNorm")
        dtype = self._dtype_for_module(module)
        scale = self._optional_channel_weight(
            name, "weight", getattr(module, "weight", None), dtype=dtype
        )
        if scale is None:
            scale = self._constant_channel_weight(
                name, "weight", torch.ones(channels), dtype=dtype
            )
        out_features = self._builder.rms_norm(
            input=features,
            scale=scale,
            eps=float(module.eps),
            result_type=_feature_type(channels, dtype),
            result=f"{_safe_value_name(name)}_features",
        )
        return self._feature_output(
            name, sparse, out_features, output_kind, channels, dtype
        )

    @module_lowering(spnn.ReLU, nn.ReLU)
    def relu(
        self,
        name: str,
        module: nn.Module,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del module
        return self.activation(name, "relu", input=input)

    @module_lowering(spnn.LeakyReLU, nn.LeakyReLU)
    def leaky_relu(
        self,
        name: str,
        module: spnn.LeakyReLU | nn.LeakyReLU,
        input: ArtifactValue,
    ) -> ArtifactValue:
        return self.activation(
            name,
            "leaky_relu",
            input=input,
            alpha=float(module.negative_slope),
        )

    @module_lowering(spnn.SiLU, nn.SiLU)
    def silu(
        self,
        name: str,
        module: nn.Module,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del module
        return self.activation(name, "silu", input=input)

    @module_lowering(spnn.GELU, nn.GELU)
    def gelu(
        self,
        name: str,
        module: spnn.GELU | nn.GELU,
        input: ArtifactValue,
    ) -> ArtifactValue:
        return self.activation(
            name,
            "gelu",
            input=input,
            approximate=str(module.approximate or "none"),
        )

    @module_lowering(spnn.Sigmoid, nn.Sigmoid)
    def sigmoid(
        self,
        name: str,
        module: nn.Module,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del module
        return self.activation(name, "sigmoid", input=input)

    @module_lowering(spnn.Tanh, nn.Tanh)
    def tanh(
        self,
        name: str,
        module: nn.Module,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del module
        return self.activation(name, "tanh", input=input)

    @module_lowering(spnn.Softplus, nn.Softplus)
    def softplus(
        self,
        name: str,
        module: spnn.Softplus | nn.Softplus,
        input: ArtifactValue,
    ) -> ArtifactValue:
        return self.activation(
            name,
            "softplus",
            input=input,
            beta=float(module.beta),
            threshold=float(module.threshold),
        )

    def activation(
        self,
        name: str,
        kind: str,
        *,
        input: ArtifactValue | None = None,
        approximate: str = "none",
        alpha: float = 0.01,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> ArtifactValue:
        value = self._current_or(input)
        if value.kind == "sparse_tensor":
            sparse, features = self._sparse_features(name, value)
            out_features = self._activation_features(
                name,
                features,
                value,
                kind,
                approximate=approximate,
                alpha=alpha,
                beta=beta,
                threshold=threshold,
            )
            out = self._builder.sparse_with_features(
                input=sparse,
                features=out_features,
                result_type=SparseTensorType(dtype=self.input_dtype),
                result=_safe_value_name(name),
            )
            return ArtifactValue(out, "sparse_tensor", value.channels)
        out = self._activation_features(
            name,
            value.value,
            value,
            kind,
            approximate=approximate,
            alpha=alpha,
            beta=beta,
            threshold=threshold,
        )
        return ArtifactValue(out, "dense_tensor", value.channels)

    @module_lowering(spnn.GlobalAvgPool)
    def global_avg_pool(
        self,
        name: str,
        module: spnn.GlobalAvgPool,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del module
        return self.global_pool(name, "avg", input)

    @module_lowering(spnn.GlobalMaxPool)
    def global_max_pool(
        self,
        name: str,
        module: spnn.GlobalMaxPool,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del module
        return self.global_pool(name, "max", input)

    @module_lowering(spnn.GlobalSumPool)
    def global_sum_pool(
        self,
        name: str,
        module: spnn.GlobalSumPool,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del module
        return self.global_pool(name, "sum", input)

    def global_pool(
        self,
        name: str,
        mode: Literal["sum", "avg", "max"],
        input: ArtifactValue | None = None,
    ) -> ArtifactValue:
        value = self._current_or(input)
        self._require_sparse_name(value, "GlobalPool")
        if self.batch_size is None:
            raise ValueError("global_pool artifact requires a static batch_size.")
        channels = value.channels
        out = self._builder.global_pool(
            input=value.value,
            mode=mode,
            batch_size=int(self.batch_size),
            result_type=_feature_type(channels, self.input_dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "dense_tensor", channels)

    @module_lowering(nn.Linear)
    def linear(
        self,
        name: str,
        module: nn.Linear,
        input: ArtifactValue | None = None,
    ) -> ArtifactValue:
        value = self._current_or(input)
        self._require_dense(value, module)
        dtype = _torch_dtype_name(module.weight.dtype)
        weight = self._weight(
            name, "weight", module.weight.detach(), family="linear", layout="linear_o_i"
        )
        bias = None
        if module.bias is not None:
            bias = self._weight(
                name, "bias", module.bias.detach(), family="bias", layout="bias_c"
            )
        out = self._builder.linear(
            input=value.value,
            weight=weight,
            bias=bias,
            result_type=_feature_type(module.out_features, dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "dense_tensor", module.out_features)

    @module_lowering(nn.Dropout)
    def dropout(
        self,
        name: str,
        module: nn.Dropout,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del name
        if module.training:
            raise ValueError("Dropout artifact is only supported in eval mode.")
        return input

    @module_lowering(nn.Flatten)
    def flatten(
        self,
        name: str,
        module: nn.Flatten,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del name
        if (
            input.kind != "dense_tensor"
            or module.start_dim != 1
            or module.end_dim != -1
        ):
            raise ValueError(
                "Flatten artifact only supports dense classifier heads with "
                "start_dim=1 and end_dim=-1."
            )
        return input

    @module_lowering(nn.Identity)
    def identity(
        self,
        name: str,
        module: nn.Identity,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del name, module
        return input

    @module_lowering(spnn.InstanceNorm)
    def instance_norm(
        self,
        name: str,
        module: spnn.InstanceNorm,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del self, name, module, input
        raise ValueError(
            "InstanceNorm artifact is not supported without a dedicated "
            "lattice normalization op."
        )

    @module_lowering(spnn.GroupNorm)
    def group_norm(
        self,
        name: str,
        module: spnn.GroupNorm,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del self, name, module, input
        raise ValueError(
            "GroupNorm artifact is not supported without a dedicated lattice "
            "normalization op."
        )

    @module_lowering(spnn.SparseCrop)
    def sparse_crop(
        self,
        name: str,
        module: spnn.SparseCrop,
        input: ArtifactValue,
    ) -> ArtifactValue:
        del self, name, module, input
        raise ValueError(
            "SparseCrop artifact is not supported until the lattice dialect "
            "defines a sparse crop op."
        )

    def sparse_binary(
        self,
        name: str,
        lhs: ArtifactValue,
        rhs: ArtifactValue,
        op: str,
        *,
        join: str = "outer",
        lhs_fill: float = 0.0,
        rhs_fill: float = 0.0,
    ) -> ArtifactValue:
        self._require_sparse_name(lhs, f"sparse {op} lhs")
        self._require_sparse_name(rhs, f"sparse {op} rhs")
        if (
            lhs.channels is not None
            and rhs.channels is not None
            and lhs.channels != rhs.channels
        ):
            raise ValueError(f"sparse {op} requires matching channel counts.")
        out = self._builder.sparse_binary(
            lhs=lhs.value,
            rhs=rhs.value,
            op=op,
            join=join,
            lhs_fill=float(lhs_fill),
            rhs_fill=float(rhs_fill),
            result_type=SparseTensorType(dtype=self.input_dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "sparse_tensor", lhs.channels or rhs.channels)

    def sparse_add(
        self,
        name: str,
        lhs: ArtifactValue,
        rhs: ArtifactValue,
        *,
        join: str = "outer",
        lhs_fill: float = 0.0,
        rhs_fill: float = 0.0,
    ) -> ArtifactValue:
        return self.sparse_binary(
            name, lhs, rhs, "add", join=join, lhs_fill=lhs_fill, rhs_fill=rhs_fill
        )

    def sparse_cat(
        self, name: str, lhs: ArtifactValue, rhs: ArtifactValue, *, join: str = "inner"
    ) -> ArtifactValue:
        self._require_sparse_name(lhs, "sparse cat lhs")
        self._require_sparse_name(rhs, "sparse cat rhs")
        channels = None
        if lhs.channels is not None and rhs.channels is not None:
            channels = lhs.channels + rhs.channels
        out = self._builder.sparse_cat(
            lhs=lhs.value,
            rhs=rhs.value,
            join=join,
            result_type=SparseTensorType(dtype=self.input_dtype),
            result=_safe_value_name(name),
        )
        return ArtifactValue(out, "sparse_tensor", channels)

    def output(
        self,
        *values: ArtifactValue,
        names: tuple[str, ...] | None = None,
    ) -> None:
        values = values or (self.current,)
        if names is not None and len(names) != len(values):
            raise ValueError("output names must match the number of artifact outputs")
        if names is None:
            names = (
                (self.output_name,)
                if len(values) == 1
                else tuple(
                    f"{self.output_name}_{index}" for index in range(len(values))
                )
            )
        roles = tuple(
            "sparse_tensor" if value.kind == "sparse_tensor" else "tensor"
            for value in values
        )
        self._builder.return_(
            *(value.value for value in values), names=names, roles=roles
        )
        self._finalized = True

    def to_mlir(self) -> str:
        if not self._finalized:
            self.output()
        return self._builder.to_mlir()

    def save(
        self, artifact_dir: str | Path, *, clean: bool = True, validate: bool = True
    ):
        from .io import _save_artifact

        return _save_artifact(
            artifact_dir,
            self.to_mlir(),
            self._weights,
            clean=clean,
            validate=validate,
        )

    def _current_or(self, value: ArtifactValue | None) -> ArtifactValue:
        if value is not None:
            return value
        return self.current

    def _active_rows_type(self) -> TensorType:
        return TensorType("tensor<1xi32>")

    def _feature_input(self, name: str, value: ArtifactValue):
        if value.kind == "sparse_tensor":
            sparse, features = self._sparse_features(name, value)
            return sparse, features, "sparse_tensor"
        self._require_dense_name(value, name)
        return None, value.value, "dense_tensor"

    def _feature_output(
        self, name: str, sparse, features, kind: str, channels: int | None, dtype: str
    ) -> ArtifactValue:
        if kind == "sparse_tensor":
            out = self._builder.sparse_with_features(
                input=sparse,
                features=features,
                result_type=SparseTensorType(dtype=dtype),
                result=_safe_value_name(name),
            )
            return ArtifactValue(out, "sparse_tensor", channels)
        return ArtifactValue(features, "dense_tensor", channels)

    def _sparse_features(self, name: str, value: ArtifactValue):
        self._require_sparse_name(value, name)
        channels = value.channels
        coords, features, active = self._builder.sparse_decompose(
            input=value.value,
            result_types=(
                TensorType("tensor<?x4xi32>"),
                _feature_type(channels, self.input_dtype),
                TensorType("tensor<1xi32>"),
            ),
            result_1=f"{_safe_value_name(name)}_features_in",
            result_2=f"{_safe_value_name(name)}_active",
        )
        del coords, active
        return value.value, features

    def _activation_features(
        self,
        name: str,
        features,
        value: ArtifactValue,
        kind: str,
        *,
        approximate: str,
        alpha: float,
        beta: float,
        threshold: float,
    ):
        return self._builder.activation(
            input=features,
            kind=kind,
            approximate=approximate,
            alpha=float(alpha),
            beta=float(beta),
            threshold=float(threshold),
            result_type=_feature_type(value.channels, self.input_dtype),
            result=f"{_safe_value_name(name)}_features",
        )

    def _weight(
        self,
        module_name: str,
        parameter_name: str,
        tensor: torch.Tensor,
        *,
        family: str,
        layout: str,
    ):
        key = f"{_safe_key(module_name)}.{parameter_name}"
        if key in self._weights or f"{key}.weight" in self._weights:
            raise ValueError(f"duplicate exported weight key: {key}")
        tensor = tensor.detach().cpu().contiguous()
        packing = dense_packing()
        stored_dtype = _torch_dtype_name(tensor.dtype)
        if self.quantize_bits is not None and family in {"conv3d", "linear"}:
            packed = pack_quantized_weight(
                tensor,
                bits=self.quantize_bits,
                group_size=self.quantize_group_size,
                scale_dtype=self.quantize_scale_dtype,
            )
            self._weights[f"{key}.weight"] = packed.weight
            self._weights[f"{key}.scales"] = packed.scales
            self._weights[f"{key}.biases"] = packed.biases
            packing = quantized_packing(
                f"int{self.quantize_bits}",
                group_size=self.quantize_group_size,
                scale_dtype=self.quantize_scale_dtype,
            )
        else:
            self._weights[key] = tensor
        return self._builder.weight(
            sym_name=_safe_value_name(key),
            storage_key=key,
            layout=layout,
            packing=packing,
            result_type=WeightType(family, stored_dtype),
            result=_safe_value_name(key),
        )

    def _optional_channel_weight(
        self,
        module_name: str,
        parameter_name: str,
        tensor: torch.Tensor | None,
        *,
        family: str = "channel",
        layout: str = "channel_c",
        dtype: str,
    ):
        del dtype
        if tensor is None:
            return None
        return self._weight(
            module_name, parameter_name, tensor, family=family, layout=layout
        )

    def _constant_channel_weight(
        self,
        module_name: str,
        parameter_name: str,
        tensor: torch.Tensor,
        *,
        family: str = "channel",
        layout: str = "channel_c",
        dtype: str,
    ):
        torch_dtype = torch.float16 if dtype == "f16" else torch.float32
        return self._weight(
            module_name,
            parameter_name,
            tensor.to(dtype=torch_dtype),
            family=family,
            layout=layout,
        )

    def _require_sparse(self, value: ArtifactValue, module: nn.Module) -> None:
        self._require_sparse_name(value, type(module).__name__)

    def _require_dense_normalized_weights(self, module: nn.Module) -> None:
        if self.quantize_bits is not None:
            raise ValueError(
                f"{type(module).__name__} artifact export does not support "
                "packed weights because normalization depends on weight squares."
            )

    def _require_sparse_name(self, value: ArtifactValue, name: str) -> None:
        if value.kind != "sparse_tensor":
            raise ValueError(f"{name} expects sparse_tensor, got {value.kind}.")

    def _require_dense(self, value: ArtifactValue, module: nn.Module) -> None:
        self._require_dense_name(value, type(module).__name__)

    def _require_dense_name(self, value: ArtifactValue, name: str) -> None:
        if value.kind != "dense_tensor":
            raise ValueError(f"{name} expects dense_tensor, got {value.kind}.")

    def _dtype_for_module(self, module: nn.Module) -> str:
        for parameter in module.parameters(recurse=False):
            return _torch_dtype_name(parameter.dtype)
        return self.input_dtype


SUPPORTED_MODULE_TYPES = tuple(_MODULE_LOWERINGS)


def _single_normalized_dim(value, name: str) -> int:
    if isinstance(value, int):
        return int(value)
    dims = tuple(int(item) for item in value)
    if len(dims) != 1:
        raise ValueError(f"{name} artifact expects one normalized feature dimension.")
    return dims[0]


def _conv_weight_to_mlx(
    module: spnn.Conv3d,
    *,
    relation_order: bool,
) -> torch.Tensor:
    kernel_size = _triple(module.kernel_size)
    weight = module.kernel.detach()
    if weight.ndim == 2:
        weight = weight.reshape(1, weight.shape[0], weight.shape[1])
    expected_kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if weight.ndim != 3 or weight.shape[0] != expected_kernel_volume:
        raise ValueError(
            f"Conv3d kernel shape {tuple(weight.shape)} does not match kernel_size={kernel_size}."
        )
    if relation_order or expected_kernel_volume % 2 == 0:
        spatial_weight = weight.reshape(
            *kernel_size,
            module.in_channels,
            module.out_channels,
        )
    else:
        # Legacy TorchSparse CUDA maps enumerate odd kernels with x fastest.
        # The lattice relation contract enumerates z fastest.
        spatial_weight = weight.reshape(
            kernel_size[2],
            kernel_size[1],
            kernel_size[0],
            module.in_channels,
            module.out_channels,
        ).permute(2, 1, 0, 3, 4)
    return spatial_weight.permute(4, 0, 1, 2, 3).contiguous()


def _feature_type(channels: int | None, dtype: str) -> TensorType:
    if channels is None:
        return TensorType(f"tensor<?x?x{dtype}>")
    return TensorType(f"tensor<?x{int(channels)}x{dtype}>")


def _triple(value) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(item) for item in value)
    raise ValueError(f"expected an int or length-3 tuple, got {value!r}.")


def _float_triple(value) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(float(item) for item in value)
    raise ValueError(f"expected a scalar or length-3 tuple, got {value!r}.")


def _torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.float32:
        return "f32"
    raise ValueError(f"unsupported tensor dtype for lattice artifact: {dtype}")


def _safe_key(value: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_.]+", "_", value).strip("._")
    return out or "layer"


def _safe_value_name(value: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    if not out:
        return "value"
    if out[0].isdigit():
        return f"v_{out}"
    return out
