from __future__ import annotations

import re
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
        "torch_lattice.export requires the MLIR artifact API from "
        "lattice-contract; install a lattice-contract build that exports "
        "MLIRModuleBuilder, TensorType, SparseTensorType, WeightType, and "
        "DIALECT_SCHEMA_DIGEST."
    ) from exc

ValueKind = Literal["sparse_tensor", "dense_tensor"]


@dataclass(frozen=True)
class ExportValue:
    """A value in the Torch-to-lattice export graph."""

    value: object
    kind: ValueKind
    channels: int | None


class TorchLatticeExportBuilder:
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
        if quantize_bits is not None and quantize_bits not in (4, 8):
            raise ValueError("quantize_bits must be None, 4, or 8.")
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
    def current(self) -> ExportValue:
        if self._value is None:
            raise ValueError("builder has no current value; pass explicit inputs or create a default sparse input.")
        return self._value

    @property
    def weights(self) -> dict[str, torch.Tensor]:
        return dict(self._weights)

    def sparse_input(self) -> ExportValue:
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
            TensorType("tensor<1xi32>"),
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
        return ExportValue(value=value, kind="sparse_tensor", channels=None)

    def sparse_argument(
        self,
        name: str,
        *,
        dtype: str | None = None,
        channels: int | None = None,
        stride=(1, 1, 1),
    ) -> ExportValue:
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
            TensorType("tensor<1xi32>"),
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
        return ExportValue(value=value, kind="sparse_tensor", channels=channels)

    def dense_argument(
        self,
        name: str,
        type: str | TensorType,
        *,
        channels: int | None = None,
    ) -> ExportValue:
        value = self._builder.argument(name, type if isinstance(type, TensorType) else TensorType(type))
        return ExportValue(value=value, kind="dense_tensor", channels=channels)

    def module(self, name: str, module: nn.Module) -> ExportValue:
        value = self.lower_module(name, module, self._value)
        self._value = value
        return value

    def lower_module(
        self,
        name: str,
        module: nn.Module,
        *inputs: ExportValue,
    ) -> ExportValue:
        if isinstance(module, spnn.TargetConv3d):
            if len(inputs) != 2:
                raise ValueError("TargetConv3d export expects input and target sparse tensors.")
            return self.target_conv3d(name, module, inputs[0], inputs[1])
        if len(inputs) != 1:
            raise ValueError(f"module {name} must consume exactly one symbolic lattice value.")
        input = inputs[0]
        if isinstance(
            module,
            (
                spnn.Conv3d,
                spnn.SubmConv3d,
                spnn.ConvTranspose3d,
                spnn.GenerativeConvTranspose3d,
            ),
        ):
            return self.conv3d(name, module, input)
        if isinstance(module, spnn.BatchNorm):
            return self.batch_norm(name, module, input)
        if isinstance(module, (spnn.LayerNorm, nn.LayerNorm)):
            return self.layer_norm(name, module, input)
        if _is_rms_norm(module):
            return self.rms_norm(name, module, input)
        if isinstance(module, spnn.InstanceNorm):
            raise ValueError("InstanceNorm export is not supported without a dedicated lattice normalization op.")
        if isinstance(module, spnn.GroupNorm):
            raise ValueError("GroupNorm export is not supported without a dedicated lattice normalization op.")
        activation = _activation_spec(module)
        if activation is not None:
            return self.activation(name, input=input, **activation)
        if isinstance(module, spnn.Pool3d):
            return self.pool3d(name, module, input)
        if isinstance(module, spnn.SparseCrop):
            raise ValueError("SparseCrop export is not supported until the lattice dialect defines a sparse crop op.")
        if isinstance(module, spnn.GlobalAvgPool):
            return self.global_pool(name, "avg", input)
        if isinstance(module, spnn.GlobalMaxPool):
            return self.global_pool(name, "max", input)
        if isinstance(module, nn.Linear):
            return self.linear(name, module, input)
        if isinstance(module, nn.Dropout):
            if module.training:
                raise ValueError("Dropout export is only supported in eval mode.")
            return input
        if isinstance(module, nn.Flatten):
            if input.kind != "dense_tensor" or module.start_dim != 1 or module.end_dim != -1:
                raise ValueError("Flatten export only supports dense classifier heads with start_dim=1 and end_dim=-1.")
            return input
        if isinstance(module, nn.Identity):
            return input
        raise ValueError(f"unsupported module for lattice export: {type(module).__name__}")

    def conv3d(
        self,
        name: str,
        module: spnn.Conv3d,
        input: ExportValue | None = None,
    ) -> ExportValue:
        value = self._current_or(input)
        self._require_sparse(value, module)
        sparse_type = SparseTensorType(dtype=self._dtype_for_module(module))
        weight = self._weight(
            name,
            "weight",
            _conv_weight_to_mlx(module),
            family="conv3d",
            layout="conv3d_o_zyx_i",
        )
        bias = None
        if module.bias is not None:
            bias = self._weight(name, "bias", module.bias.detach(), family="bias", layout="bias_c")

        kwargs = {
            "input": value.value,
            "weight": weight,
            "bias": bias,
            "kernel_size": _triple(module.kernel_size),
            "result_type": sparse_type,
            "result": _safe_value_name(name),
        }
        if isinstance(module, spnn.GenerativeConvTranspose3d):
            out = self._builder.generative_conv_transpose3d(**kwargs, stride=_triple(module.stride))
        elif isinstance(module, spnn.ConvTranspose3d):
            out = self._builder.conv_transpose3d(
                **kwargs,
                stride=_triple(module.stride),
                padding=_triple(module.padding),
                dilation=_triple(module.dilation),
            )
        elif isinstance(module, spnn.SubmConv3d):
            out = self._builder.subm_conv3d(**kwargs, dilation=_triple(module.dilation))
        elif isinstance(module, spnn.Conv3d):
            out = self._builder.conv3d(
                **kwargs,
                stride=_triple(module.stride),
                padding=_triple(module.padding),
                dilation=_triple(module.dilation),
            )
        else:  # pragma: no cover - protected by lower_module dispatch.
            raise TypeError(f"unsupported convolution module: {type(module).__name__}")
        return ExportValue(out, "sparse_tensor", module.out_channels)

    def target_conv3d(
        self,
        name: str,
        module: spnn.TargetConv3d,
        input: ExportValue,
        target: ExportValue,
    ) -> ExportValue:
        self._require_sparse(input, module)
        self._require_sparse(target, module)
        weight = self._weight(
            name,
            "weight",
            _conv_weight_to_mlx(module),
            family="conv3d",
            layout="conv3d_o_zyx_i",
        )
        bias = None
        if module.bias is not None:
            bias = self._weight(name, "bias", module.bias.detach(), family="bias", layout="bias_c")
        out = self._builder.target_conv3d(
            input=input.value,
            target=target.value,
            weight=weight,
            bias=bias,
            kernel_size=_triple(module.kernel_size),
            stride=_triple(module.stride),
            padding=_triple(module.padding),
            dilation=_triple(module.dilation),
            result_type=SparseTensorType(dtype=self._dtype_for_module(module)),
            result=_safe_value_name(name),
        )
        return ExportValue(out, "sparse_tensor", module.out_channels)

    def pool3d(
        self,
        name: str,
        module: spnn.Pool3d,
        input: ExportValue | None = None,
    ) -> ExportValue:
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
        return ExportValue(out, "sparse_tensor", value.channels)

    def voxelize(
        self,
        name: str,
        *,
        points: ExportValue,
        features: ExportValue,
        batch_indices: ExportValue,
        active_rows: ExportValue,
        voxel_size,
        origin=0.0,
        reduction: Literal["sum", "mean"] = "mean",
        stride=1,
    ) -> ExportValue:
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
        return ExportValue(out, "sparse_tensor", features.channels)

    def devoxelize(
        self,
        name: str,
        *,
        points: ExportValue,
        voxels: ExportValue,
        batch_indices: ExportValue,
        point_active_rows: ExportValue,
        voxel_size,
        origin=0.0,
        interpolation: Literal["nearest", "linear"] = "nearest",
    ) -> ExportValue:
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
        return ExportValue(out, "dense_tensor", voxels.channels)

    def batch_norm(
        self,
        name: str,
        module: spnn.BatchNorm,
        input: ExportValue | None = None,
    ) -> ExportValue:
        value = self._current_or(input)
        self._require_sparse(value, module)
        sparse, features = self._sparse_features(name, value)
        dtype = self._dtype_for_module(module)
        scale = self._optional_channel_weight(name, "weight", module.weight, dtype=dtype)
        bias = self._optional_channel_weight(name, "bias", module.bias, family="bias", layout="bias_c", dtype=dtype)
        mean = self._optional_channel_weight(name, "running_mean", module.running_mean, dtype=dtype)
        var = self._optional_channel_weight(name, "running_var", module.running_var, dtype=dtype)
        if scale is None:
            scale = self._constant_channel_weight(name, "weight", torch.ones(module.num_features), dtype=dtype)
        if bias is None:
            bias = self._constant_channel_weight(name, "bias", torch.zeros(module.num_features), family="bias", layout="bias_c", dtype=dtype)
        if mean is None or var is None:
            raise ValueError("BatchNorm export requires frozen running_mean and running_var.")
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
        return ExportValue(out, "sparse_tensor", module.num_features)

    def layer_norm(
        self,
        name: str,
        module: nn.LayerNorm,
        input: ExportValue | None = None,
    ) -> ExportValue:
        value = self._current_or(input)
        sparse, features, output_kind = self._feature_input(name, value)
        channels = _single_normalized_dim(module.normalized_shape, "LayerNorm")
        dtype = self._dtype_for_module(module)
        scale = self._optional_channel_weight(name, "weight", module.weight, dtype=dtype)
        bias = self._optional_channel_weight(name, "bias", module.bias, family="bias", layout="bias_c", dtype=dtype)
        if scale is None:
            scale = self._constant_channel_weight(name, "weight", torch.ones(channels), dtype=dtype)
        if bias is None:
            bias = self._constant_channel_weight(name, "bias", torch.zeros(channels), family="bias", layout="bias_c", dtype=dtype)
        out_features = self._builder.layer_norm(
            input=features,
            scale=scale,
            bias=bias,
            eps=float(module.eps),
            result_type=_feature_type(channels, dtype),
            result=f"{_safe_value_name(name)}_features",
        )
        return self._feature_output(name, sparse, out_features, output_kind, channels, dtype)

    def rms_norm(
        self,
        name: str,
        module: nn.Module,
        input: ExportValue | None = None,
    ) -> ExportValue:
        value = self._current_or(input)
        sparse, features, output_kind = self._feature_input(name, value)
        channels = _single_normalized_dim(module.normalized_shape, "RMSNorm")
        dtype = self._dtype_for_module(module)
        scale = self._optional_channel_weight(name, "weight", getattr(module, "weight", None), dtype=dtype)
        if scale is None:
            scale = self._constant_channel_weight(name, "weight", torch.ones(channels), dtype=dtype)
        out_features = self._builder.rms_norm(
            input=features,
            scale=scale,
            eps=float(module.eps),
            result_type=_feature_type(channels, dtype),
            result=f"{_safe_value_name(name)}_features",
        )
        return self._feature_output(name, sparse, out_features, output_kind, channels, dtype)

    def activation(
        self,
        name: str,
        kind: str,
        *,
        input: ExportValue | None = None,
        approximate: str = "none",
        alpha: float = 0.01,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> ExportValue:
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
            return ExportValue(out, "sparse_tensor", value.channels)
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
        return ExportValue(out, "dense_tensor", value.channels)

    def global_pool(
        self,
        name: str,
        mode: Literal["avg", "max"],
        input: ExportValue | None = None,
    ) -> ExportValue:
        value = self._current_or(input)
        self._require_sparse_name(value, "GlobalPool")
        if self.batch_size is None:
            raise ValueError("global_pool export requires a static batch_size.")
        channels = value.channels
        out = self._builder.global_pool(
            input=value.value,
            mode=mode,
            batch_size=int(self.batch_size),
            result_type=_feature_type(channels, self.input_dtype),
            result=_safe_value_name(name),
        )
        return ExportValue(out, "dense_tensor", channels)

    def linear(
        self,
        name: str,
        module: nn.Linear,
        input: ExportValue | None = None,
    ) -> ExportValue:
        value = self._current_or(input)
        self._require_dense(value, module)
        dtype = _torch_dtype_name(module.weight.dtype)
        weight = self._weight(name, "weight", module.weight.detach(), family="linear", layout="linear_o_i")
        bias = None
        if module.bias is not None:
            bias = self._weight(name, "bias", module.bias.detach(), family="bias", layout="bias_c")
        out = self._builder.linear(
            input=value.value,
            weight=weight,
            bias=bias,
            result_type=_feature_type(module.out_features, dtype),
            result=_safe_value_name(name),
        )
        return ExportValue(out, "dense_tensor", module.out_features)

    def sparse_binary(
        self,
        name: str,
        lhs: ExportValue,
        rhs: ExportValue,
        op: str,
        *,
        join: str = "outer",
        lhs_fill: float = 0.0,
        rhs_fill: float = 0.0,
    ) -> ExportValue:
        self._require_sparse_name(lhs, f"sparse {op} lhs")
        self._require_sparse_name(rhs, f"sparse {op} rhs")
        if lhs.channels is not None and rhs.channels is not None and lhs.channels != rhs.channels:
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
        return ExportValue(out, "sparse_tensor", lhs.channels or rhs.channels)

    def sparse_add(self, name: str, lhs: ExportValue, rhs: ExportValue, *, join: str = "outer", lhs_fill: float = 0.0, rhs_fill: float = 0.0) -> ExportValue:
        return self.sparse_binary(name, lhs, rhs, "add", join=join, lhs_fill=lhs_fill, rhs_fill=rhs_fill)

    def sparse_cat(self, name: str, lhs: ExportValue, rhs: ExportValue, *, join: str = "inner") -> ExportValue:
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
        return ExportValue(out, "sparse_tensor", channels)

    def output(self, value: ExportValue | None = None, *, name: str | None = None) -> None:
        value = value or self.current
        role = "sparse_tensor" if value.kind == "sparse_tensor" else "tensor"
        self._builder.return_(value.value, names=(name or self.output_name,), roles=(role,))
        self._finalized = True

    def to_mlir(self) -> str:
        if not self._finalized:
            self.output()
        return self._builder.to_mlir()

    def save(self, artifact_dir: str | Path, *, clean: bool = True, validate: bool = True):
        del validate
        from .artifact import _save_artifact

        return _save_artifact(artifact_dir, self.to_mlir(), self._weights, clean=clean)

    def _current_or(self, value: ExportValue | None) -> ExportValue:
        if value is not None:
            return value
        return self.current

    def _feature_input(self, name: str, value: ExportValue):
        if value.kind == "sparse_tensor":
            sparse, features = self._sparse_features(name, value)
            return sparse, features, "sparse_tensor"
        self._require_dense_name(value, name)
        return None, value.value, "dense_tensor"

    def _feature_output(self, name: str, sparse, features, kind: str, channels: int | None, dtype: str) -> ExportValue:
        if kind == "sparse_tensor":
            out = self._builder.sparse_with_features(
                input=sparse,
                features=features,
                result_type=SparseTensorType(dtype=dtype),
                result=_safe_value_name(name),
            )
            return ExportValue(out, "sparse_tensor", channels)
        return ExportValue(features, "dense_tensor", channels)

    def _sparse_features(self, name: str, value: ExportValue):
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
        value: ExportValue,
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

    def _weight(self, module_name: str, parameter_name: str, tensor: torch.Tensor, *, family: str, layout: str):
        key = f"{_safe_key(module_name)}.{parameter_name}"
        if key in self._weights or f"{key}.weight" in self._weights:
            raise ValueError(f"duplicate exported weight key: {key}")
        tensor = tensor.detach().cpu().contiguous()
        packing = dense_packing()
        stored_dtype = _torch_dtype_name(tensor.dtype)
        if self.quantize_bits is not None and family in {"conv3d", "linear"}:
            packed = _pack_quantized_weight(
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

    def _optional_channel_weight(self, module_name: str, parameter_name: str, tensor: torch.Tensor | None, *, family: str = "channel", layout: str = "channel_c", dtype: str):
        del dtype
        if tensor is None:
            return None
        return self._weight(module_name, parameter_name, tensor, family=family, layout=layout)

    def _constant_channel_weight(self, module_name: str, parameter_name: str, tensor: torch.Tensor, *, family: str = "channel", layout: str = "channel_c", dtype: str):
        torch_dtype = torch.float16 if dtype == "f16" else torch.float32
        return self._weight(module_name, parameter_name, tensor.to(dtype=torch_dtype), family=family, layout=layout)

    def _require_sparse(self, value: ExportValue, module: nn.Module) -> None:
        self._require_sparse_name(value, type(module).__name__)

    def _require_sparse_name(self, value: ExportValue, name: str) -> None:
        if value.kind != "sparse_tensor":
            raise ValueError(f"{name} expects sparse_tensor, got {value.kind}.")

    def _require_dense(self, value: ExportValue, module: nn.Module) -> None:
        self._require_dense_name(value, type(module).__name__)

    def _require_dense_name(self, value: ExportValue, name: str) -> None:
        if value.kind != "dense_tensor":
            raise ValueError(f"{name} expects dense_tensor, got {value.kind}.")

    def _dtype_for_module(self, module: nn.Module) -> str:
        for parameter in module.parameters(recurse=False):
            return _torch_dtype_name(parameter.dtype)
        return self.input_dtype


@dataclass(frozen=True)
class PackedWeight:
    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor


def _pack_quantized_weight(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    scale_dtype: str,
) -> PackedWeight:
    rows, kernel_rows, out_channels = _weight_rows(tensor.to(torch.float32))
    storage_channels = _round_up(rows.shape[1], group_size)
    if storage_channels != rows.shape[1]:
        rows = torch.nn.functional.pad(rows, (0, storage_channels - rows.shape[1]))
    grouped = rows.reshape(rows.shape[0], -1, group_size)
    maximum = grouped.amax(dim=2)
    minimum = grouped.amin(dim=2)
    qmax = float((1 << bits) - 1)
    scales = (minimum - maximum) / qmax
    biases = maximum
    normalized = torch.where(
        scales.unsqueeze(2) != 0,
        (grouped - biases.unsqueeze(2)) / scales.unsqueeze(2),
        torch.zeros_like(grouped),
    )
    codes = normalized.round().clamp(0, qmax).to(torch.uint32).reshape(rows.shape[0], storage_channels)
    packed = _pack_codes(codes, bits)
    scale_dtype_torch = torch.float16 if scale_dtype == "f16" else torch.float32
    packed = packed.reshape(kernel_rows, out_channels, -1).contiguous()
    scales = scales.to(scale_dtype_torch).reshape(kernel_rows, out_channels, -1).contiguous()
    biases = biases.to(scale_dtype_torch).reshape(kernel_rows, out_channels, -1).contiguous()
    if kernel_rows > 1:
        packed = packed.transpose(1, 2).contiguous()
        scales = scales.transpose(1, 2).contiguous()
        biases = biases.transpose(1, 2).contiguous()
    return PackedWeight(packed.cpu(), scales.cpu(), biases.cpu())


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    values_per_word = 32 // bits
    words = codes.to(torch.int64).reshape(codes.shape[0], -1, values_per_word)
    packed = torch.zeros(words.shape[:2], dtype=torch.int64)
    for lane in range(values_per_word):
        packed |= words[:, :, lane] << (bits * lane)
    return packed.to(torch.uint32)


def _weight_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if tensor.ndim == 2:
        out_channels, _ = tensor.shape
        return tensor, 1, int(out_channels)
    if tensor.ndim == 3:
        kernel_rows, in_channels, out_channels = tensor.shape
        rows = tensor.transpose(1, 2).reshape(kernel_rows * out_channels, in_channels)
        return rows, int(kernel_rows), int(out_channels)
    if tensor.ndim == 5:
        out_channels, kx, ky, kz, in_channels = tensor.shape
        rows = tensor.permute(1, 2, 3, 0, 4).reshape(kx * ky * kz * out_channels, in_channels)
        return rows, int(kx * ky * kz), int(out_channels)
    raise ValueError("quantized export supports linear, kernel-major, and 5D convolution weights.")




def _activation_spec(module: nn.Module) -> dict[str, object] | None:
    specs: tuple[tuple[type[nn.Module], dict[str, object]], ...] = (
        ((spnn.ReLU, nn.ReLU), {"kind": "relu"}),
        ((spnn.SiLU, nn.SiLU), {"kind": "silu"}),
        ((spnn.Sigmoid, nn.Sigmoid), {"kind": "sigmoid"}),
        ((spnn.Tanh, nn.Tanh), {"kind": "tanh"}),
        ((spnn.GELU, nn.GELU), {"kind": "gelu"}),
        ((spnn.Softplus, nn.Softplus), {"kind": "softplus"}),
    )
    for types, attrs in specs:
        if isinstance(module, types):
            out = dict(attrs)
            if isinstance(module, (spnn.GELU, nn.GELU)):
                out["approximate"] = str(module.approximate or "none")
            if isinstance(module, (spnn.Softplus, nn.Softplus)):
                out["beta"] = float(module.beta)
                out["threshold"] = float(module.threshold)
            return out
    if isinstance(module, (spnn.LeakyReLU, nn.LeakyReLU)):
        return {"kind": "leaky_relu", "alpha": float(module.negative_slope)}
    return None


def _is_rms_norm(module: nn.Module) -> bool:
    return isinstance(module, spnn.RMSNorm) or (
        hasattr(nn, "RMSNorm") and isinstance(module, nn.RMSNorm)
    )


def _single_normalized_dim(value, name: str) -> int:
    if isinstance(value, int):
        return int(value)
    dims = tuple(int(item) for item in value)
    if len(dims) != 1:
        raise ValueError(f"{name} export expects one normalized feature dimension.")
    return dims[0]

def _conv_weight_to_mlx(module: spnn.Conv3d) -> torch.Tensor:
    kernel_size = _triple(module.kernel_size)
    weight = module.kernel.detach()
    if weight.ndim == 2:
        weight = weight.reshape(1, weight.shape[0], weight.shape[1])
    expected_kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if weight.ndim != 3 or weight.shape[0] != expected_kernel_volume:
        raise ValueError(f"Conv3d kernel shape {tuple(weight.shape)} does not match kernel_size={kernel_size}.")
    return weight.reshape(*kernel_size, module.in_channels, module.out_channels).permute(4, 0, 1, 2, 3).contiguous()


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
    raise ValueError(f"unsupported tensor dtype for lattice export: {dtype}")


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


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
