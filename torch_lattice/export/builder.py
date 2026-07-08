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
    """Explicit builder for Torch-to-lattice MLIR artifacts.

    The public ``module`` method keeps the ergonomic sequential-builder API.
    ``lower_module`` and sparse merge helpers are used by the FX interpreter so
    branch topology is represented directly in MLIR without relying on mutable
    current-value side effects.
    """

    def __init__(
        self,
        *,
        input_name: str = "input",
        output_name: str = "output",
        input_dtype: str = "f32",
        batch_size: int | None = None,
    ) -> None:
        self.input_name = input_name
        self.output_name = output_name
        self.input_dtype = input_dtype
        self.batch_size = batch_size
        self._builder = MLIRModuleBuilder()
        self._weights: dict[str, torch.Tensor] = {}
        self._value = self.sparse_input()
        self._finalized = False

    @property
    def current(self) -> ExportValue:
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
            stride=(1, 1, 1),
            coord_order="batch_x_y_z",
            result_type=sparse_type,
            result=self.input_name,
        )
        return ExportValue(value=value, kind="sparse_tensor", channels=None)

    def module(self, name: str, module: nn.Module) -> ExportValue:
        """Append ``module`` to the sequential explicit-builder chain."""

        value = self.lower_module(name, module, self._value)
        self._value = value
        return value

    def lower_module(
        self,
        name: str,
        module: nn.Module,
        input: ExportValue,
    ) -> ExportValue:
        """Lower one supported module against an explicit symbolic input."""

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
        if isinstance(module, spnn.InstanceNorm):
            raise ValueError("InstanceNorm export is not supported by the lattice MLIR slice.")
        if isinstance(module, spnn.GroupNorm):
            raise ValueError("GroupNorm export is not supported by the lattice MLIR slice.")
        if isinstance(module, (spnn.ReLU, nn.ReLU)):
            return self.activation(name, "relu", input=input)
        if isinstance(module, (spnn.LeakyReLU, nn.LeakyReLU)):
            return self.activation(name, "leaky_relu", input=input, alpha=float(module.negative_slope))
        if isinstance(module, (spnn.SiLU, nn.SiLU)):
            return self.activation(name, "silu", input=input)
        if isinstance(module, spnn.GlobalAvgPool):
            return self.global_pool(name, "avg", input)
        if isinstance(module, spnn.GlobalMaxPool):
            return self.global_pool(name, "max", input)
        if isinstance(module, nn.Linear):
            return self.linear(name, module, input)
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
            bias = self._weight(
                name,
                "bias",
                module.bias.detach(),
                family="bias",
                layout="bias_c",
            )

        kwargs = {
            "input": value.value,
            "weight": weight,
            "bias": bias,
            "kernel_size": _triple(module.kernel_size),
            "result_type": sparse_type,
            "result": _safe_value_name(name),
        }
        if isinstance(module, spnn.GenerativeConvTranspose3d):
            out = self._builder.generative_conv_transpose3d(
                **kwargs,
                stride=_triple(module.stride),
            )
        elif isinstance(module, spnn.ConvTranspose3d):
            out = self._builder.conv_transpose3d(
                **kwargs,
                stride=_triple(module.stride),
                padding=_triple(module.padding),
                dilation=_triple(module.dilation),
            )
        elif isinstance(module, spnn.SubmConv3d):
            out = self._builder.subm_conv3d(
                **kwargs,
                dilation=_triple(module.dilation),
            )
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

    def activation(
        self,
        name: str,
        kind: str,
        *,
        input: ExportValue | None = None,
        alpha: float = 0.01,
    ) -> ExportValue:
        value = self._current_or(input)
        if value.kind == "sparse_tensor":
            sparse, features = self._sparse_features(name, value)
            out_features = self._activation_features(name, features, value, kind, alpha=alpha)
            out = self._builder.sparse_with_features(
                input=sparse,
                features=out_features,
                result_type=SparseTensorType(dtype=self.input_dtype),
                result=_safe_value_name(name),
            )
            return ExportValue(out, "sparse_tensor", value.channels)
        out = self._activation_features(name, value.value, value, kind, alpha=alpha)
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
        weight = self._weight(
            name,
            "weight",
            module.weight.detach(),
            family="linear",
            layout="linear_o_i",
        )
        bias = None
        if module.bias is not None:
            bias = self._weight(
                name,
                "bias",
                module.bias.detach(),
                family="bias",
                layout="bias_c",
            )
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
        return ExportValue(out, "sparse_tensor", lhs.channels or rhs.channels)

    def sparse_add(
        self,
        name: str,
        lhs: ExportValue,
        rhs: ExportValue,
        *,
        join: str = "outer",
        lhs_fill: float = 0.0,
        rhs_fill: float = 0.0,
    ) -> ExportValue:
        return self.sparse_binary(
            name,
            lhs,
            rhs,
            "add",
            join=join,
            lhs_fill=lhs_fill,
            rhs_fill=rhs_fill,
        )

    def sparse_cat(
        self,
        name: str,
        lhs: ExportValue,
        rhs: ExportValue,
        *,
        join: str = "inner",
    ) -> ExportValue:
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
        value = value or self._value
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

        return _save_artifact(
            artifact_dir,
            self.to_mlir(),
            self._weights,
            clean=clean,
        )

    def _current_or(self, value: ExportValue | None) -> ExportValue:
        return self._value if value is None else value

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

    def _activation_features(self, name: str, features, value: ExportValue, kind: str, *, alpha: float):
        return self._builder.activation(
            input=features,
            kind=kind,
            approximate="none",
            alpha=float(alpha),
            beta=1.0,
            threshold=20.0,
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
        if key in self._weights:
            raise ValueError(f"duplicate exported weight key: {key}")
        tensor = tensor.detach().cpu().contiguous()
        self._weights[key] = tensor
        return self._builder.weight(
            sym_name=_safe_value_name(key),
            storage_key=key,
            layout=layout,
            packing=dense_packing(),
            result_type=WeightType(family, _torch_dtype_name(tensor.dtype)),
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
        return self._weight(module_name, parameter_name, tensor, family=family, layout=layout)

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

    def _require_sparse(self, value: ExportValue, module: nn.Module) -> None:
        self._require_sparse_name(value, type(module).__name__)

    def _require_sparse_name(self, value: ExportValue, name: str) -> None:
        if value.kind != "sparse_tensor":
            raise ValueError(f"{name} expects sparse_tensor, got {value.kind}.")

    def _require_dense(self, value: ExportValue, module: nn.Module) -> None:
        if value.kind != "dense_tensor":
            raise ValueError(f"{type(module).__name__} expects dense_tensor, got {value.kind}.")

    def _dtype_for_module(self, module: nn.Module) -> str:
        for parameter in module.parameters(recurse=False):
            return _torch_dtype_name(parameter.dtype)
        return self.input_dtype


def _conv_weight_to_mlx(module: spnn.Conv3d) -> torch.Tensor:
    kernel_size = _triple(module.kernel_size)
    weight = module.kernel.detach()
    if weight.ndim == 2:
        weight = weight.reshape(1, weight.shape[0], weight.shape[1])
    expected_kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if weight.ndim != 3 or weight.shape[0] != expected_kernel_volume:
        raise ValueError(
            f"Conv3d kernel shape {tuple(weight.shape)} does not match kernel_size={kernel_size}."
        )
    return (
        weight.reshape(*kernel_size, module.in_channels, module.out_channels)
        .permute(4, 0, 1, 2, 3)
        .contiguous()
    )


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


def _torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.float32:
        return "f32"
    raise ValueError(f"unsupported tensor dtype for lattice export: {dtype}")


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
