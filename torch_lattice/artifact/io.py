from __future__ import annotations

import inspect
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from torch_lattice import SparseTensor

try:
    from lattice_contract import (
        ARTIFACT_GRAPH_FILE,
        ARTIFACT_WEIGHT_FILE,
        CURRENT_DIALECT_VERSION,
        DIALECT_SCHEMA_DIGEST,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_lattice.artifact requires the MLIR artifact API from lattice-contract"
    ) from exc

from .builder import ArtifactValue, TorchLatticeArtifactBuilder
from .fx import lower_fx_artifact

__all__ = [
    "LatticeArtifactSaveResult",
    "LatticeModelArtifactError",
    "LatticeModelArtifactOptions",
    "save_lattice_model_artifact",
]

_STORAGE_KEY = re.compile(r'storage_key\s*=\s*"([^"]+)"')


class LatticeModelArtifactError(ValueError):
    """Raised when a Torch model cannot form a valid lattice artifact."""


@dataclass(frozen=True, slots=True)
class LatticeModelArtifactOptions:
    """Options for producing a portable lattice MLIR artifact."""

    input_dtype: str | None = None
    batch_size: int | None = None
    clean: bool = True
    validate: bool = True
    quantize_bits: int | None = None
    quantize_group_size: int = 32
    quantize_scale_dtype: str = "f16"


@dataclass(frozen=True, slots=True)
class LatticeArtifactSaveResult:
    """Paths and weight keys written for one lattice artifact."""

    artifact_dir: Path
    graph_path: Path
    weights_path: Path
    weight_keys: tuple[str, ...]


def save_lattice_model_artifact(
    model: nn.Module,
    artifact_dir: str | Path,
    *,
    example_inputs: tuple[Any, ...],
    options: LatticeModelArtifactOptions | None = None,
    output_names: tuple[str, ...] | None = None,
) -> LatticeArtifactSaveResult:
    """Export an eval-mode Torch model as MLIR plus safetensors.

    ``example_inputs`` defines the public artifact ABI. Sparse inputs become
    coordinate, feature, and active-row argument groups; dense tensors retain
    their rank, dtype, and static non-leading dimensions.
    """

    if any(module.training for module in model.modules()):
        raise LatticeModelArtifactError(
            "artifact export requires eval mode via model.eval() so normalization and dropout "
            "semantics are deterministic"
        )
    if not isinstance(example_inputs, tuple) or not example_inputs:
        raise LatticeModelArtifactError("example_inputs must be a non-empty tuple")
    options = options or LatticeModelArtifactOptions()
    names = _input_names(model, len(example_inputs))
    input_dtype = options.input_dtype or _input_feature_dtype(example_inputs)
    batch_size = options.batch_size
    if batch_size is None:
        batch_size = _common_batch_size(example_inputs)

    builder = TorchLatticeArtifactBuilder(
        input_dtype=input_dtype,
        batch_size=batch_size,
        quantize_bits=options.quantize_bits,
        quantize_group_size=options.quantize_group_size,
        quantize_scale_dtype=options.quantize_scale_dtype,
        create_default_input=False,
    )
    symbolic_inputs = tuple(
        _symbolic_input(builder, name, value)
        for name, value in zip(names, example_inputs, strict=True)
    )
    lower_fx_artifact(
        builder,
        model,
        inputs=symbolic_inputs,
        output_names=output_names,
    )
    return builder.save(
        artifact_dir,
        clean=options.clean,
        validate=options.validate,
    )


def _save_artifact(
    artifact_dir: str | Path,
    graph: str,
    weights: dict[str, torch.Tensor],
    *,
    clean: bool,
    validate: bool,
) -> LatticeArtifactSaveResult:
    artifact_path = Path(artifact_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists() and not clean:
        raise FileExistsError(f"artifact directory already exists: {artifact_path}")

    temp = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_path.name}-",
            dir=artifact_path.parent,
        )
    )
    backup = temp.with_name(f"{temp.name}.previous")
    try:
        graph_path = temp / ARTIFACT_GRAPH_FILE
        weights_path = temp / ARTIFACT_WEIGHT_FILE
        graph_path.write_text(graph, encoding="utf-8")
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in weights.items()},
            weights_path,
            metadata={"format": "torch"},
        )
        if validate:
            _validate_payload(graph, weights_path)
        if artifact_path.exists():
            os.replace(artifact_path, backup)
        os.replace(temp, artifact_path)
        if backup.exists():
            _remove_path(backup)
    except Exception:
        _remove_path(temp)
        if backup.exists() and not artifact_path.exists():
            os.replace(backup, artifact_path)
        raise

    return LatticeArtifactSaveResult(
        artifact_dir=artifact_path,
        graph_path=artifact_path / ARTIFACT_GRAPH_FILE,
        weights_path=artifact_path / ARTIFACT_WEIGHT_FILE,
        weight_keys=tuple(sorted(weights)),
    )


def _validate_payload(graph: str, weights_path: Path) -> None:
    expected_attrs = (
        f"lattice.ir_version = {CURRENT_DIALECT_VERSION}",
        f'lattice.schema_digest = "{DIALECT_SCHEMA_DIGEST}"',
        f'lattice.weight_file = "{ARTIFACT_WEIGHT_FILE}"',
        "func.func @forward(",
        "return ",
    )
    missing_attrs = [item for item in expected_attrs if item not in graph]
    if missing_attrs:
        raise LatticeModelArtifactError(
            "artifact graph is missing contract metadata: " + ", ".join(missing_attrs)
        )
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = frozenset(handle.keys())
    storage_keys = frozenset(_STORAGE_KEY.findall(graph))
    for storage_key in storage_keys:
        if storage_key not in keys and not any(
            key.startswith(f"{storage_key}.") for key in keys
        ):
            raise LatticeModelArtifactError(
                f"artifact weight is missing storage key {storage_key!r}"
            )
    unexpected = sorted(
        key
        for key in keys
        if not any(
            key == storage_key or key.startswith(f"{storage_key}.")
            for storage_key in storage_keys
        )
    )
    if unexpected:
        raise LatticeModelArtifactError(
            "artifact contains unreferenced weights: " + ", ".join(unexpected)
        )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _symbolic_input(
    builder: TorchLatticeArtifactBuilder,
    name: str,
    value: Any,
) -> ArtifactValue:
    if isinstance(value, SparseTensor):
        return builder.sparse_argument(
            name,
            dtype=_torch_dtype_name(value.feats.dtype),
            channels=int(value.feats.shape[1]),
            stride=value.stride,
        )
    if isinstance(value, torch.Tensor):
        return builder.dense_argument(
            name,
            _tensor_type(value),
            channels=int(value.shape[-1]) if value.ndim == 2 else None,
        )
    raise LatticeModelArtifactError(
        f"unsupported artifact input {name!r}: {type(value).__name__}"
    )


def _input_names(model: nn.Module, count: int) -> tuple[str, ...]:
    parameters = tuple(inspect.signature(model.forward).parameters.values())
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ):
        raise LatticeModelArtifactError("variadic model inputs are not exportable")
    parameters = tuple(
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    required = sum(
        parameter.default is inspect.Parameter.empty
        for parameter in inspect.signature(model.forward).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if count < required or count > len(parameters):
        raise LatticeModelArtifactError(
            f"model forward accepts {required}..{len(parameters)} positional inputs, "
            f"got {count}"
        )
    return parameters[:count]


def _input_feature_dtype(inputs: tuple[Any, ...]) -> str:
    dtypes = {
        _torch_dtype_name(value.feats.dtype)
        for value in inputs
        if isinstance(value, SparseTensor)
    }
    if len(dtypes) > 1:
        raise LatticeModelArtifactError("all sparse inputs must use one feature dtype")
    return next(iter(dtypes), "f32")


def _common_batch_size(inputs: tuple[Any, ...]) -> int | None:
    sizes = {
        len(value.batch_counts)
        if value.batch_counts is not None
        else int(value.spatial_range[0])
        for value in inputs
        if isinstance(value, SparseTensor)
        and (value.batch_counts is not None or value.spatial_range is not None)
    }
    if len(sizes) > 1:
        raise LatticeModelArtifactError("sparse inputs disagree on batch size")
    return next(iter(sizes), None)


def _tensor_type(value: torch.Tensor) -> str:
    dtype = _torch_dtype_name(value.dtype)
    if value.ndim == 0:
        return f"tensor<{dtype}>"
    dims = ["?", *(str(int(item)) for item in value.shape[1:])]
    return f"tensor<{'x'.join(dims)}x{dtype}>"


def _torch_dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "f16",
        torch.float32: "f32",
        torch.int32: "i32",
        torch.int64: "i64",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise LatticeModelArtifactError(
            f"unsupported artifact tensor dtype: {dtype}"
        ) from exc
