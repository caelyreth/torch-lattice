from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import save_file
from torch import nn

from torch_lattice import SparseTensor

try:
    from lattice_contract import ARTIFACT_GRAPH_FILE, ARTIFACT_WEIGHT_FILE
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError(
        "torch_lattice.artifact requires the MLIR artifact API from "
        "lattice-contract; install a lattice-contract build that exports "
        "ARTIFACT_GRAPH_FILE, ARTIFACT_WEIGHT_FILE, MLIRModuleBuilder, and "
        "DIALECT_SCHEMA_DIGEST."
    ) from exc

from .builder import TorchLatticeArtifactBuilder
from .fx import lower_fx_artifact

__all__ = [
    "LatticeArtifact",
    "LatticeArtifactError",
    "LatticeArtifactOptions",
    "save_lattice_artifact",
]

ArtifactMethod = Literal["fx", "explicit"]


class LatticeArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class LatticeArtifactOptions:
    """Options for producing a portable lattice MLIR artifact."""

    input_dtype: str = "f32"
    batch_size: int | None = None
    clean: bool = True
    validate: bool = True
    quantize_bits: int | None = None
    quantize_group_size: int = 32
    quantize_scale_dtype: str = "f16"
    input_stride: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class LatticeArtifact:
    """Result of exporting a Torch model to a lattice artifact directory."""

    artifact_dir: Path
    graph_path: Path
    weights_path: Path
    weight_keys: tuple[str, ...]


def save_lattice_artifact(
    model: nn.Module,
    artifact_dir: str | Path,
    *,
    input_name: str = "input",
    output_name: str = "output",
    sample_input: SparseTensor | None = None,
    options: LatticeArtifactOptions | None = None,
    method: ArtifactMethod = "fx",
) -> LatticeArtifact:
    """Save ``model`` as ``graph.mlir`` plus ``weights.safetensors``.

    ``torch.fx`` is the default front-end. For explicit construction use
    :class:`TorchLatticeArtifactBuilder` directly and call ``save``.
    """

    if method != "fx":
        raise LatticeArtifactError(
            "save_lattice_artifact currently accepts method='fx'; use "
            "TorchLatticeArtifactBuilder for explicit graph construction."
        )

    options = _options_with_sample_defaults(options, sample_input)
    artifact_builder = TorchLatticeArtifactBuilder(
        input_name=input_name,
        output_name=output_name,
        input_dtype=options.input_dtype,
        batch_size=options.batch_size,
        quantize_bits=options.quantize_bits,
        quantize_group_size=options.quantize_group_size,
        quantize_scale_dtype=options.quantize_scale_dtype,
        input_stride=options.input_stride,
    )
    lower_fx_artifact(artifact_builder, model)
    return artifact_builder.save(
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
) -> LatticeArtifact:
    artifact_path = Path(artifact_dir)
    if artifact_path.exists() and clean:
        shutil.rmtree(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)

    graph_path = artifact_path / ARTIFACT_GRAPH_FILE
    graph_path.write_text(graph, encoding="utf-8")

    weights_path = artifact_path / ARTIFACT_WEIGHT_FILE
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in weights.items()},
        weights_path,
        metadata={"format": "torch"},
    )

    return LatticeArtifact(
        artifact_dir=artifact_path,
        graph_path=graph_path,
        weights_path=weights_path,
        weight_keys=tuple(sorted(weights)),
    )


def _options_with_sample_defaults(
    options: LatticeArtifactOptions | None,
    sample_input: SparseTensor | None,
) -> LatticeArtifactOptions:
    options = options or LatticeArtifactOptions()
    dtype = options.input_dtype
    batch_size = options.batch_size
    if sample_input is None:
        return options
    if dtype == "f32":
        dtype = _torch_dtype_name(sample_input.feats.dtype)
    if batch_size is None:
        batch_size = _batch_size_from_sample(sample_input)
    input_stride = tuple(int(value) for value in sample_input.stride)
    return LatticeArtifactOptions(
        input_dtype=dtype,
        batch_size=batch_size,
        input_stride=input_stride,
        clean=options.clean,
        validate=options.validate,
        quantize_bits=options.quantize_bits,
        quantize_group_size=options.quantize_group_size,
        quantize_scale_dtype=options.quantize_scale_dtype,
    )


def _torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.float32:
        return "f32"
    raise LatticeArtifactError(f"unsupported sparse feature dtype: {dtype}")


def _batch_size_from_sample(sample_input: SparseTensor) -> int:
    if sample_input.spatial_range is not None and len(sample_input.spatial_range) > 0:
        return int(sample_input.spatial_range[0])
    if sample_input.coords.numel() == 0:
        return 0
    return int(sample_input.coords[:, 0].max().item()) + 1
