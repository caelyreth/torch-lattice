"""One-time conversion of explicitly described TorchSparse checkpoints.

This module belongs to conformance tooling rather than ``torch_lattice``. A
checkpoint does not describe its kernel-row enumeration, so conversion requires
an explicit kernel specification instead of guessing from tensor shapes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from lattice_contract import (
    CANONICAL_CONV3D_WEIGHT_LAYOUT,
    kernel_positions,
    kernel_row_permutation,
)
from safetensors.torch import save_file

_MANIFEST_SCHEMA = 'lattice-checkpoint-conversion/v2'
_SOURCE_LAYOUT_POLICY = 'torchsparse_k_i_o_volume_parity'
_TARGET_LAYOUT = 'lattice_k_i_o_z_fastest'

type SourceKernelLayout = Literal['torchsparse', 'minkowski_engine']


@dataclass(frozen=True, slots=True)
class ConvertedKernel:
    """One kernel permutation recorded in a conversion manifest."""

    source_key: str
    target_key: str
    kernel_size: tuple[int, int, int]
    source_layout: str
    row_permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CheckpointConversion:
    """Paths and immutable details from one checkpoint conversion."""

    source: Path
    output: Path
    manifest: Path
    tensor_count: int
    kernels: tuple[ConvertedKernel, ...]


@dataclass(frozen=True, slots=True)
class StateDictConversion:
    """Canonical state prepared for one concrete Torch Lattice model."""

    state_dict: dict[str, torch.Tensor]
    kernels: tuple[ConvertedKernel, ...]


def convert_torchsparse_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    kernel_specs: Mapping[str, Sequence[int]],
    manifest: str | Path | None = None,
) -> CheckpointConversion:
    """Convert a trusted legacy state mapping to canonical lattice weights.

    ``kernel_specs`` maps each legacy sparse-kernel state key to its exact
    ``(Kx, Ky, Kz)`` size. Every ``.kernel`` tensor must be listed. The source
    is deliberately loaded with PyTorch's pickle-based checkpoint reader, so
    this function must only be used with trusted checkpoint files.
    """

    source_path = Path(source)
    output_path = Path(output)
    normalized_specs = _normalize_kernel_specs(kernel_specs)
    state = _state_mapping(
        torch.load(source_path, map_location='cpu', weights_only=False)
    )
    converted: dict[str, torch.Tensor] = {}
    records: list[ConvertedKernel] = []

    for source_key, value in state.items():
        name = str(source_key)
        if not isinstance(value, torch.Tensor):
            continue
        if name.endswith('.kernel') and name not in normalized_specs:
            raise ValueError(
                f'legacy sparse kernel {name!r} has no kernel specification.'
            )
        if name in normalized_specs:
            target_name, converted_value, record = _convert_kernel(
                name, value, normalized_specs[name]
            )
            records.append(record)
        else:
            target_name = name
            converted_value = value
        if target_name in converted:
            raise ValueError(
                f'checkpoint conversion maps multiple tensors to {target_name!r}.'
            )
        converted[target_name] = converted_value.detach().cpu().contiguous()

    missing = sorted(set(normalized_specs) - set(str(key) for key in state))
    if missing:
        raise ValueError(
            'checkpoint is missing declared kernel tensors: ' + ', '.join(missing)
        )
    if not converted:
        raise ValueError('checkpoint contains no tensor state.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, output_path)
    manifest_path = (
        Path(manifest) if manifest is not None else output_path.with_suffix('.json')
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                'schema': _MANIFEST_SCHEMA,
                'source_checkpoint': str(source_path),
                'source_kernel_layout_policy': _SOURCE_LAYOUT_POLICY,
                'target_kernel_layout': _TARGET_LAYOUT,
                'target_artifact_weight_layout': CANONICAL_CONV3D_WEIGHT_LAYOUT,
                'tensor_count': len(converted),
                'kernels': [asdict(record) for record in records],
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    return CheckpointConversion(
        source=source_path,
        output=output_path,
        manifest=manifest_path,
        tensor_count=len(converted),
        kernels=tuple(records),
    )


def migrate_sparse_state_dict(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    kernel_sizes: Mapping[str, Sequence[int]],
    source_layout: SourceKernelLayout,
) -> StateDictConversion:
    """Prepare a strict canonical state for a concrete sparse model.

    ``target`` must be the state dictionary of the model that will receive the
    result. ``kernel_sizes`` maps fully qualified sparse-module names to their
    ``(Kx, Ky, Kz)`` geometry. Legacy ``<module>.kernel`` tensors are renamed,
    reshaped when pointwise, and permuted into the canonical row order. The
    target's derived ``kernel_positions`` buffers are retained rather than
    copied from a legacy checkpoint.

    This is intentionally model-state-only. Optimizer state carries parameter
    identity and moment tensors that require an explicit training-resume policy.
    """

    normalized_sizes = {
        str(name): _triple(size, name=f'kernel size for {name!r}')
        for name, size in kernel_sizes.items()
    }
    pending = {str(name): value for name, value in source.items()}
    converted: dict[str, torch.Tensor] = {}
    records: list[ConvertedKernel] = []

    for target_key, target_value in target.items():
        module_name, _, field = target_key.rpartition('.')
        kernel_size = normalized_sizes.get(module_name)
        if kernel_size is not None and field == 'weight':
            source_key = f'{module_name}.kernel'
            try:
                source_value = pending.pop(source_key)
            except KeyError as exc:
                raise ValueError(
                    f'legacy state is missing sparse kernel {source_key!r}.'
                ) from exc
            value, record = _canonical_kernel(
                source_key,
                source_value,
                kernel_size,
                source_layout=source_layout,
            )
            if tuple(value.shape) != tuple(target_value.shape):
                raise ValueError(
                    f'converted kernel {source_key!r} has shape '
                    f'{tuple(value.shape)} but target {target_key!r} expects '
                    f'{tuple(target_value.shape)}.'
                )
            converted[target_key] = value.to(
                dtype=target_value.dtype,
                device=target_value.device,
            )
            records.append(record)
            continue

        if kernel_size is not None and field == 'kernel_positions':
            converted[target_key] = target_value.detach().clone()
            continue

        try:
            value = pending.pop(target_key)
        except KeyError as exc:
            raise ValueError(
                f'legacy state is missing target value {target_key!r}.'
            ) from exc
        if tuple(value.shape) != tuple(target_value.shape):
            raise ValueError(
                f'legacy value {target_key!r} has shape {tuple(value.shape)} '
                f'but the canonical model expects {tuple(target_value.shape)}.'
            )
        converted[target_key] = value.to(
            dtype=target_value.dtype,
            device=target_value.device,
        )

    if pending:
        raise ValueError(
            'legacy state has no canonical target for: '
            + ', '.join(sorted(pending))
        )
    return StateDictConversion(converted, tuple(records))


def load_kernel_specs(path: str | Path) -> dict[str, tuple[int, int, int]]:
    """Read a strict ``{"state.key": [Kx, Ky, Kz]}`` kernel-spec mapping."""

    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('kernel specification must be a JSON object.')
    return _normalize_kernel_specs(raw)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Convert a trusted TorchSparse checkpoint to lattice safetensors.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--kernel-spec',
        type=Path,
        required=True,
        help='JSON mapping of legacy sparse-kernel state keys to [Kx, Ky, Kz].',
    )
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args(argv)
    result = convert_torchsparse_checkpoint(
        args.source,
        args.output,
        kernel_specs=load_kernel_specs(args.kernel_spec),
        manifest=args.manifest,
    )
    print(
        json.dumps(
            {
                'output': str(result.output),
                'manifest': str(result.manifest),
                'tensor_count': result.tensor_count,
                'kernel_count': len(result.kernels),
            },
            indent=2,
        )
    )


def _convert_kernel(
    source_key: str,
    value: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> tuple[str, torch.Tensor, ConvertedKernel]:
    expected_rows = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if value.ndim != 3 or int(value.shape[0]) != expected_rows:
        raise ValueError(
            f'legacy kernel {source_key!r} has shape {tuple(value.shape)}; '
            f'expected ({expected_rows}, Cin, Cout) for kernel_size={kernel_size}.'
        )
    if not source_key.endswith('.kernel'):
        raise ValueError(
            f'kernel specification key {source_key!r} must name a legacy .kernel tensor.'
        )
    converted, record = _canonical_kernel(
        source_key,
        value,
        kernel_size,
        source_layout='torchsparse',
    )
    target_key = source_key.removesuffix('.kernel') + '.weight'
    return (
        target_key,
        converted,
        record,
    )


def _canonical_kernel(
    source_key: str,
    value: torch.Tensor,
    kernel_size: tuple[int, int, int],
    *,
    source_layout: SourceKernelLayout,
) -> tuple[torch.Tensor, ConvertedKernel]:
    expected_rows = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if expected_rows == 1 and value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or int(value.shape[0]) != expected_rows:
        raise ValueError(
            f'legacy kernel {source_key!r} has shape {tuple(value.shape)}; '
            f'expected ({expected_rows}, Cin, Cout) for kernel_size={kernel_size}.'
        )
    layout_name, source_positions = _source_positions(
        kernel_size, source_layout
    )
    permutation = kernel_row_permutation(
        source_positions, kernel_positions(kernel_size)
    )
    rows = torch.tensor(permutation, dtype=torch.long, device=value.device)
    target_key = source_key.removesuffix('.kernel') + '.weight'
    return (
        value.index_select(0, rows),
        ConvertedKernel(
            source_key,
            target_key,
            kernel_size,
            layout_name,
            permutation,
        ),
    )


def _source_positions(
    kernel_size: tuple[int, int, int],
    source_layout: SourceKernelLayout,
) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    if source_layout == 'torchsparse':
        return _torchsparse_positions(kernel_size)
    if source_layout == 'minkowski_engine':
        x_size, y_size, z_size = kernel_size
        return (
            'minkowski_engine_hypercube_k_i_o_x_fastest',
            tuple(
                (x, y, z)
                for z in range(z_size)
                for y in range(y_size)
                for x in range(x_size)
            ),
        )
    raise ValueError(f'unsupported source kernel layout: {source_layout!r}.')


def _normalize_kernel_specs(
    value: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, int, int]]:
    normalized: dict[str, tuple[int, int, int]] = {}
    for key, shape in value.items():
        name = str(key)
        dimensions = tuple(int(item) for item in shape)
        if not name.endswith('.kernel'):
            raise ValueError(
                f'kernel specification key {name!r} must end with .kernel.'
            )
        if len(dimensions) != 3 or any(item <= 0 for item in dimensions):
            raise ValueError(
                f'kernel specification for {name!r} must contain three positive integers.'
            )
        normalized[name] = dimensions
    return normalized


def _triple(value: Sequence[int], *, name: str) -> tuple[int, int, int]:
    values = tuple(int(item) for item in value)
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f'{name} must contain three positive integers.')
    return values[0], values[1], values[2]


def _state_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError('checkpoint must be a mapping.')
    for key in ('model_state_dict', 'state_dict', 'model'):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return nested
    return value


def _torchsparse_positions(
    kernel_size: tuple[int, int, int],
) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    """Return the historical TorchSparse row convention for one kernel.

    TorchSparse used x-fastest rows only for odd kernel volumes. Its even-volume
    path already used x/y/z nesting with z fastest. This branch is retained here
    solely to migrate an explicitly declared legacy checkpoint; library runtime
    code never accepts either representation.
    """

    x_size, y_size, z_size = kernel_size
    if x_size * y_size * z_size % 2 == 0:
        return 'torchsparse_k_i_o_z_fastest', kernel_positions(kernel_size)
    return (
        'torchsparse_k_i_o_x_fastest',
        tuple(
            (x, y, z)
            for z in range(z_size)
            for y in range(y_size)
            for x in range(x_size)
        ),
    )


__all__ = [
    'CheckpointConversion',
    'ConvertedKernel',
    'StateDictConversion',
    'convert_torchsparse_checkpoint',
    'load_kernel_specs',
    'main',
    'migrate_sparse_state_dict',
]
