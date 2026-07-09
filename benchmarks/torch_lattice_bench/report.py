from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

import torch_lattice
from torch_lattice_bench.harness import BenchmarkResult


def environment() -> dict[str, Any]:
    return {
        'git_sha': _git_sha(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'torch_version': torch.__version__,
        'torch_lattice_version': getattr(torch_lattice, '__version__', 'unknown'),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        'allow_tf32': torch_lattice.backends.allow_tf32,
        'allow_fp16': torch_lattice.backends.allow_fp16,
        'hash_rsv_ratio': torch_lattice.backends.hash_rsv_ratio,
    }


def write_json(path: Path, *, results: list[BenchmarkResult]) -> None:
    payload = {'environment': environment(), 'results': [result.to_json() for result in results]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_summary(path: Path, *, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table(results, color=False) + '\n', encoding='utf-8')


def table(results: list[BenchmarkResult], *, color: bool = False) -> str:
    if not results:
        return _style('No benchmark results.', color, '33')
    headers = ['case', 'mode', 'device', 'params', 'workload', 'median_ms', 'p95_ms', 'throughput', 'notes']
    rows = []
    for result in results:
        rows.append([
            f'{result.group}/{result.case}',
            result.mode,
            result.device,
            _params(result.params),
            _workload(result.workload),
            'skip' if result.median_ms is None else f'{result.median_ms:.3f}',
            'skip' if result.p95_ms is None else f'{result.p95_ms:.3f}',
            _throughput(result.units),
            result.notes[:80],
        ])
    widths = [max(len(headers[col]), *(len(row[col]) for row in rows)) for col in range(len(headers))]
    lines = [
        '  '.join(_style(headers[col].ljust(widths[col]), color, '1') for col in range(len(headers))),
        '  '.join('-' * width for width in widths),
    ]
    lines.extend('  '.join(row[col].ljust(widths[col]) for col in range(len(row))) for row in rows)
    return '\n'.join(lines)


def _params(params: dict[str, Any]) -> str:
    return ','.join(f'{key}={params[key]}' for key in ('N', 'points', 'channels', 'layout', 'dtype', 'kernel', 'stride') if key in params)


def _workload(workload: dict[str, int | float]) -> str:
    labels = (
        ('points', 'P'),
        ('n_in', 'Nin'),
        ('n_out', 'Nout'),
        ('edges', 'E'),
        ('channels_in', 'Cin'),
        ('channels_out', 'Cout'),
        ('kernel_volume', 'K'),
        ('avg_neighbors', 'avgN'),
        ('memory_mb', 'MB'),
    )
    parts = []
    for key, label in labels:
        value = workload.get(key)
        if isinstance(value, int):
            parts.append(f'{label}={value}')
        elif isinstance(value, float):
            parts.append(f'{label}={value:.2f}')
    return ','.join(parts)


def _throughput(units: dict[str, float]) -> str:
    return ', '.join(f'{value:.1f} {key}' for key, value in list(units.items())[:2])


def _style(text: str, enabled: bool, *codes: str) -> str:
    if not enabled or not codes:
        return text
    return f'\033[{";".join(codes)}m{text}\033[0m'


def _git_sha() -> str:
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return result.stdout.strip()


__all__ = ['environment', 'table', 'write_json', 'write_summary']
