from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

import torch

from torch_lattice import SparseTensor

type Mode = Literal['cold_op', 'hot_op', 'backward']
type Params = Mapping[str, Any]
type WorkloadMetrics = dict[str, int | float]
type MetricFactory = Callable[[Params, Any, Any | None, Any | None], WorkloadMetrics]


class SkipCase(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    group: str
    params: tuple[Params, ...]
    setup: Callable[[Params], Any]
    prepare: Callable[[Any], Any]
    run: Callable[[Any], Any]
    backward: Callable[[Any], Any] | None = None
    metrics: MetricFactory | None = None
    units: tuple[str, ...] = ()
    modes: tuple[Mode, ...] | None = None

    def supports(self, mode: Mode) -> bool:
        if self.modes is not None and mode not in self.modes:
            return False
        return mode != 'backward' or self.backward is not None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    case: str
    group: str
    mode: Mode
    device: str
    params: dict[str, Any]
    warmup: int
    repeats: int
    median_ms: float | None
    min_ms: float | None
    p90_ms: float | None
    p95_ms: float | None
    samples_ms: tuple[float, ...]
    workload: WorkloadMetrics
    units: dict[str, float]
    skipped: bool = False
    notes: str = ''

    def to_json(self) -> dict[str, Any]:
        return {
            'case': self.case,
            'group': self.group,
            'mode': self.mode,
            'device': self.device,
            'params': _jsonable(self.params),
            'warmup': self.warmup,
            'repeats': self.repeats,
            'median_ms': self.median_ms,
            'min_ms': self.min_ms,
            'p90_ms': self.p90_ms,
            'p95_ms': self.p95_ms,
            'samples_ms': list(self.samples_ms),
            'workload': self.workload,
            'units': self.units,
            'skipped': self.skipped,
            'notes': self.notes,
        }


type ProgressStart = Callable[[BenchmarkCase, Params, Mode, str], None]
type ProgressResult = Callable[[BenchmarkResult, BenchmarkCase, Params, Mode, str], None]
type ProgressSkip = Callable[[BenchmarkCase, Params, Mode, str], None]
type ProgressError = Callable[[BenchmarkCase, Params, Mode, str, BaseException], None]


class CudaTimer:
    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __call__(self, fn: Callable[[], Any], warmup: int, repeats: int) -> tuple[tuple[float, ...], Any | None]:
        last_output = None
        for _ in range(warmup):
            last_output = fn()
            force(last_output)
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(repeats):
            self.start.record()
            last_output = fn()
            force(last_output)
            self.end.record()
            torch.cuda.synchronize()
            samples.append(float(self.start.elapsed_time(self.end)))
        return tuple(samples), last_output


def run_case(
    case: BenchmarkCase,
    params: Params,
    *,
    mode: Mode,
    device: str,
    warmup: int,
    repeats: int,
) -> BenchmarkResult | None:
    if not case.supports(mode):
        return None
    fixture = case.setup(params)
    force(fixture)
    prepared = case.prepare(fixture)
    force(prepared)
    action = _action(case, fixture, prepared, mode)
    timer = CudaTimer()
    torch.cuda.reset_peak_memory_stats()
    samples, output = timer(action, warmup, repeats)
    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    workload = _derive_workload_metrics(params, fixture=fixture, prepared=prepared, output=output, extra=case.metrics)
    workload['memory_mb'] = memory_mb
    return BenchmarkResult(
        case=case.name,
        group=case.group,
        mode=mode,
        device=device,
        params=dict(params),
        warmup=warmup,
        repeats=repeats,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        p90_ms=_percentile(samples, 90),
        p95_ms=_percentile(samples, 95),
        samples_ms=samples,
        workload=workload,
        units=_derive_units(samples, params, workload, case.units),
    )


def run_cases(
    cases: Iterable[BenchmarkCase],
    *,
    modes: Sequence[Mode],
    device: str,
    warmup: int,
    repeats: int,
    include: str | None = None,
    keep_going: bool = True,
    on_start: ProgressStart | None = None,
    on_result: ProgressResult | None = None,
    on_skip: ProgressSkip | None = None,
    on_error: ProgressError | None = None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for case in cases:
        if include is not None and include not in case.name:
            continue
        for params in case.params:
            for mode in modes:
                if not case.supports(mode):
                    if on_skip is not None:
                        on_skip(case, params, mode, device)
                    continue
                if on_start is not None:
                    on_start(case, params, mode, device)
                try:
                    result = run_case(case, params, mode=mode, device=device, warmup=warmup, repeats=repeats)
                except Exception as error:
                    if on_error is not None:
                        on_error(case, params, mode, device, error)
                    if not keep_going:
                        raise
                    result = _skip_result(case, params, mode, device, warmup, repeats, error)
                if result is not None:
                    if on_result is not None:
                        on_result(result, case, params, mode, device)
                    results.append(result)
    return results


def force(value: Any) -> None:
    tensors = tuple(_collect_tensors(value))
    if tensors:
        torch.cuda.synchronize()


def _action(case: BenchmarkCase, fixture: Any, prepared: Any, mode: Mode) -> Callable[[], Any]:
    if mode == 'cold_op':
        return lambda: case.run(case.prepare(fixture))
    if mode == 'hot_op':
        return lambda: case.run(prepared)
    if case.backward is None:
        raise ValueError(f'{case.name} does not support backward.')
    return lambda: case.backward(case.prepare(fixture))


def _skip_result(
    case: BenchmarkCase,
    params: Params,
    mode: Mode,
    device: str,
    warmup: int,
    repeats: int,
    error: BaseException,
) -> BenchmarkResult:
    return BenchmarkResult(
        case=case.name,
        group=case.group,
        mode=mode,
        device=device,
        params=dict(params),
        warmup=warmup,
        repeats=repeats,
        median_ms=None,
        min_ms=None,
        p90_ms=None,
        p95_ms=None,
        samples_ms=(),
        workload=_derive_workload_metrics(params, fixture=None, prepared=None, output=None),
        units={},
        skipped=True,
        notes=f'{type(error).__name__}: {error}',
    )


def _collect_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, SparseTensor):
        yield value.feats
        yield value.coords
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _collect_tensors(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            yield from _collect_tensors(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _collect_tensors(getattr(value, field.name))


def _derive_workload_metrics(
    params: Params,
    *,
    fixture: Any,
    prepared: Any | None,
    output: Any | None,
    extra: MetricFactory | None = None,
) -> WorkloadMetrics:
    metrics: WorkloadMetrics = {}
    for source in (params,):
        for key, target in (('N', 'N'), ('points', 'points'), ('channels', 'channels_in'), ('kernel', 'kernel_volume')):
            value = source.get(key)
            if isinstance(value, int | float):
                metrics.setdefault(target, value)
    for value in (prepared, fixture):
        if isinstance(value, SparseTensor):
            metrics.setdefault('n_in', int(value.feats.shape[0]))
            metrics.setdefault('channels_in', int(value.feats.shape[1]))
            break
    if isinstance(output, SparseTensor):
        metrics['n_out'] = int(output.feats.shape[0])
        metrics['channels_out'] = int(output.feats.shape[1])
        metrics['elements'] = int(output.feats.numel())
    elif isinstance(output, torch.Tensor):
        metrics.setdefault('n_out', int(output.shape[0]) if output.ndim else 1)
        metrics['elements'] = int(output.numel())
    elif isinstance(output, dict):
        coords = output.get('coords')
        if isinstance(coords, torch.Tensor):
            metrics['n_out'] = int(coords.shape[0])
    if extra is not None:
        metrics.update(extra(params, fixture, prepared, output))
    edges = metrics.get('edges')
    n_out = metrics.get('n_out')
    if isinstance(edges, int | float) and isinstance(n_out, int | float) and n_out > 0:
        metrics['avg_neighbors'] = edges / n_out
    return metrics


def _derive_units(samples: Sequence[float], params: Params, workload: WorkloadMetrics, units: Sequence[str]) -> dict[str, float]:
    median_seconds = statistics.median(samples) / 1000.0
    if median_seconds <= 0.0:
        return {}
    out = {}
    for unit in units or _default_units(workload):
        raw = workload.get(unit, params.get(unit))
        if isinstance(raw, int | float):
            out[f'{unit}_per_s'] = float(raw) / median_seconds
    return out


def _default_units(workload: WorkloadMetrics) -> tuple[str, ...]:
    return tuple(unit for unit in ('edges', 'elements', 'points', 'n_out', 'n_in', 'N') if unit in workload)


def _percentile(samples: Sequence[float], pct: int) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


__all__ = ['BenchmarkCase', 'BenchmarkResult', 'Mode', 'SkipCase', 'run_case', 'run_cases']
