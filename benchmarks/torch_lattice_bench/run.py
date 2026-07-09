from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import torch

import torch_lattice
from torch_lattice_bench.catalog import GROUPS, MODES, PRESETS
from torch_lattice_bench.cases import all_cases
from torch_lattice_bench.console import make_console
from torch_lattice_bench.report import write_json, write_summary

_RESULTS_DIR = Path('benchmarks/results')


def main() -> None:
    args = _parser().parse_args()
    if args.smoke:
        args.preset = 'smoke'
    if not torch.cuda.is_available() or not str(args.device).startswith('cuda'):
        raise RuntimeError('torch-lattice benchmark requires a CUDA device.')

    _configure_backend(args)
    device = torch.device(args.device)
    groups = tuple(args.group) if args.group else GROUPS
    modes = tuple(args.mode) if args.mode else _default_modes(groups)
    n_values = tuple(args.n_values) if args.n_values else None
    channels = tuple(args.channels) if args.channels else None
    layouts = tuple(args.layout) if args.layout else None

    console = make_console(args.color, quiet=args.quiet)
    cases = all_cases(
        args.preset,
        groups=groups,
        n_values=n_values,
        channels=channels,
        layouts=layouts,
        dtype=args.dtype,
        device=device,
    )

    if args.list:
        for case in cases:
            print(f'{case.group}/{case.name}')
        return

    total = _count_runs(cases, modes, args.case_filter)
    console.set_total(total)
    json_path, summary_path = _report_paths(args)

    from torch_lattice_bench.harness import run_cases

    console.heading(f'device {args.device}')
    results = run_cases(
        cases,
        modes=modes,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        include=args.case_filter,
        keep_going=not args.fail_fast,
        on_start=console.start,
        on_result=console.done,
        on_skip=console.skipped,
        on_error=console.failed,
    )
    write_json(json_path, results=results)
    write_summary(summary_path, results=results)
    console.report(json_path, summary_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='torch-lattice-bench',
        description='Benchmark torch-lattice CUDA sparse operator and module surfaces.',
    )
    parser.add_argument('--preset', choices=PRESETS, default='smoke')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', action='append', choices=MODES)
    parser.add_argument('--group', '--groups', dest='group', action='append', choices=GROUPS)
    parser.add_argument('--case-filter')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', '--iters', dest='repeats', type=int, default=20)
    parser.add_argument('--size', '--points', dest='n_values', action='append', type=_positive_int)
    parser.add_argument('--channels', action='append', type=_positive_int)
    parser.add_argument('--layout', '--pattern', '--patterns', dest='layout', action='append', choices=('isolated', 'line', 'plane', 'grid', 'block2', 'block3', 'block4', 'block8'))
    parser.add_argument('--dtype', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--output')
    parser.add_argument('--color', choices=('auto', 'always', 'never'), default='auto')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Alias for --preset smoke.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-tf32', dest='allow_tf32', action='store_true', default=True)
    parser.add_argument('--no-allow-tf32', dest='allow_tf32', action='store_false')
    parser.add_argument('--allow-fp16', dest='allow_fp16', action='store_true', default=True)
    parser.add_argument('--no-allow-fp16', dest='allow_fp16', action='store_false')
    parser.add_argument('--hash-rsv-ratio', type=int, default=64)
    parser.add_argument('--fail-fast', action='store_true')
    return parser


def _configure_backend(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch_lattice.backends.allow_tf32 = args.allow_tf32
    torch_lattice.backends.allow_fp16 = args.allow_fp16
    torch_lattice.backends.hash_rsv_ratio = max(args.hash_rsv_ratio, torch_lattice.backends.hash_rsv_ratio)
    torch_lattice.backends.benchmark = True


def _default_modes(groups: Sequence[str]) -> tuple[str, ...]:
    if tuple(groups) == ('train',):
        return ('backward',)
    return ('cold_op', 'hot_op')


def _count_runs(cases, modes: Sequence[str], include: str | None) -> int:
    total = 0
    for case in cases:
        if include is not None and include not in case.name:
            continue
        for _params in case.params:
            total += sum(1 for mode in modes if case.supports(mode))
    return total


def _report_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output:
        json_path = Path(args.output)
        if not json_path.is_absolute() and json_path.parent == Path('.'):
            json_path = _RESULTS_DIR / json_path
        if json_path.suffix != '.json':
            json_path = json_path.with_suffix('.json')
    else:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        json_path = _RESULTS_DIR / f'torch-lattice-bench-{args.preset}-{stamp}.json'
    return json_path, json_path.with_suffix('.summary.txt')


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


if __name__ == '__main__':
    main()
