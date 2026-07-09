from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator, Literal

import torch

PackageName = Literal['torch_lattice', 'torchsparse']
Family = Literal[
    'pointwise_chain',
    'branch_add',
    'branch_cat',
    'global_pool',
    'batchnorm_chain',
    'spatial_subm_mapping',
    'stride2_forward',
]

FAMILIES: tuple[Family, ...] = (
    'pointwise_chain',
    'branch_add',
    'branch_cat',
    'global_pool',
    'batchnorm_chain',
    'spatial_subm_mapping',
    'stride2_forward',
)


@dataclass(frozen=True)
class CompatCase:
    family: Family
    seed: int
    output_kind: Literal['sparse', 'dense']


def main() -> None:
    args = _parse_args()
    if args.command == 'run':
        _run_package(args.package, Path(args.output), cases=args.cases, seed=args.seed, device=args.device)
        return
    if args.command == 'compare':
        report = _compare(Path(args.left), Path(args.right))
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        if report['failed']:
            raise SystemExit(1)
        return
    if args.command == 'all':
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        current = out_dir / 'torch_lattice.pt'
        original = out_dir / 'torchsparse.pt'
        report = out_dir / 'report.json'
        _subprocess_run('torch_lattice', current, cases=args.cases, seed=args.seed, device=args.device)
        _subprocess_run('torchsparse', original, cases=args.cases, seed=args.seed, device=args.device)
        result = _compare(current, original)
        report.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result['summary'], indent=2))
        if result['failed']:
            raise SystemExit(1)
        return
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check original TorchSparse -> torch-lattice migration semantics.'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run')
    run.add_argument('--package', choices=('torch_lattice', 'torchsparse'), required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--cases', type=int, default=70)
    run.add_argument('--seed', type=int, default=20260709)
    run.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')

    compare = sub.add_parser('compare')
    compare.add_argument('--left', required=True)
    compare.add_argument('--right', required=True)
    compare.add_argument('--output', required=True)

    all_cmd = sub.add_parser('all')
    all_cmd.add_argument('--output', default='/tmp/torch_lattice_torchsparse_compat')
    all_cmd.add_argument('--cases', type=int, default=70)
    all_cmd.add_argument('--seed', type=int, default=20260709)
    all_cmd.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    return parser.parse_args()


def _subprocess_run(package: PackageName, output: Path, *, cases: int, seed: int, device: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        'run',
        '--package',
        package,
        '--output',
        str(output),
        '--cases',
        str(cases),
        '--seed',
        str(seed),
        '--device',
        device,
    ]
    subprocess.run(command, check=True)


def _run_package(package: PackageName, output: Path, *, cases: int, seed: int, device: str) -> None:
    lattice, spnn = _import_package(package)
    selected_device = _device(device)
    rows: list[dict[str, Any]] = []
    with _gather_scatter_conv(package):
        for index, case in enumerate(_cases(cases, seed)):
            torch.manual_seed(case.seed)
            x = _input(lattice, case.seed).to(selected_device)
            model = _model(package, spnn, lattice, case.family).to(selected_device).eval()
            with torch.no_grad():
                y = model(x)
            rows.append(_serialize_output(index, case, y))
    torch.save({'package': package, 'cases': rows}, output)



@contextmanager
def _gather_scatter_conv(package: PackageName) -> Iterator[None]:
    try:
        conv_module = __import__(
            f"{package}.nn.functional.conv",
            fromlist=["Dataflow", "conv_config"],
        )
        dataflow = conv_module.Dataflow
        conv_config = conv_module.conv_config
        previous = conv_config.get_global_conv_config()
        config = conv_config.get_default_conv_config()
        config.dataflow = dataflow.GatherScatter
        config.ifsort = False
        conv_config.set_global_conv_config(config)
    except Exception:
        previous = None
        conv_config = None
    try:
        yield
    finally:
        if conv_config is not None:
            if previous is None:
                conv_config.clear_global_conv_config()
            else:
                conv_config.set_global_conv_config(previous)

def _import_package(package: PackageName):
    if package == 'torch_lattice':
        import torch_lattice as lattice
        from torch_lattice import nn as spnn
    else:
        import torchsparse as lattice
        from torchsparse import nn as spnn
    return lattice, spnn


def _device(value: str) -> torch.device:
    if value == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable.')
        return torch.device('cuda')
    if value == 'auto' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _cases(count: int, seed: int) -> list[CompatCase]:
    out: list[CompatCase] = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        output_kind = 'dense' if family == 'global_pool' else 'sparse'
        out.append(CompatCase(family, seed + index * 9973, output_kind))
    return out


def _input(lattice, seed: int):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    coords = torch.tensor(
        [[batch, x, y, 0] for batch in range(2) for x in range(5) for y in range(2)],
        dtype=torch.int32,
    )
    feats = torch.randn((coords.shape[0], 4), generator=generator) * 0.25
    return lattice.SparseTensor(feats=feats, coords=coords, spatial_range=(2, 5, 2, 1))


def _model(package: PackageName, spnn, lattice, family: Family):
    if family == 'pointwise_chain':
        return torch.nn.Sequential(
            spnn.Conv3d(4, 6, kernel_size=1, bias=True),
            spnn.ReLU(),
            spnn.Conv3d(6, 3, kernel_size=1, bias=False),
        )
    if family == 'branch_add':
        return _Branch(spnn, lattice, merge='add')
    if family == 'branch_cat':
        return _Branch(spnn, lattice, merge='cat')
    if family == 'global_pool':
        return torch.nn.Sequential(
            spnn.Conv3d(4, 5, kernel_size=1, bias=True),
            spnn.ReLU(),
            spnn.GlobalAvgPool(),
        )
    if family == 'batchnorm_chain':
        model = torch.nn.Sequential(
            spnn.Conv3d(4, 4, kernel_size=1, bias=False),
            spnn.BatchNorm(4),
            spnn.ReLU(),
        )
        model.eval()
        return model
    if family == 'spatial_subm_mapping':
        if package == 'torch_lattice' and hasattr(spnn, 'SubmConv3d'):
            return torch.nn.Sequential(spnn.SubmConv3d(4, 3, kernel_size=3, bias=True))
        return torch.nn.Sequential(spnn.Conv3d(4, 3, kernel_size=3, stride=1, bias=True))
    if family == 'stride2_forward':
        return torch.nn.Sequential(spnn.Conv3d(4, 3, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=True))
    raise AssertionError(family)


class _Branch(torch.nn.Module):
    def __init__(self, spnn, lattice, *, merge: Literal['add', 'cat']) -> None:
        super().__init__()
        self.left = spnn.Conv3d(4, 3, kernel_size=1, bias=False)
        self.right = spnn.Conv3d(4, 3, kernel_size=1, bias=False)
        self.tail = spnn.Conv3d(6 if merge == 'cat' else 3, 2, kernel_size=1, bias=True)
        self.lattice = lattice
        self.merge = merge

    def forward(self, x):
        lhs = self.left(x)
        rhs = self.right(x)
        if self.merge == 'cat':
            merged = self.lattice.cat([lhs, rhs])
        else:
            merged = lhs + rhs
        return self.tail(merged)


def _serialize_output(index: int, case: CompatCase, output) -> dict[str, Any]:
    row = {'index': index, 'family': case.family, 'seed': case.seed, 'kind': case.output_kind}
    if case.output_kind == 'dense':
        row['output'] = output.detach().cpu()
        return row
    row['coords'] = output.coords.detach().cpu()
    row['feats'] = output.feats.detach().cpu()
    return row


def _compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = torch.load(left_path, map_location='cpu', weights_only=False)
    right = torch.load(right_path, map_location='cpu', weights_only=False)
    rows: list[dict[str, Any]] = []
    failed = 0
    for lhs, rhs in zip(left['cases'], right['cases'], strict=True):
        result = _compare_case(lhs, rhs)
        failed += 0 if result['ok'] else 1
        rows.append(result)
    max_abs = max((row['max_abs'] for row in rows), default=0.0)
    max_rel = max((row['max_rel'] for row in rows), default=0.0)
    return {
        'left_package': left['package'],
        'right_package': right['package'],
        'failed': failed,
        'summary': {
            'cases': len(rows),
            'failed': failed,
            'max_abs': max_abs,
            'max_rel': max_rel,
            'families': sorted({row['family'] for row in rows}),
        },
        'cases': rows,
    }


def _compare_case(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    if lhs['family'] != rhs['family'] or lhs['kind'] != rhs['kind']:
        raise ValueError('case order mismatch')
    if lhs['kind'] == 'dense':
        left = lhs['output']
        right = rhs['output']
        coord_equal = True
    else:
        coord_equal = torch.equal(lhs['coords'], rhs['coords'])
        left = lhs['feats']
        right = rhs['feats']
    diff = (left - right).abs()
    denom = torch.maximum(left.abs(), right.abs()).clamp_min(1e-12)
    rel = diff / denom
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    ok = coord_equal and max_abs == 0.0 and max_rel == 0.0
    return {
        'index': lhs['index'],
        'family': lhs['family'],
        'kind': lhs['kind'],
        'coords_equal': coord_equal,
        'max_abs': max_abs,
        'max_rel': max_rel,
        'ok': ok,
    }


if __name__ == '__main__':
    main()
