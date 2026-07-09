from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from safetensors.torch import save_file
from torch import nn

import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice import nn as spnn
from torch_lattice.artifact import (
    LatticeModelArtifactOptions,
    TorchLatticeArtifactBuilder,
    dequantize_artifact_weight,
    lower_fx_artifact,
    save_lattice_model_artifact,
)
from torch_lattice.nn.functional.conv import Dataflow, conv_config

FixtureFamily = Literal[
    'sparse_classifier',
    'sparse_feature_chain',
    'sparse_branch',
    'target_branch',
    'transpose_chain',
    'generative_transpose',
    'point_voxel',
    'quantized_classifier',
]

DEFAULT_FAMILIES: tuple[FixtureFamily, ...] = (
    'sparse_classifier',
    'sparse_feature_chain',
    'sparse_branch',
    'target_branch',
    'transpose_chain',
    'generative_transpose',
    'point_voxel',
    'quantized_classifier',
)


@dataclass(frozen=True)
class SparseInputSpec:
    channels: int
    batch_size: int
    spatial_shape: tuple[int, int, int]
    rows_per_batch: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    stride: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class FuzzCase:
    name: str
    family: FixtureFamily
    seed: int
    output_kind: Literal['dense', 'sparse']
    input_mode: Literal['sparse_kwargs', 'dense_kwargs']
    rtol: float
    atol: float
    metadata: dict[str, Any]


class SparseChainModel(nn.Module):
    def __init__(
        self,
        stages: list[nn.Module],
        *,
        global_pool: nn.Module | None = None,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.global_pool = global_pool
        self.head = head

    def forward(self, x: SparseTensor):
        for stage in self.stages:
            x = stage(x)
        if self.global_pool is None:
            return x
        out = self.global_pool(x)
        if self.head is not None:
            out = self.head(out)
        return out


class SparseBranchModel(nn.Module):
    def __init__(
        self,
        left: nn.Module,
        right: nn.Module,
        tail: nn.Module,
        *,
        merge: Literal['add', 'cat'],
        join: Literal['inner', 'outer'],
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.tail = tail
        self.merge = merge
        self.join = join

    def forward(self, x: SparseTensor) -> SparseTensor:
        lhs = self.left(x)
        rhs = self.right(x)
        if self.merge == 'cat':
            merged = torch_lattice.cat([lhs, rhs], join=self.join)
        else:
            merged = torch_lattice.sparse_add(lhs, rhs, join=self.join)
        return self.tail(merged)


class TargetBranchModel(nn.Module):
    def __init__(self, pre: nn.Module, target_conv: spnn.TargetConv3d) -> None:
        super().__init__()
        self.pre = pre
        self.target_conv = target_conv

    def forward(self, x: SparseTensor, target: SparseTensor) -> SparseTensor:
        return self.target_conv(self.pre(x), target)


class PointVoxelRoundTrip(nn.Module):
    def __init__(
        self,
        *,
        voxel_size: tuple[float, float, float],
        origin: tuple[float, float, float],
        reduction: Literal['sum', 'mean'],
        interpolation: Literal['nearest', 'linear'],
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.origin = origin
        self.reduction = reduction
        self.interpolation = interpolation

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        batch_indices: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> torch.Tensor:
        voxels = torch_lattice.voxelize(
            points,
            features,
            batch_indices=batch_indices,
            active_rows=active_rows,
            voxel_size=self.voxel_size,
            origin=self.origin,
            reduction=self.reduction,
        )
        return torch_lattice.devoxelize(
            points,
            voxels,
            batch_indices=batch_indices,
            point_active_rows=active_rows,
            voxel_size=self.voxel_size,
            origin=self.origin,
            interpolation=self.interpolation,
        )


@contextmanager
def _conv_dataflow(dataflow: Dataflow, *, kmap_mode: str | None = None) -> Iterator[None]:
    previous = conv_config.get_global_conv_config()
    config = conv_config.get_default_conv_config()
    config.dataflow = dataflow
    config.ifsort = False
    if kmap_mode is not None:
        config.kmap_mode = kmap_mode
    conv_config.set_global_conv_config(config)
    try:
        yield
    finally:
        if previous is None:
            conv_config.clear_global_conv_config()
        else:
            conv_config.set_global_conv_config(previous)


def main() -> None:
    args = _parse_args()
    families = _parse_families(args.families)
    root = Path(args.output)
    if root.exists() and args.clean:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    cases: list[dict[str, Any]] = []

    for index in range(args.cases):
        family = families[index % len(families)] if args.round_robin else rng.choice(families)
        seed = rng.randrange(1 << 31)
        case = _build_case(
            root,
            index=index,
            family=family,
            seed=seed,
            train_steps=args.train_steps,
            device=device,
        )
        cases.append(
            {
                'name': case.name,
                'family': case.family,
                'seed': case.seed,
                'output_kind': case.output_kind,
                'input_mode': case.input_mode,
                'rtol': case.rtol,
                'atol': case.atol,
                **case.metadata,
            }
        )

    manifest = {
        'schema': 'torch_lattice_fuzz_fixtures.v1',
        'seed': args.seed,
        'case_count': len(cases),
        'device': str(device),
        'families': list(families),
        'cases': cases,
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.archive:
        _archive(root, Path(args.archive))
    print(root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate randomized Torch CUDA -> lattice MLIR fuzz fixtures.'
    )
    parser.add_argument('--output', default='/tmp/torch_lattice_fuzz_fixtures')
    parser.add_argument('--archive', default=None)
    parser.add_argument('--cases', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260709)
    parser.add_argument('--train-steps', type=int, default=4)
    parser.add_argument(
        '--families',
        default=','.join(DEFAULT_FAMILIES),
        help='Comma-separated family names, or "all".',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cuda', 'cpu'),
        default='auto',
    )
    parser.add_argument('--round-robin', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--clean', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _parse_families(value: str) -> tuple[FixtureFamily, ...]:
    if value.strip() == 'all':
        return DEFAULT_FAMILIES
    names = tuple(item.strip() for item in value.split(',') if item.strip())
    allowed = set(DEFAULT_FAMILIES)
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f'unknown fixture families: {unknown}')
    if not names:
        raise ValueError('at least one fixture family is required.')
    return names  # type: ignore[return-value]


def _select_device(value: str) -> torch.device:
    if value == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        return torch.device('cuda')
    if value == 'auto' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _build_case(
    root: Path,
    *,
    index: int,
    family: FixtureFamily,
    seed: int,
    train_steps: int,
    device: torch.device,
) -> FuzzCase:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    name = f'{index:04d}_{family}_{seed:08x}'
    case_dir = root / name
    case_dir.mkdir(parents=True, exist_ok=False)
    if family == 'point_voxel':
        case = _point_voxel_case(case_dir, name=name, seed=seed, rng=rng, device=device)
    elif family == 'target_branch':
        case = _target_branch_case(case_dir, name=name, seed=seed, rng=rng, train_steps=train_steps, device=device)
    elif family == 'sparse_branch':
        case = _sparse_branch_case(case_dir, name=name, seed=seed, rng=rng, train_steps=train_steps, device=device)
    elif family == 'transpose_chain':
        case = _transpose_case(case_dir, name=name, seed=seed, rng=rng, train_steps=train_steps, device=device)
    elif family == 'generative_transpose':
        case = _generative_transpose_case(case_dir, name=name, seed=seed, rng=rng, train_steps=train_steps, device=device)
    elif family == 'quantized_classifier':
        case = _classifier_case(case_dir, name=name, seed=seed, rng=rng, train_steps=train_steps, device=device, quantize_bits=rng.choice((4, 8)))
    elif family == 'sparse_feature_chain':
        case = _sparse_feature_chain_case(case_dir, name=name, seed=seed, rng=rng, train_steps=train_steps, device=device)
    else:
        case = _classifier_case(case_dir, name=name, seed=seed, rng=rng, train_steps=train_steps, device=device, quantize_bits=None)
    (case_dir / 'case.json').write_text(json.dumps(case.metadata | {
        'name': case.name,
        'family': case.family,
        'seed': case.seed,
        'output_kind': case.output_kind,
        'input_mode': case.input_mode,
        'rtol': case.rtol,
        'atol': case.atol,
    }, indent=2), encoding='utf-8')
    return case


def _classifier_case(
    case_dir: Path,
    *,
    name: str,
    seed: int,
    rng: random.Random,
    train_steps: int,
    device: torch.device,
    quantize_bits: int | None,
) -> FuzzCase:
    in_channels = rng.choice((2, 3, 4, 5))
    hidden = rng.choice((4, 6, 8))
    out_channels = rng.choice((2, 3, 5))
    spec = _sparse_spec(rng, channels=in_channels)
    x = _random_sparse_input(spec, rng)
    stages, ops = _random_sparse_stages(rng, in_channels, hidden, classifier=True)
    model = SparseChainModel(stages, global_pool=spnn.GlobalAvgPool(), head=nn.Linear(hidden, out_channels))
    target = torch.randn((spec.batch_size, out_channels), generator=_torch_generator(seed + 17)) * 0.25
    _train_dense(model, x, target, steps=train_steps, device=device)
    model.eval()
    expected_model = (
        _quantized_reference_model(
            model,
            bits=quantize_bits,
            group_size=32,
            scale_dtype="f16",
        )
        if quantize_bits is not None
        else model
    )
    expected = _dense_output(expected_model, x, device)
    save_lattice_model_artifact(
        model,
        case_dir,
        sample_input=x,
        options=LatticeModelArtifactOptions(
            batch_size=spec.batch_size,
            quantize_bits=quantize_bits,
            quantize_group_size=32,
        ),
    )
    _save_sparse_inputs(case_dir, '', x)
    save_file({'output': expected}, case_dir / 'expected.safetensors')
    return FuzzCase(
        name,
        'quantized_classifier' if quantize_bits is not None else 'sparse_classifier',
        seed,
        'dense',
        'sparse_kwargs',
        5e-2 if quantize_bits == 4 else 1e-2 if quantize_bits == 8 else 2e-3,
        5e-2 if quantize_bits == 4 else 1e-2 if quantize_bits == 8 else 2e-3,
        {
            'input': _spec_metadata(spec),
            'ops': ops + ['global_avg_pool', f'linear:{hidden}->{out_channels}'],
            'train_steps': train_steps,
            'quantize_bits': quantize_bits,
        },
    )


def _sparse_feature_chain_case(
    case_dir: Path,
    *,
    name: str,
    seed: int,
    rng: random.Random,
    train_steps: int,
    device: torch.device,
) -> FuzzCase:
    in_channels = rng.choice((2, 3, 4))
    hidden = rng.choice((3, 5, 7))
    spec = _sparse_spec(rng, channels=in_channels)
    x = _random_sparse_input(spec, rng)
    stages, ops = _random_sparse_stages(rng, in_channels, hidden, classifier=False)
    model = SparseChainModel(stages)
    _train_sparse(model, x, steps=train_steps, device=device)
    model.eval()
    expected = _sparse_output(model, x, device=device)
    save_lattice_model_artifact(
        model,
        case_dir,
        sample_input=x,
        options=LatticeModelArtifactOptions(batch_size=spec.batch_size),
    )
    _save_sparse_inputs(case_dir, '', x)
    _save_sparse_expected(case_dir, expected)
    return FuzzCase(
        name,
        'sparse_feature_chain',
        seed,
        'sparse',
        'sparse_kwargs',
        2e-3,
        2e-3,
        {'input': _spec_metadata(spec), 'ops': ops, 'train_steps': train_steps},
    )


def _sparse_branch_case(
    case_dir: Path,
    *,
    name: str,
    seed: int,
    rng: random.Random,
    train_steps: int,
    device: torch.device,
) -> FuzzCase:
    channels = rng.choice((2, 3, 4))
    branch_channels = rng.choice((3, 4, 6))
    merge = rng.choice(('add', 'cat'))
    join = rng.choice(('inner', 'outer'))
    spec = _sparse_spec(rng, channels=channels)
    x = _random_sparse_input(spec, rng)
    left = _conv_block(channels, branch_channels, rng, prefix='left')
    right = _conv_block(channels, branch_channels, rng, prefix='right')
    tail_in = branch_channels * 2 if merge == 'cat' else branch_channels
    tail = _conv_block(tail_in, rng.choice((2, 3, 5)), rng, prefix='tail')
    model = SparseBranchModel(left, right, tail, merge=merge, join=join)
    _train_sparse(model, x, steps=train_steps, device=device)
    model.eval()
    expected = _sparse_output(model, x, device=device)
    save_lattice_model_artifact(
        model,
        case_dir,
        sample_input=x,
        options=LatticeModelArtifactOptions(batch_size=spec.batch_size),
    )
    _save_sparse_inputs(case_dir, '', x)
    _save_sparse_expected(case_dir, expected)
    return FuzzCase(
        name,
        'sparse_branch',
        seed,
        'sparse',
        'sparse_kwargs',
        3e-3,
        3e-3,
        {
            'input': _spec_metadata(spec),
            'merge': merge,
            'join': join,
            'train_steps': train_steps,
            'ops': [f'branch_{merge}:{join}', f'tail_channels:{tail_in}'],
        },
    )


def _target_branch_case(
    case_dir: Path,
    *,
    name: str,
    seed: int,
    rng: random.Random,
    train_steps: int,
    device: torch.device,
) -> FuzzCase:
    in_channels = rng.choice((2, 3, 4))
    hidden = rng.choice((3, 4, 5))
    out_channels = rng.choice((2, 3))
    spec = _sparse_spec(rng, channels=in_channels)
    x = _random_sparse_input(spec, rng)
    target = _target_from_input(x, rng)
    pre = _conv_block(in_channels, hidden, rng, prefix='pre')
    target_conv = spnn.TargetConv3d(hidden, out_channels, kernel_size=1, bias=rng.choice((True, False)))
    model = TargetBranchModel(pre, target_conv)
    _train_sparse(model, x, target, steps=train_steps, device=device)
    model.eval()
    expected = _sparse_output(model, x, target, device=device)
    builder = TorchLatticeArtifactBuilder(input_dtype='f32', batch_size=spec.batch_size)
    target_value = builder.sparse_argument('target', channels=target.feats.shape[1])
    lower_fx_artifact(builder, model, inputs=(builder.current, target_value))
    builder.save(case_dir)
    _save_sparse_inputs(case_dir, '', x, extra={'target': target})
    _save_sparse_expected(case_dir, expected)
    return FuzzCase(
        name,
        'target_branch',
        seed,
        'sparse',
        'sparse_kwargs',
        3e-3,
        3e-3,
        {
            'input': _spec_metadata(spec),
            'target_rows': int(target.feats.shape[0]),
            'train_steps': train_steps,
            'ops': [f'target_conv:{hidden}->{out_channels}'],
        },
    )


def _transpose_case(
    case_dir: Path,
    *,
    name: str,
    seed: int,
    rng: random.Random,
    train_steps: int,
    device: torch.device,
) -> FuzzCase:
    in_channels = rng.choice((2, 3))
    hidden = rng.choice((3, 4))
    out_channels = rng.choice((2, 3))
    width = rng.choice((4, 6, 8))
    spec = SparseInputSpec(
        channels=in_channels,
        batch_size=1,
        spatial_shape=(width, 1, 1),
        rows_per_batch=(width,),
    )
    x = _contiguous_line_input(spec, seed + 41)
    model = SparseChainModel([
        spnn.Conv3d(in_channels, hidden, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False),
        spnn.ConvTranspose3d(hidden, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=True),
    ])
    with _conv_dataflow(Dataflow.GatherScatter):
        _train_sparse(model, x, steps=train_steps, device=device)
        model.eval()
        expected = _sparse_output(model, x, device=device)
        save_lattice_model_artifact(model, case_dir, sample_input=x)
    _save_sparse_inputs(case_dir, '', x)
    _save_sparse_expected(case_dir, expected)
    return FuzzCase(
        name,
        'transpose_chain',
        seed,
        'sparse',
        'sparse_kwargs',
        3e-3,
        3e-3,
        {'input': _spec_metadata(spec), 'train_steps': train_steps, 'ops': ['conv_stride2', 'conv_transpose_stride2']},
    )


def _generative_transpose_case(
    case_dir: Path,
    *,
    name: str,
    seed: int,
    rng: random.Random,
    train_steps: int,
    device: torch.device,
) -> FuzzCase:
    in_channels = rng.choice((2, 3))
    out_channels = rng.choice((2, 3, 4))
    width = rng.choice((2, 3, 4))
    spec = SparseInputSpec(
        channels=in_channels,
        batch_size=1,
        spatial_shape=(width, 1, 1),
        rows_per_batch=(rng.randint(1, width),),
        stride=(2, 1, 1),
    )
    x = _random_sparse_input(spec, rng)
    del device
    model = SparseChainModel([
        spnn.GenerativeConvTranspose3d(in_channels, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=True),
        spnn.Tanh(),
    ]).eval()
    expected = _generative_transpose_reference(model, x)
    save_lattice_model_artifact(model, case_dir, sample_input=x)
    _save_sparse_inputs(case_dir, '', x)
    _save_sparse_expected(case_dir, expected)
    return FuzzCase(
        name,
        'generative_transpose',
        seed,
        'sparse',
        'sparse_kwargs',
        3e-3,
        3e-3,
        {'input': _spec_metadata(spec), 'train_steps': 0, 'reference': 'python_generative_transpose', 'ops': ['generative_conv_transpose_stride2']},
    )


def _point_voxel_case(
    case_dir: Path,
    *,
    name: str,
    seed: int,
    rng: random.Random,
    device: torch.device,
) -> FuzzCase:
    del device
    batch_size = 1
    channels = rng.choice((2, 3, 5))
    rows_per_batch = (rng.randint(3, 7),)
    point_count = sum(rows_per_batch)
    points = torch.rand((point_count, 3), generator=_torch_generator(seed + 100)) * rng.choice((2.0, 3.0, 4.0))
    features = torch.randn((point_count, channels), generator=_torch_generator(seed + 101)) * 0.5
    batch_indices = torch.cat([
        torch.full((count,), batch, dtype=torch.int32)
        for batch, count in enumerate(rows_per_batch)
    ])
    active_rows = torch.tensor(rows_per_batch, dtype=torch.int32)
    reduction = rng.choice(('sum', 'mean'))
    interpolation = 'nearest'
    model = PointVoxelRoundTrip(
        voxel_size=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        reduction=reduction,
        interpolation=interpolation,
    ).eval()
    expected = model(points, features, batch_indices, active_rows).detach().cpu()
    builder = TorchLatticeArtifactBuilder(input_dtype='f32', create_default_input=False)
    points_value = builder.dense_argument('points', 'tensor<?x3xf32>')
    features_value = builder.dense_argument('features', f'tensor<?x{channels}xf32>', channels=channels)
    batch_value = builder.dense_argument('batch_indices', 'tensor<?xi32>')
    active_value = builder.dense_argument('active_rows', f'tensor<{batch_size}xi32>')
    lower_fx_artifact(builder, model, inputs=(points_value, features_value, batch_value, active_value))
    builder.save(case_dir)
    save_file(
        {
            'points': points,
            'features': features,
            'batch_indices': batch_indices,
            'active_rows': active_rows,
        },
        case_dir / 'inputs.safetensors',
    )
    save_file({'output': expected}, case_dir / 'expected.safetensors')
    return FuzzCase(
        name,
        'point_voxel',
        seed,
        'dense',
        'dense_kwargs',
        1e-4,
        1e-4,
        {
            'point_count': point_count,
            'channels': channels,
            'batch_size': batch_size,
            'rows_per_batch': list(rows_per_batch),
            'reduction': reduction,
            'interpolation': interpolation,
            'ops': ['voxelize', 'devoxelize'],
        },
    )


def _random_sparse_stages(
    rng: random.Random,
    in_channels: int,
    out_channels: int,
    *,
    classifier: bool,
) -> tuple[list[nn.Module], list[str]]:
    stages: list[nn.Module] = []
    ops: list[str] = []
    channels = in_channels
    depth = rng.randint(2, 5)
    for index in range(depth):
        next_channels = out_channels if index == depth - 1 else rng.choice((3, 4, 5, 6, 8))
        conv_kind = rng.choice(('conv1', 'subm1'))
        if conv_kind == 'conv1':
            stages.append(spnn.Conv3d(channels, next_channels, kernel_size=1, bias=rng.choice((True, False))))
        elif conv_kind == 'subm3':
            stages.append(spnn.SubmConv3d(channels, next_channels, kernel_size=3, bias=rng.choice((True, False))))
        else:
            stages.append(spnn.SubmConv3d(channels, next_channels, kernel_size=1, bias=rng.choice((True, False))))
        ops.append(f'{conv_kind}:{channels}->{next_channels}')
        channels = next_channels
        if rng.random() < 0.55:
            activation = _activation(rng)
            stages.append(activation)
            ops.append(type(activation).__name__)
        if rng.random() < 0.25:
            norm = rng.choice((spnn.LayerNorm(channels), spnn.RMSNorm(channels)))
            stages.append(norm)
            ops.append(type(norm).__name__)
    return stages, ops


def _conv_block(in_channels: int, out_channels: int, rng: random.Random, *, prefix: str) -> nn.Sequential:
    layers: list[nn.Module] = [
        spnn.Conv3d(in_channels, out_channels, kernel_size=1, bias=rng.choice((True, False))),
        _activation(rng),
    ]
    if rng.random() < 0.4:
        layers.append(spnn.LayerNorm(out_channels))
    del prefix
    return nn.Sequential(*layers)


def _activation(rng: random.Random) -> nn.Module:
    choice = rng.choice(('relu', 'leaky_relu', 'silu', 'gelu', 'sigmoid', 'tanh', 'softplus'))
    if choice == 'relu':
        return spnn.ReLU()
    if choice == 'leaky_relu':
        return spnn.LeakyReLU(negative_slope=rng.choice((0.01, 0.05, 0.1)))
    if choice == 'silu':
        return spnn.SiLU()
    if choice == 'gelu':
        return spnn.GELU(approximate=rng.choice(('none', 'tanh')))
    if choice == 'sigmoid':
        return spnn.Sigmoid()
    if choice == 'tanh':
        return spnn.Tanh()
    return spnn.Softplus(beta=rng.choice((1.0, 2.0)), threshold=20.0)


def _sparse_spec(rng: random.Random, *, channels: int) -> SparseInputSpec:
    batch_size = rng.choice((1, 2, 3))
    spatial_shape = (rng.randint(3, 7), rng.randint(1, 4), rng.randint(1, 3))
    volume = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    rows_per_batch = tuple(rng.randint(2, min(volume, 10)) for _ in range(batch_size))
    return SparseInputSpec(channels, batch_size, spatial_shape, rows_per_batch)


def _random_sparse_input(spec: SparseInputSpec, rng: random.Random) -> SparseTensor:
    coords: list[tuple[int, int, int, int]] = []
    for batch, count in enumerate(spec.rows_per_batch):
        candidates = [
            (batch, x, y, z)
            for x in range(spec.spatial_shape[0])
            for y in range(spec.spatial_shape[1])
            for z in range(spec.spatial_shape[2])
        ]
        coords.extend(rng.sample(candidates, k=count))
    coords.sort()
    coord_tensor = torch.tensor(coords, dtype=torch.int32)
    feats = torch.randn((len(coords), spec.channels), generator=_torch_generator(rng.randrange(1 << 31)), dtype=spec.dtype) * 0.5
    return SparseTensor(
        feats=feats,
        coords=coord_tensor,
        stride=spec.stride,
        spatial_range=(spec.batch_size, *spec.spatial_shape),
    )


def _contiguous_line_input(spec: SparseInputSpec, seed: int) -> SparseTensor:
    coords = torch.tensor(
        [(batch, x, 0, 0) for batch, count in enumerate(spec.rows_per_batch) for x in range(count)],
        dtype=torch.int32,
    )
    feats = torch.randn((coords.shape[0], spec.channels), generator=_torch_generator(seed), dtype=spec.dtype) * 0.5
    return SparseTensor(
        feats=feats,
        coords=coords,
        stride=spec.stride,
        spatial_range=(spec.batch_size, *spec.spatial_shape),
    )


def _target_from_input(input: SparseTensor, rng: random.Random) -> SparseTensor:
    rows = input.coords.cpu()
    count = rng.randint(1, max(1, rows.shape[0]))
    indices = sorted(rng.sample(range(rows.shape[0]), k=count))
    coords = rows[indices].contiguous()
    return SparseTensor(
        feats=torch.zeros((count, 1), dtype=input.feats.dtype),
        coords=coords,
        stride=input.stride,
        spatial_range=input.spatial_range,
    )


def _train_dense(model: nn.Module, x: SparseTensor, target: torch.Tensor, *, steps: int, device: torch.device) -> None:
    if steps <= 0:
        return
    model.train()
    model.to(device)
    x_dev = _sparse_to(x, device)
    target_dev = target.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x_dev), target_dev)
        loss.backward()
        optimizer.step()
    model.cpu()


def _train_sparse(model: nn.Module, *inputs: SparseTensor, steps: int, device: torch.device) -> None:
    if steps <= 0:
        return
    model.train()
    model.to(device)
    dev_inputs = tuple(_sparse_to(item, device) for item in inputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(*dev_inputs)
        loss = output.feats.float().square().mean()
        loss.backward()
        optimizer.step()
    model.cpu()


def _dense_output(model: nn.Module, x: SparseTensor, device: torch.device) -> torch.Tensor:
    model.to(device).eval()
    with torch.no_grad():
        out = model(_sparse_to(x, device)).detach().cpu()
    model.cpu()
    return out


def _sparse_output(model: nn.Module, *inputs: SparseTensor, device: torch.device) -> SparseTensor:
    model.to(device).eval()
    with torch.no_grad():
        out = model(*(_sparse_to(item, device) for item in inputs)).cpu()
    model.cpu()
    return out


def _generative_transpose_reference(model: SparseChainModel, tensor: SparseTensor) -> SparseTensor:
    conv = model.stages[0]
    activation = model.stages[1]
    if not isinstance(conv, spnn.GenerativeConvTranspose3d):
        raise TypeError('expected first stage to be GenerativeConvTranspose3d.')
    kernel_size = tuple(int(item) for item in conv.kernel_size)
    stride = tuple(int(item) for item in conv.stride)
    offsets = [
        (x, y, z)
        for x in range(kernel_size[0])
        for y in range(kernel_size[1])
        for z in range(kernel_size[2])
    ]
    rows: dict[tuple[int, int, int, int], torch.Tensor] = {}
    for coord, feat in zip(tensor.coords, tensor.feats, strict=True):
        for kernel_id, offset in enumerate(offsets):
            out_coord = (
                int(coord[0]),
                int(coord[1]) * stride[0] + offset[0],
                int(coord[2]) * stride[1] + offset[1],
                int(coord[3]) * stride[2] + offset[2],
            )
            value = feat @ conv.kernel[kernel_id]
            rows[out_coord] = rows.get(out_coord, torch.zeros_like(value)) + value
    coords = torch.tensor(sorted(rows), dtype=torch.int32)
    feats = torch.stack([rows[tuple(coord.tolist())] for coord in coords])
    if conv.bias is not None:
        feats = feats + conv.bias
    if isinstance(activation, spnn.Tanh):
        feats = torch.tanh(feats)
    else:
        raise TypeError('generative reference currently expects Tanh activation.')
    return SparseTensor(
        feats=feats.detach(),
        coords=coords,
        stride=tuple(int(tensor.stride[index]) // stride[index] for index in range(3)),
        spatial_range=tensor.spatial_range,
    )


def _quantized_reference_model(
    model: nn.Module,
    *,
    bits: int | None,
    group_size: int,
    scale_dtype: str,
) -> nn.Module:
    if bits is None:
        return model
    reference = copy.deepcopy(model).cpu().eval()
    for module in reference.modules():
        if isinstance(
            module,
            (
                spnn.Conv3d,
                spnn.SubmConv3d,
                spnn.ConvTranspose3d,
                spnn.GenerativeConvTranspose3d,
                spnn.TargetConv3d,
            ),
        ):
            dequantized = dequantize_artifact_weight(
                _conv_weight_to_mlx_like(module),
                bits=bits,
                group_size=group_size,
                scale_dtype=scale_dtype,
            )
            kernel_size = tuple(int(item) for item in module.kernel_size)
            kernel = (
                dequantized.permute(1, 2, 3, 4, 0)
                .reshape(
                    kernel_size[0] * kernel_size[1] * kernel_size[2],
                    module.in_channels,
                    module.out_channels,
                )
                .contiguous()
            )
            if module.kernel.ndim == 2:
                kernel = kernel.reshape(module.in_channels, module.out_channels)
            module.kernel.data.copy_(kernel.to(module.kernel.dtype))
        elif isinstance(module, nn.Linear):
            module.weight.data.copy_(
                dequantize_artifact_weight(
                    module.weight.detach(),
                    bits=bits,
                    group_size=group_size,
                    scale_dtype=scale_dtype,
                ).to(module.weight.dtype)
            )
    return reference


def _conv_weight_to_mlx_like(module: spnn.Conv3d) -> torch.Tensor:
    kernel_size = tuple(int(item) for item in module.kernel_size)
    weight = module.kernel.detach()
    if weight.ndim == 2:
        weight = weight.reshape(1, weight.shape[0], weight.shape[1])
    return (
        weight.reshape(*kernel_size, module.in_channels, module.out_channels)
        .permute(4, 0, 1, 2, 3)
        .contiguous()
    )


def _sparse_to(tensor: SparseTensor, device: torch.device) -> SparseTensor:
    return SparseTensor(
        feats=tensor.feats.to(device),
        coords=tensor.coords.to(device),
        stride=tensor.stride,
        spatial_range=tensor.spatial_range,
    )


def _save_sparse_inputs(
    case_dir: Path,
    prefix: str,
    tensor: SparseTensor,
    *,
    extra: dict[str, SparseTensor] | None = None,
) -> None:
    values = {
        f'{prefix}coords': tensor.coords.cpu(),
        f'{prefix}features': tensor.feats.cpu(),
        f'{prefix}active': _active_rows(tensor),
    }
    for name, sparse in (extra or {}).items():
        values[f'{name}_coords'] = sparse.coords.cpu()
        values[f'{name}_features'] = sparse.feats.cpu()
        values[f'{name}_active'] = _active_rows(sparse)
    save_file(values, case_dir / 'inputs.safetensors')


def _save_sparse_expected(case_dir: Path, expected: SparseTensor) -> None:
    save_file(
        {
            'output.coords': expected.coords.cpu(),
            'output.features': expected.feats.cpu(),
            'output.active': _active_rows(expected),
        },
        case_dir / 'expected.safetensors',
    )


def _active_rows(tensor: SparseTensor) -> torch.Tensor:
    return torch.tensor([tensor.feats.shape[0]], dtype=torch.int32)


def _spec_metadata(spec: SparseInputSpec) -> dict[str, Any]:
    return {
        'channels': spec.channels,
        'batch_size': spec.batch_size,
        'spatial_shape': list(spec.spatial_shape),
        'rows_per_batch': list(spec.rows_per_batch),
        'stride': list(spec.stride),
    }


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    return generator


def _archive(root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(root, arcname=root.name)


if __name__ == '__main__':
    main()
