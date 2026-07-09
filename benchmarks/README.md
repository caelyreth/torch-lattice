# Torch Lattice CUDA Benchmarks

The benchmark suite is a workspace package with the same shape as the
`mlx-lattice` benchmark runner: a small CLI, colorized progress, reusable case
catalogs, structured JSON reports, and operation groups that can be swept across
input amount, density layout, channel count, dtype, and execution mode.

## CLI

Default smoke run:

```bash
uv run lattice-bench
```

Focused convolution smoke with explicit amount/density/channel choices:

```bash
uv run lattice-bench \
  --preset smoke \
  --group conv \
  --size 8192 \
  --channels 32 \
  --layout block2 \
  --layout grid \
  --warmup 5 \
  --repeats 20 \
  --output conv-smoke.json
```

Training/backward cases are explicit:

```bash
uv run lattice-bench \
  --group train \
  --mode backward \
  --size 8192 \
  --channels 32
```

Useful knobs:

- `--preset smoke|standard|full` selects the default matrix size.
- `--group tensor|hash|dense|kmap|conv|nn|train` can be repeated.
- `--mode cold_op|hot_op|backward` can be repeated.
- `--size N` can be repeated to sweep input amount.
- `--channels C` can be repeated to sweep feature width.
- `--layout isolated|line|plane|grid|block2|block3|block4|block8` can be repeated
  to sweep spatial density/locality.
- `--dtype fp16|fp32` selects sparse feature and module dtype.
- `--color auto|always|never` controls progress coloring.
- `--case-filter text` selects matching case names.
- `--fail-fast` stops on the first case error; otherwise failed/unsupported
  case-shape combinations are recorded as skipped result rows.

Relative `--output` paths are written under `benchmarks/results`. Each run also
writes a `.summary.txt` next to the JSON report.

## Groups

- `tensor`: SparseTensor construction/device/dtype paths, feature concat,
  generative add, global pooling, crop, activations, BatchNorm, and GroupNorm.
- `hash`: `sphash`, 27-offset kernel hash, self query, and count kernels.
- `dense`: dense materialization, voxelize, devoxelize, and trilinear weight
  calculation.
- `kmap`: downsample/upsample helpers and kernel-map construction across
  implicit GEMM, Fetch-on-Demand, and Gather-Scatter dataflows.
- `conv`: pointwise, 3x3, stride-2, Fetch-on-Demand, Gather-Scatter, and explicit
  submanifold convolution module paths.
- `nn`: composed sparse modules including classifier-style, residual add/cat,
  and activation-chain paths.
- `train`: forward/backward convolution paths.

## Density layouts

The synthetic coordinate generator mirrors the MLX benchmark suite:

- `isolated`: low neighborhood overlap.
- `line`: one-dimensional contiguous structure.
- `plane`: two-dimensional contiguous structure.
- `grid`: contiguous 3D lattice.
- `block2`, `block3`, `block4`, `block8`: repeated dense local blocks.

## Output schema

JSON reports contain:

- `environment`: git SHA, Python/Torch/Torch-Lattice versions, CUDA device,
  backend flags, and hash reserve ratio.
- `results`: one row per case/parameter/mode with params, samples, median/min/p90
  /p95 latency, workload metrics, throughput units, skipped status, and notes.

This makes CUDA-side reports directly comparable with MLX-side reports at the
case/mode/parameter level while preserving Torch-specific CUDA backend details.
