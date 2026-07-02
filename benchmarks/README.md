# TorchSparse Sparse Operator Benchmarks

This suite benchmarks TorchSparse hot-path sparse tensor operations on synthetic
600K-point CUDA inputs with controlled sparsity patterns:

- `isolated`: no local spatial locality
- `line`: one-dimensional contiguous structure
- `plane`: two-dimensional contiguous structure
- `block2`, `block3`, `block5`, `block8`: repeated dense local blocks
- `grid`: contiguous 3D lattice

Default full run:

```bash
uv run python benchmarks/sparse_ops_benchmark.py --output benchmark.json
```

If the local `uv` version is older than the project requirement, run the same
script through the environment Python:

```bash
.venv/bin/python benchmarks/sparse_ops_benchmark.py --output benchmark.json
```

Quick smoke run:

```bash
uv run python benchmarks/sparse_ops_benchmark.py --smoke
```

The benchmark uses CUDA events, synchronizes every measurement, reports mean,
median, p90, min/max latency, peak allocated memory, input points, channels, and
output points where applicable.  Operator groups can be selected with
`--groups tensor hash dense kmap conv train`.

The default suite covers:

- Sparse tensor construction, device conversion, dtype conversion, feature cat,
  generative add, global pooling, crop, activations, batch norm, and group norm.
- Hash/query/count kernels: `sphash`, 27-offset `sphash`, `sphashquery`, and
  `spcount`.
- Dense/voxel paths: `to_dense`, `spvoxelize`, `spdevoxelize`, and trilinear
  interpolation weight calculation.
- Kernel-map helpers: `spdownsample` and `spupsample_generative`.
- Convolution hot paths: 1x1 matmul, 3x3 implicit GEMM unsorted, 3x3 implicit
  GEMM sorted, 3x3 fetch-on-demand fused/no-fusion, 3x3 gather-scatter, 2x2
  stride-2 implicit GEMM, and a two-layer `stride2 -> subm3` chain that checks
  hashmap reuse across common backbone stages.
- Training hot path: 3x3 unsorted/sorted implicit GEMM forward plus backward
  and cached Fetch-on-Demand fallback backward.

Convolution results include both `_cold` and `_warm` entries.  `_cold` creates a
fresh `SparseTensor` each measured iteration, so it includes kernel-map build
cost.  `_warm` reuses the same `SparseTensor` after one priming call, so it
reflects steady-state convolution with cached maps.

Some dense operations are intentionally skipped when the dense output would be
pathologically large for the sparse shape.  The skip is recorded as a result row
with `skipped=true`; adjust the guard with `--max-dense-elements`.
