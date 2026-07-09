## Torch Lattice

Torch Lattice is a CUDA training-side sparse point-cloud library and artifact
producer for the lattice MLIR contract. It is a project-owned fork of
TorchSparse, but the public semantics are aligned to `mlx-lattice` deployment
rather than to historical TorchSparse API quirks.

### Tooling boundary

The repository keeps generated-artifact checks and benchmarks in workspace
packages instead of root scripts. After `uv sync --all-packages`, use the
workspace scripts from the repository root:

- `e2e-fixtures` writes fixed, small regression fixtures.
- `fuzz` generates randomized CUDA provenance archives for MLX
  replay.
- `migration` compares the supported original TorchSparse
  migration subset.
- `bench` measures CUDA performance with the same synthetic data
  families used by MLX-side benchmarking.

### Convolution semantics

Convolution classes are explicit:

- `torch_lattice.nn.Conv3d` is forward support-generating sparse convolution
  and exports to `lattice.conv3d`, including `stride=1`.
- `torch_lattice.nn.SubmConv3d` is support-preserving submanifold convolution
  and exports to `lattice.subm_conv3d`.
- `torch_lattice.nn.ConvTranspose3d` exports to `lattice.conv_transpose3d`.
- `torch_lattice.nn.GenerativeConvTranspose3d` exports to
  `lattice.generative_conv_transpose3d`.

Artifact builders lower module identity directly. They do not infer submanifold
semantics from stride, padding, or legacy indice-key conventions.

Credit: this project is based on MIT Han Lab's original
[TorchSparse](https://github.com/mit-han-lab/torchsparse) project.

### Migration compatibility checks

Original TorchSparse and `torch-lattice` are not assumed to have identical class
semantics. The supported migration rule is explicit:

- original `torchsparse.nn.Conv3d(kernel_size > 1, stride = 1)` maps to
  `torch_lattice.nn.SubmConv3d`;
- original pointwise `Conv3d(kernel_size = 1)` maps to `torch_lattice.nn.Conv3d`;
- original strided forward convolutions map to `torch_lattice.nn.Conv3d` with the
  same stride;
- branch, cat, global pool, batch norm, and pointwise feature chains must match
  exactly for the covered inference subset.

Use the permanent compatibility CLI to verify the mapped subset against the kept
original TorchSparse worktree/package. It runs both native packages in separate
subprocesses because their extensions register overlapping native type names:

```bash
uv run migration all \
  --cases 70 \
  --seed 20260709 \
  --device cuda \
  --output /tmp/torch_lattice_torchsparse_compat
```

### CUDA-to-MLX artifact conformance

Fuzz fixtures are CUDA provenance data for the MLX artifact runtime. Each case
contains `graph.mlir`, `weights.safetensors`, exact inputs, expected outputs,
and tolerances. Quantized fixture expected outputs are generated from the
artifact-packed/dequantized weight contract, not from the pre-quantized dense
training weights.

Generate a self-contained archive for MLX-side replay with:

```bash
uv run fuzz \
  --cases 32 \
  --seed 20260709 \
  --train-steps 4 \
  --families all \
  --device cuda \
  --output /tmp/torch_lattice_fuzz \
  --archive /tmp/torch_lattice_fuzz.tar.gz
```

The corresponding MLX-side command is:

```bash
uv run conformance replay \
  /tmp/torch_lattice_fuzz.tar.gz \
  --report /tmp/torch_lattice_fuzz_report.json
```
