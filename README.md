## Torch Lattice

`torch-lattice` is the Torch/CUDA training-side companion to
[`mlx-lattice`](https://github.com/caelyreth/mlx-lattice). It keeps the sparse
model authoring and CUDA provenance workflow on the Torch side, then exports
portable lattice MLIR artifacts for MLX/Metal deployment.

`torch-lattice` is a project-owned fork of MIT HAN Lab's TorchSparse. The public
semantics are aligned to `mlx-lattice` and the lattice MLIR contract rather than
to historical TorchSparse API quirks.

[MLX Lattice](https://github.com/caelyreth/mlx-lattice) | [Acknowledgements](#acknowledgements)

### Install

`torch-lattice` currently targets Python 3.14, PyTorch CUDA 12.8 wheels, and a
CUDA 12.8 build environment.

For development from a checkout:

```bash
uv sync --all-packages --extra test
```

The repository also provides a CUDA Linux GitHub workflow that builds and smoke
checks the native CUDA wheel on an Ubuntu runner.

### Relationship to MLX Lattice

The two packages are intentionally split by runtime role:

- `torch-lattice` is the CUDA training and artifact-production side.
- `mlx-lattice` is the Apple Silicon inference and deployment side.
- `lattice-contract` defines the shared artifact constants and MLIR contract
  metadata used by both sides.

Portable artifacts use `graph.mlir` plus `weights.safetensors`. Torch-side
exporters write those files; MLX-side artifact loading compiles them into an
executable MLX program.

### Convolution semantics

Convolution classes are explicit:

- `torch_lattice.nn.Conv3d` is forward support-generating sparse convolution and
  exports to `lattice.conv3d`, including `stride=1`.
- `torch_lattice.nn.SubmConv3d` is support-preserving submanifold convolution and
  exports to `lattice.subm_conv3d`.
- `torch_lattice.nn.ConvTranspose3d` exports to `lattice.conv_transpose3d`.
- `torch_lattice.nn.GenerativeConvTranspose3d` exports to
  `lattice.generative_conv_transpose3d`.

Artifact builders lower module identity directly. They do not infer submanifold
semantics from stride, padding, or legacy indice-key conventions.

### Tooling

After `uv sync --all-packages`, use the workspace scripts from the repository
root:

```bash
uv run bench --preset smoke
uv run fuzz --cases 32 --device cuda --archive /tmp/torch_lattice_fuzz.tar.gz
uv run conformance fuzz --cases 32 --device cuda
uv run migration all --device cuda
```

The corresponding MLX-side replay command is:

```bash
uv run conformance replay /tmp/torch_lattice_fuzz.tar.gz \
  --report /tmp/torch_lattice_fuzz_report.json
```

### Migration compatibility checks

Original TorchSparse and `torch-lattice` are not assumed to have identical class
semantics. The supported migration rule is explicit:

- original `torchsparse.nn.Conv3d(kernel_size > 1, stride = 1)` maps to
  `torch_lattice.nn.SubmConv3d`;
- original pointwise `Conv3d(kernel_size = 1)` maps to `torch_lattice.nn.Conv3d`;
- original strided forward convolutions map to `torch_lattice.nn.Conv3d` with the
  same stride.

The `migration` CLI verifies the covered subset against a kept original
TorchSparse package/worktree in separate subprocesses.

### Development

Common local checks:

```bash
uv run --all-packages --extra test pytest tests -q
uv run bench --list
```

Build CUDA Linux distributions locally with:

```bash
export CUDA_PATH=/usr/local/cuda-12.8
uv build \
  --sdist \
  --wheel \
  --config-setting=cmake.define.CMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
  --config-setting=cmake.define.CUDAToolkit_ROOT="$CUDA_PATH"
```

### Acknowledgements

`torch-lattice` is based on MIT HAN Lab's original
[TorchSparse](https://github.com/mit-han-lab/torchsparse) project.

It is developed together with
[`mlx-lattice`](https://github.com/caelyreth/mlx-lattice), which provides the
MLX/Metal deployment runtime for the same artifact contract.

### License

Open sourced under the [MIT license](./LICENSE).
