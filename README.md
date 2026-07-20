## Torch Lattice

`torch-lattice` is a sparse convolution library for PyTorch and CUDA. Use it to
build and train sparse point-cloud or voxel models on NVIDIA GPUs, then export
the supported model graph and weights as a portable artifact for
[`mlx-lattice`](https://github.com/caelyreth/mlx-lattice) on Apple Silicon.

It builds on MIT HAN Lab's TorchSparse work while making sparse support
semantics and artifact export explicit. This is a maintained fork, not a
drop-in package rename for every historical TorchSparse model.

[Documentation](https://torch-lattice.iki.moe)
| [PyPI](https://pypi.org/project/torch-lattice/)
| [MLX Lattice](https://github.com/caelyreth/mlx-lattice)
| [Acknowledgements](#acknowledgements)

### Install

The published wheel supports Linux `x86_64`, Python 3.14, and the PyTorch CUDA
12.8 wheel stack. You need a compatible NVIDIA driver at runtime; `nvcc` and a
local CUDA toolkit are only needed to build the extension from source.

Add it to a `uv` project:

```bash
uv add torch-lattice --torch-backend cu128
```

Or install it into an existing environment:

```bash
uv pip install --torch-backend cu128 torch-lattice
```

### Build a sparse model

Sparse coordinates use `(batch, x, y, z)` integer rows. Each feature row belongs
to the coordinate row at the same index.

```python
import torch
import torch_lattice as tl
import torch_lattice.nn as spnn

coords = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 2, 1, 0],
    ],
    dtype=torch.int,
    device="cuda",
)
features = torch.randn(coords.shape[0], 8, device="cuda")
x = tl.SparseTensor(coords=coords, feats=features, stride=1)

model = torch.nn.Sequential(
    spnn.SubmConv3d(8, 16, kernel_size=3),
    spnn.BatchNorm(16),
    spnn.ReLU(inplace=True),
    spnn.Conv3d(16, 32, kernel_size=2, stride=2),
    spnn.GlobalAvgPool(),
    torch.nn.Linear(32, 4),
).cuda()

logits = model(x)
```

Choose the convolution class for the support behavior you intend:

- `SubmConv3d` keeps the input active coordinates and updates their features.
- `Conv3d` creates a forward sparse output support. Pass
  `coordinates=target` when the output support is owned by the caller.
- `ConvTranspose3d` uses a transposed sparse relation; use
  `GenerativeConvTranspose3d` when the operation must generate fine support.

This distinction matters when porting a TorchSparse model. In particular, a
legacy spatial `Conv3d(kernel_size > 1, stride = 1)` used for in-place feature
refinement normally becomes `SubmConv3d`. The
[migration guide](https://torch-lattice.iki.moe/reference/concepts/migration.html)
explains the full mapping and checkpoint conversion process.

### Export for MLX

Torch Lattice exports an eval-mode model as a portable directory containing
`graph.mlir` and `weights.safetensors`. The artifact describes the model graph;
it is not a serialized Python module. `mlx-lattice` loads the same directory for
MLX/Metal inference.

```python
from torch_lattice.artifact import save_lattice_model_artifact

artifact = save_lattice_model_artifact(
    model.eval(),
    "artifacts/example-model",
    example_inputs=(x,),
    output_names=("logits",),
)

print(artifact.graph_path)
print(artifact.weights_path)
```

The artifact contract is shared through `lattice-contract`. It records the
public input/output ABI, sparse support semantics, weight layout, and dialect
schema. Keep a known input/output fixture with every production export, then
replay it on MLX before deployment.

### Documentation

The complete guide is available at
[torch-lattice.iki.moe](https://torch-lattice.iki.moe):

- [Installation](https://torch-lattice.iki.moe/getting-started/installation.html)
- [Quickstart](https://torch-lattice.iki.moe/getting-started/quickstart.html)
- [Training-to-MLX workflow](https://torch-lattice.iki.moe/getting-started/workflow.html)
- [Migration from TorchSparse and MinkowskiEngine](https://torch-lattice.iki.moe/reference/concepts/migration.html)
- [Artifact contract](https://torch-lattice.iki.moe/reference/concepts/artifacts.html)
- [API reference](https://torch-lattice.iki.moe/api/)

### Tooling

The workspace keeps performance measurement and correctness checks separate.
After `uv sync --all-packages`, run benchmarks when investigating throughput and
use conformance tools when checking a model or artifact boundary:

```bash
uv run bench --preset smoke
uv run fuzz --cases 32 --device cuda --archive /tmp/torch_lattice_fuzz.tar.gz
uv run migration all --device cuda
```

Replay a CUDA-generated fuzz archive on the MLX side:

```bash
uv run conformance replay /tmp/torch_lattice_fuzz.tar.gz \
  --report /tmp/torch_lattice_fuzz_report.json
```

See the [tooling reference](https://torch-lattice.iki.moe/reference/tooling/)
for fixed fixtures, checkpoint conversion, and focused benchmark options.

### Development

Developing the native extension requires a Linux CUDA environment with Python
3.14, `uv >= 0.11.25`, PyTorch `2.11.0+cu128`, and a compatible CUDA toolkit.

```bash
export CUDA_PATH=/usr/local/cuda-12.8
uv sync --all-packages --extra test
uv run --all-packages --extra test pytest tests -q
```

Build documentation without compiling CUDA code:

```bash
uv sync --group docs --no-install-workspace
uv run --no-sync sphinx-build -W -b html docs docs/_build/html
```

### Acknowledgements

`torch-lattice` is based on MIT HAN Lab's original
[TorchSparse](https://github.com/mit-han-lab/torchsparse) project. It is
developed with [`mlx-lattice`](https://github.com/caelyreth/mlx-lattice), the
MLX/Metal runtime for the shared artifact contract.

### License

Open sourced under the [MIT license](./LICENSE).
