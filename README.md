## Torch Lattice

Torch Lattice is a CUDA training-side sparse point-cloud library and artifact
producer for the lattice MLIR contract. It is a project-owned fork of
TorchSparse, but the public semantics are aligned to `mlx-lattice` deployment
rather than to historical TorchSparse API quirks.

### Convolution semantics

Convolution classes are explicit:

- `torch_lattice.nn.Conv3d` is forward support-generating sparse convolution
  and exports to `lattice.conv3d`, including `stride=1`.
- `torch_lattice.nn.SubmConv3d` is support-preserving submanifold convolution
  and exports to `lattice.subm_conv3d`.
- `torch_lattice.nn.ConvTranspose3d` exports to `lattice.conv_transpose3d`.
- `torch_lattice.nn.GenerativeConvTranspose3d` exports to
  `lattice.generative_conv_transpose3d`.

Exporters lower module identity directly. They do not infer submanifold
semantics from stride, padding, or legacy indice-key conventions.

Credit: this project is based on MIT Han Lab's original
[TorchSparse](https://github.com/mit-han-lab/torchsparse) project.
