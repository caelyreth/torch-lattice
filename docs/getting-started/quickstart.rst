Quickstart
==========

A sparse tensor is a pair of aligned tensors: coordinates and features. The
coordinate convention is always ``(batch, x, y, z)``.

.. code-block:: python

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
       device='cuda',
   )
   feats = torch.randn(coords.shape[0], 8, device='cuda')

   x = tl.SparseTensor(coords=coords, feats=feats, stride=1)

   model = torch.nn.Sequential(
       spnn.SubmConv3d(8, 16, kernel_size=3),
       spnn.BatchNorm(16),
       spnn.ReLU(inplace=True),
       spnn.Conv3d(16, 32, kernel_size=2, stride=2),
       spnn.GlobalAvgPool(),
       torch.nn.Linear(32, 4),
   ).cuda()

   y = model(x)

Support semantics
-----------------

The convolution class name is part of the semantic contract:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Module
     - Support behavior
     - Typical use
   * - ``SubmConv3d``
     - Preserve the input coordinate support.
     - Feature refinement inside the same sparse support.
   * - ``Conv3d``
     - Generate output coordinates according to kernel and stride.
     - Downsampling or support-expanding forward convolution.
   * - ``Conv3d(...)(x, coordinates=target)``
     - Produce values only at caller-provided target coordinates.
     - Cross-support or detector/head style projections without a second
       convolution class.
   * - ``ConvTranspose3d``
     - Transposed sparse relation.
     - Upsampling.
   * - ``GenerativeConvTranspose3d``
     - Generative transposed support.
     - Decoder-style support generation.

Exporting an artifact
---------------------

Artifact export lowers a Torch graph and its state dict to a portable MLIR bundle.
The bundle can then be copied to a Mac and loaded by ``mlx-lattice``.

.. code-block:: python

   from torch_lattice.artifact import save_lattice_model_artifact

   result = save_lattice_model_artifact(
       model.eval(),
       'artifacts/example-model',
       example_inputs=(x,),
       output_names=('logits',),
   )
   print(result.graph_path)
   print(result.weights_path)

For production export, prefer explicit modules and stable module names. They make
artifact diffs, replay failures, and state-dict review much easier to diagnose.
Export is eval-only and ``example_inputs`` is required: the values define input
names, sparse strides, channel counts, dtypes, and dense tensor shapes in the
public artifact ABI. The exporter writes exactly ``graph.mlir`` and
``weights.safetensors``.
