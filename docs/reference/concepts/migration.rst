Migrating from TorchSparse
==========================

Torch Lattice started from MIT HAN Lab's TorchSparse codebase, but the public
semantics are intentionally tightened for artifact export and MLX replay.

What remains familiar
---------------------

The core authoring shape remains close to TorchSparse:

* sparse tensors carry coordinates and features;
* modules follow ``torch.nn.Module`` conventions;
* CUDA kernels provide sparse convolution, pooling, hashing, voxelization, and
  coordinate-query utilities;
* models can be trained with normal PyTorch optimizers and losses.

What is stricter
----------------

The main difference is sparse support semantics. Torch Lattice does not ask users
or exporters to infer intent from a legacy ``Conv3d`` call. Use explicit module
classes for explicit support behavior.

Migration checks
----------------

The ``migration`` CLI compares covered model fragments against a kept original
TorchSparse worktree. Use it when changing core convolution, pooling, or
coordinate behavior:

.. code-block:: bash

   uv run migration all --device cuda

A migration pass is not a replacement for model-level validation, but it catches
semantic drift in the compatibility subset that matters most for ported models.
