Compatibility notes
===================

Torch Lattice compatibility is about three layers: original TorchSparse migration,
Torch/CUDA runtime compatibility, and MLX artifact compatibility.

Original TorchSparse
--------------------

The project is a fork lineage, not a drop-in promise. Covered migration behavior
is tested through the migration CLI. Code that depends on implicit stride-1
convolution semantics should be rewritten to explicit ``SubmConv3d`` or
``Conv3d`` according to the intended support behavior.

Torch/CUDA
----------

The package currently targets a modern PyTorch/CUDA stack. Keep PyTorch, CUDA
runtime, NVCC, and driver versions aligned. Mismatched CUDA installations are the
most common source of build failures.

MLX artifacts
-------------

Artifact compatibility is the deployment contract. If a model exports and replays
through conformance with acceptable numerical error, it is compatible at the level
that matters for MLX deployment.
