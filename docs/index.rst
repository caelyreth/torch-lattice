torch-lattice
=============

``torch-lattice`` is the Torch/CUDA training-side companion to
`mlx-lattice <https://github.com/caelyreth/mlx-lattice>`_. It keeps the
sparse training and export workflow in PyTorch while producing MLIR artifact
bundles that the MLX runtime can replay on Apple Silicon.

The library intentionally separates three concerns:

* CUDA sparse operators and ``torch.nn`` modules for training and validation;
* a small sparse tensor contract shared with the MLX side;
* artifact tooling that lowers explicit Torch graphs to a portable lattice IR.

The primary data model matches MLX Lattice:

* coordinates are integer rows ordered as ``(batch, x, y, z)``;
* features are dense Torch tensors whose rows align one-to-one with coordinates;
* sparse operators either preserve support, generate new support, or consume a
  caller-provided target support;
* exported artifacts carry graph structure and named weights separately so the
  deployment runtime can rebuild the same sparse computation.

Navigation map
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Task
     - Read first
     - API reference
   * - Install and verify CUDA
     - :doc:`getting-started/installation`
     - :doc:`reference/tooling/cuda-ci`
   * - Build a sparse tensor
     - :doc:`reference/concepts/sparse-tensor`
     - :doc:`api/core/index`
   * - Write a sparse model
     - :doc:`getting-started/quickstart`
     - :doc:`api/nn/index`
   * - Export an artifact
     - :doc:`reference/concepts/artifacts`
     - :doc:`api/artifact/index`
   * - Reason about convolution
     - :doc:`reference/concepts/convolution-semantics`
     - :doc:`api/nn/index`
   * - Migrate TorchSparse code
     - :doc:`reference/concepts/migration`
     - :doc:`project/compatibility`
   * - Run CUDA benchmarks
     - :doc:`reference/tooling/benchmarks`
     - :doc:`api/tooling/index`
   * - Generate/replay conformance cases
     - :doc:`reference/tooling/conformance`
     - :doc:`api/tooling/index`

.. note::

   ``torch-lattice`` is the CUDA-side project. It is designed for model authoring,
   training, migration checks, artifact export, and CUDA benchmark/conformance
   generation. MLX/Metal inference is handled by ``mlx-lattice`` after loading the
   exported artifact bundle.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting-started/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/concepts/index
   reference/tooling/index

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project notes

   project/stability
   project/compatibility
   project/troubleshooting
