Workflow
========

The intended end-to-end workflow has a CUDA authoring side and an MLX deployment
side.

.. code-block:: text

   Torch/CUDA training
        |
        |  export artifact bundle
        v
   graph.mlir + weights.safetensors
        |
        |  copy or publish as a model artifact
        v
   MLX/Metal inference with mlx-lattice

Authoring phase
---------------

Use Torch modules while training, validating, and running CUDA-side experiments.
Keep sparse support behavior explicit in the model architecture: choose
``SubmConv3d`` when support must be preserved and ``Conv3d`` when support should
be generated. To compute only on caller-owned support, call the same ``Conv3d``
module with ``coordinates=target``.

Export phase
------------

The exporter traces the model with ``torch.fx`` and writes:

* ``graph.mlir`` for sparse graph structure;
* ``weights.safetensors`` for tensor payloads;

Input/output names, dialect version, schema digest, and weight-file identity are
module attributes inside ``graph.mlir``; there is no sidecar metadata file.

The graph representation is intentionally not a Python pickle. It is a stable
exchange boundary between the CUDA training project and the MLX deployment
project.

Validation phase
----------------

Use the conformance tools for two different checks:

* fixed E2E fixtures verify known graph shapes and deterministic model outputs;
* fuzz fixtures generate many random sparse graphs and compare CUDA output with
  MLX replay output.

Use benchmarks for performance; use conformance for correctness. They share the
same synthetic coordinate families, but they answer different questions.
