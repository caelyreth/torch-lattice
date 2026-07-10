Artifact contract
=================

A lattice artifact is the exchange format between the CUDA training side and the
MLX deployment side. It is designed as a model artifact, not as a Python runtime
snapshot.

Bundle layout
-------------

A normal export writes:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Purpose
   * - ``graph.mlir``
     - Stable sparse graph structure and operation attributes.
   * - ``weights.safetensors``
     - Named tensor values referenced by the graph.
   * - ``graph.mlir`` module attributes
     - Input/output names, dialect version, schema digest, and weight-file
       identity. These are embedded in the graph rather than a sidecar file.

The MLIR graph owns operation semantics. The safetensors file owns numeric state.
This separation lets CUDA training and MLX inference share a contract without
sharing a Python object model.

Exporter model
--------------

The exporter uses ``torch.fx`` to observe graph topology. It records sparse ops,
standard tensor ops that are part of supported sparse modules, graph inputs,
constants, and named weights. A supported export should be explicit enough that a
reader can answer:

* which sparse support each operation consumes;
* which operation creates or preserves support;
* which named weight tensor is used by each parameterized operation;
* which graph values are public outputs.

Export requires ``model.eval()`` and a non-empty ``example_inputs`` tuple. The
model's ``forward`` signature supplies stable ABI names, while each example value
supplies dtype, rank, sparse stride, and channel metadata. Multiple positional
inputs and nested tuple/list outputs are flattened into named MLIR arguments and
returns. Use ``output_names=...`` when the default ``output_0``, ``output_1``
names are not descriptive enough.

Numerical contract
------------------

The artifact contract targets mathematical equivalence, not bitwise identity
between CUDA and Metal. Different kernel launch orders and accumulation paths can
produce small floating-point differences. Conformance reports should therefore
track absolute and relative error distributions rather than a single exact-match
boolean.
