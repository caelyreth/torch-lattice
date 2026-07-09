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
   * - metadata
     - Input/output names, artifact version, and loader-facing bookkeeping.

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

Numerical contract
------------------

The artifact contract targets mathematical equivalence, not bitwise identity
between CUDA and Metal. Different kernel launch orders and accumulation paths can
produce small floating-point differences. Conformance reports should therefore
track absolute and relative error distributions rather than a single exact-match
boolean.
