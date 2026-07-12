Migrating from TorchSparse
==========================

Torch Lattice started from MIT HAN Lab's TorchSparse codebase, but it is not a
pure drop-in rename. The fork keeps the useful CUDA sparse operator foundation
while tightening semantics for two goals: reliable Torch/CUDA training and stable
artifact export for MLX/Metal replay.

The practical rule is simple: migrate model intent, not just class names. Original
TorchSparse often encoded intent through historical defaults. Torch Lattice makes
that intent explicit in module names and artifact operations.

What remains familiar
---------------------

The authoring model is still close to TorchSparse:

* sparse tensors pair coordinates with feature rows;
* modules are normal ``torch.nn.Module`` objects;
* features are dense Torch tensors with shape ``(N, C)``;
* coordinates use integer sparse rows and batch identity is part of each row;
* CUDA kernels cover sparse convolution, pooling, hashing, voxelization,
  devoxelization, coordinate query, and dense materialization;
* models train with standard PyTorch optimizers, losses, autocast policies, and
  state dicts.

What is intentionally different
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Area
     - Original TorchSparse behavior
     - Torch Lattice behavior
   * - Package identity
     - Imported as ``torchsparse``.
     - Imported as ``torch_lattice``. This avoids pretending that tightened
       semantics are the same library surface.
   * - Deployment target
     - CUDA runtime is the primary endpoint.
     - CUDA is the training/export endpoint; MLX Lattice is the deployment
       endpoint for exported artifacts.
   * - Convolution intent
     - Some stride-1 spatial ``Conv3d`` usage historically behaved like the
       intended submanifold path by convention.
     - Support behavior is explicit: ``SubmConv3d`` preserves support and
       ``Conv3d`` generates support or consumes an explicit
       ``coordinates=target`` support.
   * - Artifact export
     - No stable MLX artifact contract.
     - Exports ``graph.mlir`` plus ``weights.safetensors`` through the lattice
       artifact layer.
   * - Graph capture
     - Model execution is normal PyTorch; export was not a first-class contract.
     - Export uses explicit FX lowering and records graph topology, parameters,
       and sparse op semantics.
   * - Tooling
     - Examples and scripts were mostly TorchSparse-specific.
     - Benchmarks, fuzz fixtures, E2E fixtures, and migration checks are
       workspace scripts with comparable MLX-side outputs.

Convolution mapping
-------------------

The most important migration step is choosing the correct sparse support
semantics. Do not mechanically replace every old ``torchsparse.nn.Conv3d`` with
``torch_lattice.nn.Conv3d``.

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Original TorchSparse usage
     - Torch Lattice replacement
     - Reason
   * - ``Conv3d(kernel_size > 1, stride=1)`` used as an in-place sparse feature
       refinement layer
     - ``SubmConv3d``
     - Preserve the input coordinate support.
   * - ``Conv3d(kernel_size=1)``
     - ``Conv3d``
     - Pointwise convolution does not expand spatial support.
   * - ``Conv3d(stride > 1)``
     - ``Conv3d`` with the same stride
     - Strided forward convolution intentionally creates a downsampled support.
   * - Transposed sparse convolution
     - ``ConvTranspose3d`` or ``GenerativeConvTranspose3d``
     - Pick based on whether the operation consumes an existing relation or
       generates output support.
   * - Minkowski transpose convolution called with a coordinate map key
     - ``ConvTranspose3d(...)(x, target)`` or its normalized/generative variant
     - Evaluate the indexed transpose relation on exact caller-owned support.
   * - Convolution at known output coordinates
     - ``Conv3d(...)(x, coordinates=target)``
     - The target support is part of the caller's graph state, while parameter
       ownership remains in the ordinary convolution module.
   * - ``MinkowskiPoolingTranspose(..., expand_coordinates=True)``
     - ``PoolTranspose3d(...)(x)``
     - Generate and deduplicate fine support from the transposed kernel relation.
   * - ``MinkowskiPoolingTranspose(...)(x, coordinates=target)``
     - ``PoolTranspose3d(...)(x, target)``
     - Preserve target support and average all coarse contributors per row.
   * - Pruning followed by kernel-1 sorting onto a known coordinate map
     - ``torch_lattice.reindex_sparse(source, target)``
     - Express the intended exact target support directly; source-only rows
       are discarded and missing target rows use the declared fill value.
   * - ``MinkowskiPruning()(x, mask)``
     - ``torch_lattice.prune_mask(x, mask)``
     - Keep selected rows in their existing order; use ``prune`` for explicit
       row indices.
   * - ``TrilinearUpsampler(in_channels=C, out_channels=C)``
     - ``TrilinearUpsample3d(stride=2)``
     - Use normalized separable interpolation on generated or target support.
   * - ``ME.SparseTensor(..., quantization_mode=UNWEIGHTED_AVERAGE)``
     - ``tl.sparse_from_coordinates(..., duplicate_reduction='mean')``
     - Average exact duplicate input rows while retaining first-occurrence
       coordinate order.

A useful check while porting is to compare coordinate counts before and after a
layer. If the original layer was intended to keep exactly the same coordinate set,
it should normally become ``SubmConv3d`` in Torch Lattice.

Sparse tensor expectations
--------------------------

Torch Lattice documents the coordinate row convention as ``(batch, x, y, z)``.
This convention matters because crop, BEV conversion, hashing, kernel-map
construction, and artifact replay all assume that the first column is batch and
the remaining columns are spatial axes.

.. code-block:: python

   import torch
   import torch_lattice as tl

   coords = torch.tensor(
       [[0, 4, 8, 2], [0, 5, 8, 2]],
       dtype=torch.int,
       device='cuda',
   )
   feats = torch.randn(coords.shape[0], 16, device='cuda')
   x = tl.SparseTensor(coords=coords, feats=feats, stride=1)

When migrating data pipelines, verify these invariants first:

* coordinate dtype is integer and coordinate shape is ``(N, 4)``;
* feature shape is ``(N, C)``;
* coordinate row ``i`` owns feature row ``i``;
* batch IDs are not stored separately from coordinates;
* transforms that crop or reshape sparse data operate on spatial columns, not the
  batch column.

Import and module namespace mapping
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Original pattern
     - Torch Lattice pattern
   * - ``import torchsparse``
     - ``import torch_lattice as tl``
   * - ``import torchsparse.nn as spnn``
     - ``import torch_lattice.nn as spnn``
   * - ``import torchsparse.nn.functional as F``
     - ``import torch_lattice.nn.functional as F``
   * - ad-hoc benchmark/example scripts
     - ``uv run bench ...`` and ``uv run conformance ...``
   * - implicit export conventions
     - ``torch_lattice.artifact.save_lattice_model_artifact(...)``

The top-level names are intentionally close enough for migration, but the new
artifact and conformance packages are part of the supported workflow. Prefer
using those over carrying old local scripts forward.

Artifact export differences
---------------------------

A Torch Lattice model can be trained in PyTorch and exported as a lattice artifact
for MLX replay. The artifact is not a serialized ``torch.nn.Module`` and should
not be treated as a Torch checkpoint.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Meaning
   * - ``graph.mlir``
     - Sparse graph structure, operation names, attributes, inputs, and outputs.
   * - ``weights.safetensors``
     - Named parameter tensors consumed by the graph.
   * - ``graph.mlir`` module attributes
     - Loader-facing artifact identity and IO bookkeeping embedded in the graph.

For exportable models, avoid depending on Python control flow that FX cannot see
from example inputs. Use explicit modules, stable state-dict names, and supported
sparse ops. Branches, adds, cats, pooling, activations, and supported convolution
families should be represented as graph operations rather than hidden side effects.

Backend and configuration differences
-------------------------------------

Torch Lattice keeps CUDA dataflow controls for performance work, but those flags
are not semantic knobs. Implicit GEMM, Fetch-on-Demand, and Gather-Scatter must
compute the same sparse result for a fixed relation and weight tensor within
normal floating-point tolerance.

When porting tuned models or scripts:

* keep backend tuning as a performance step after correctness is established;
* do not rely on a specific dataflow to change sparse support;
* keep ``torch_lattice.backends.hash_rsv_ratio`` changes close to the workload
  that requires them;
* run migration/conformance checks after changing kernel-map or convolution
  configuration.

Gameleon reproduction gate
---------------------------

Treat a Gameleon port as a semantic model migration, not an import alias. Replace
Minkowski coordinate-map-key arguments with explicit target ``SparseTensor``
values, and replace stride-one TorchSparse refinement convolutions with
``SubmConv3d``. The conformance suite composes normalized convolution, average
downsampling, target transpose convolution, pooling transpose, trilinear
upsampling, exact reindexing, and sparse concatenation in one trainable block.

Checkpoint conversion must be explicit and strict. Convert parameter names and
kernel layouts into Torch Lattice's documented state-dict layout, then require
that every expected tensor is loaded. Do not use ``strict=False`` or retain
randomly initialized missing parameters for a reproduction run. Before an
expensive training job, require a CUDA optimizer-step test, checkpoint roundtrip,
and whole-block output/gradient comparison against the source implementation.

For a historical TorchSparse checkpoint, write a JSON mapping from each
``.kernel`` key to its exact ``[Kx, Ky, Kz]`` shape and run:

.. code-block:: bash

   uv run convert-checkpoint legacy.pt weights.safetensors \
     --kernel-spec legacy-kernels.json

The resulting manifest records the exact historical source layout and row
permutation for every tensor. Historical TorchSparse used x-fastest rows for
odd-volume kernels and z-fastest rows for even-volume kernels. The converter
refuses unlisted legacy kernels and never runs as part of model loading, export,
or inference.

Fetch-on-Demand training correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Original TorchSparse used a plane-major ``(2, M)`` neighbor map for
Fetch-on-Demand forward, then passed that same storage to the pair-major
Gather-Scatter backward fallback. The fallback interpreted the storage as
``(M, 2)``, so Fetch-on-Demand training could produce incorrect and
nondeterministic input and weight gradients even though execution completed.

Torch Lattice separates the canonical pair-major relation from the
Fetch-on-Demand execution view. Both layouts are produced together when the
kernel map is built: Fetch-on-Demand forward consumes its optimized view, while
the backward fallback consumes the canonical relation expected by
Gather-Scatter. Gradient parity is checked against Gather-Scatter for both map
builders, fused and unfused execution, support-preserving and support-generating
convolutions, and supported floating-point dtypes.

No model-source change is required. Forward values can still differ slightly
between dataflows because CUDA accumulation order is different; compare them
with dtype-appropriate floating-point tolerances rather than requiring bitwise
identity.

Validation workflow
-------------------

Use three levels of validation while migrating:

#. **Layer-level semantic checks**: verify coordinate support before and after
   each migrated convolution or pooling layer.
#. **Migration compatibility checks**: run the migration CLI against the covered
   original TorchSparse subset.
#. **Artifact replay checks**: export fixtures on CUDA and replay them with MLX
   Lattice, then inspect absolute and relative error distributions.

Commands:

.. code-block:: bash

   uv run migration all --device cuda
   uv run e2e-fixtures --device cuda --archive /tmp/lattice-e2e.tar.gz
   uv run fuzz --cases 32 --device cuda --archive /tmp/lattice-fuzz.tar.gz

On the MLX side, replay the exported archive:

.. code-block:: bash

   uv run conformance replay /tmp/lattice-fuzz.tar.gz \
     --report /tmp/lattice-fuzz-report.json

Common migration mistakes
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Symptom
     - Likely cause
   * - Coordinate count grows after a layer that should be support-preserving.
     - A legacy stride-1 spatial convolution was mapped to ``Conv3d`` instead of
       ``SubmConv3d``.
   * - MLX replay cannot resolve a weight name.
     - The exported model used unstable module naming or mutated parameters
       outside the traced module state.
   * - CUDA and MLX outputs differ only at small floating-point scale.
     - Accumulation order differs across CUDA and Metal. Check percentile error
       statistics before treating this as semantic drift.
   * - Crop or BEV output uses the wrong axis.
     - The data pipeline is treating the batch column as a spatial coordinate or
       using a mismatched coordinate convention.
   * - A benchmark script works but artifact export fails.
     - The model path contains Python behavior that runs in eager mode but is not
       represented in the supported FX/lattice graph.

What not to preserve
--------------------

Do not preserve old local wrappers only to keep historical TorchSparse script
shape. If a wrapper only hides whether an operation is support-preserving,
support-generating, or target-aligned, remove it during migration. The long-term
contract is explicit sparse semantics plus artifact replay, not maximum source
compatibility with every old example.
