Stability policy
================

Torch Lattice is still pre-1.0, but the current direction is intentionally narrow:
CUDA training and export should match the MLX artifact contract rather than
preserving every historical TorchSparse behavior.

Stable expectations
-------------------

The following surfaces should be treated as long-term design anchors:

* coordinate rows use ``(batch, x, y, z)``;
* ``SparseTensor`` aligns coordinate rows with feature rows;
* sparse support behavior is explicit in module class names;
* artifacts are MLIR plus safetensors, not Python pickles;
* conformance compares CUDA output with MLX replay output.

Allowed breakage before 1.0
---------------------------

Breaking changes are acceptable when they remove ambiguous legacy behavior,
reduce exporter boilerplate, or align Torch semantics with MLX replay semantics.
When such a change affects migration from original TorchSparse, document the new
mapping and update migration/conformance checks in the same change.
