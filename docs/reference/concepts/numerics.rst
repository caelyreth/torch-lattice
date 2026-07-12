Numerical Conformance
=====================

Torch Lattice defines exact sparse semantics for coordinates, relation edges,
canonical kernel rows, and artifact tensor layouts. Floating-point reductions
are evaluated against the shared ``lattice-contract`` binary64 reference rather
than historical TorchSparse output bits, because valid CUDA and Metal kernels
can accumulate a relation in different orders.

The reference evaluator rounds feature, weight, and bias leaves to binary32,
multiplies in binary64, and uses ``math.fsum`` for the final sum. CUDA tests run
every kernel-map builder and dataflow against it with a non-cubic z-fastest
kernel and an FP32 pointwise cancellation probe.

This is a correctness boundary, not an execution route. Production CUDA kernels
continue to use optimized dataflows and may differ by normal FP32 reduction
order, but they must preserve coordinate support and meet the oracle tolerance.
No TorchSparse compatibility mode changes arithmetic or kernel-row order at
runtime.

Packed weights are a separate, affine-quantized surface. Validate them against
the corresponding dequantized execution with quantization tolerances rather
than the unquantized FP32 oracle.

For historical TorchSparse checkpoints, convert once with explicit kernel
metadata, inspect the emitted permutation manifest, and validate a known model
block before training. The loader and exporter never infer legacy row order.
