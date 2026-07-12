Convolution semantics
=====================

Torch Lattice makes sparse support behavior explicit. This avoids the legacy
TorchSparse ambiguity where a stride-1 convolution with a larger kernel could be
interpreted as submanifold-like by convention.

Forward sparse convolution
--------------------------

``Conv3d`` is support-generating. It computes output coordinates from the input
support, kernel size, stride, and dilation, then applies a sparse relation:

.. math::

   Y_o = \sum_{(i, k) \in R(o)} X_i W_k

where ``R(o)`` is the set of input rows and kernel offsets that contribute to
output row ``o``.

Submanifold convolution
-----------------------

``SubmConv3d`` is support-preserving. Its output coordinate set is the input
coordinate set, and only neighbors that also map to those output rows contribute.
This is the right replacement for original TorchSparse stride-1 spatial
convolutions when migrating existing models.

Target convolution
------------------

``Conv3d`` also computes only at a caller-provided coordinate set when invoked as
``conv(x, coordinates=target)``. This is useful when the graph already owns the
output support, for example after a branch, a proposal stage, or a known detector
head layout. Target support is an execution input, not a distinct module family;
the same weights and convolution attributes are used for generated and explicit
support.

Transpose modules use the same rule. Calling ``up(source, target)`` evaluates
only on the target support, while ``up(source)`` follows the ordinary inverse or
generative support policy. Target transpose geometry is defined by
``target = source * stride + offset * dilation - padding`` and is shared with
MLX Lattice artifact replay.

Generated transpose support
---------------------------

``GenerativeConvTranspose3d`` builds the complete deduplicated support implied
by every source row and every canonical kernel position. It is equivalent to a
target transpose whose target is exactly that generated set. The operation does
not clip coordinates to an inferred spatial range: the coordinate relation is
the source of truth, while ``spatial_range`` is propagated with the standard
transpose formula when the input supplied one.

Generated transpose currently requires ``padding=0`` and ``dilation=1``. These
restrictions avoid an underspecified generated-support convention. Use
``ConvTranspose3d(...)(source, target)`` for padded, dilated, or otherwise
caller-owned support, and ensure that target rows are unique. In either form,
the target sparse stride must equal ``source.stride / stride`` exactly.

Target transpose execution policy
---------------------------------

The target relation is currently validated through the Gather-Scatter CUDA
implementation. ``ImplicitGEMM`` and ``FetchOnDemand`` configuration choices
remain available for relations they implement, but target-conditioned transpose
does not silently select either of them. This is an execution policy, not a
semantic distinction: support and feature results are defined by the same
canonical relation and are covered by generated-versus-explicit conformance
tests. Benchmark target-transpose models as their own workload rather than
assuming the dataflow selected for ordinary forward convolution also applies.

Weight-normalized convolution
-----------------------------

``NormalizedSubmConv3d``, ``NormalizedConvTranspose3d``, and
``NormalizedGenerativeConvTranspose3d`` preserve the support behavior of their
corresponding convolution families. Non-pointwise kernels compute

.. math::

   Y = \frac{\operatorname{conv}(X, W)}
            {\sqrt{\operatorname{conv}(\mathbf{1}, W^2) + \varepsilon}} + b.

Both passes reuse the same coordinate relation. The denominator is part of the
training graph, so gradients include its dependence on ``W``. Pointwise
``1x1x1`` kernels bypass normalization and remain a matrix multiplication.
Packed artifact weights are rejected for this family because squaring an
affine-packed representation does not preserve that contract.

Dataflows
---------

The CUDA backend exposes multiple dataflow choices through functional
configuration: implicit GEMM, Fetch-on-Demand, and Gather-Scatter. They are
execution strategies, not semantic differences. For a fixed coordinate relation
and weight tensor, they must compute the same sparse result up to normal floating
point ordering differences.

Canonical kernel rows
---------------------

All convolution modules store a canonical ``weight`` parameter with shape
``(K, C_in, C_out)``. ``K`` rows enumerate ``(x, y, z)`` positions with ``z``
varying fastest, and the persistent ``kernel_positions`` state buffer records
that mapping. The artifact exporter reshapes the same tensor into
``(C_out, Kx, Ky, Kz, C_in)`` using ``conv3d_o_xyz_i``.

This intentionally differs from historical TorchSparse checkpoints. Their
odd-volume ``.kernel`` tensors used x-fastest rows while even-volume tensors
already used z-fastest rows. Convert a trusted checkpoint once on the CUDA side
with an explicit mapping from every legacy kernel key to its kernel size. The
converter writes safetensors and a JSON manifest containing the exact source
layout and row permutation for every tensor; it does not run during model
loading or inference.

The conversion helper only transforms model state. Optimizer state is rejected
from this workflow because parameter identity and optimizer moments require a
separate, explicit resume policy. Converted state dictionaries must match the
destination model exactly; do not use ``strict=False`` to mask missing values.

.. code-block:: bash

   uv run convert-checkpoint legacy.pt weights.safetensors \
     --kernel-spec legacy-kernels.json

Migration rule
--------------

When porting original TorchSparse code, use this mapping first:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Original TorchSparse usage
     - Torch Lattice usage
   * - ``Conv3d(kernel_size > 1, stride=1)``
     - ``SubmConv3d``
   * - ``Conv3d(kernel_size=1)``
     - ``Conv3d``
   * - ``Conv3d(stride > 1)``
     - ``Conv3d`` with the same stride

Do not rely on class-name compatibility alone. Check the support behavior.
