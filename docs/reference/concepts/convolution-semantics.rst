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

Dataflows
---------

The CUDA backend exposes multiple dataflow choices through functional
configuration: implicit GEMM, Fetch-on-Demand, and Gather-Scatter. They are
execution strategies, not semantic differences. For a fixed coordinate relation
and weight tensor, they must compute the same sparse result up to normal floating
point ordering differences.

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
