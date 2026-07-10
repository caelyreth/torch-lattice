Sparse tensor model
===================

A :class:`torch_lattice.SparseTensor` stores a sparse batch as aligned coordinate
and feature rows.

.. math::

   X = (C, F), \qquad C \in \mathbb{Z}^{N \times 4}, \quad F \in \mathbb{R}^{N \times C_{in}}

The coordinate row layout is:

.. code-block:: text

   [batch, x, y, z]

The feature row at index ``i`` describes the coordinate row at index ``i``. All
operators that change support must therefore construct a new coordinate tensor
and a relation from input rows to output rows.

Stride and spatial shape
------------------------

``SparseTensor.stride`` records the sparse tensor's lattice stride relative to the
input coordinate space. Downsampling convolutions and pooling increase stride;
submanifold operators preserve it.

``SparseTensor.coord_manager`` owns coordinate maps and cached sparse relations.
``SparseTensor.coord_key`` identifies one exact support inside that manager.
Feature-only operations preserve both values; crop, pooling, joins that change
rows, and support-generating convolutions create a new key. Cache reuse therefore
depends on coordinate identity and kernel execution attributes rather than stride
alone.

``SparseTensor.spatial_range`` declares ``(batch, x, y, z)`` capacity when it is
known. ``batch_counts`` optionally records active rows per batch, including empty
batches. Supplying it avoids inferring batch partitions from device data and is
recommended for global pooling and artifact export.

Batching
--------

Batch identity is part of every coordinate row. Sparse operators never merge rows
across different batch values. Concatenating samples should therefore concatenate
coordinates after assigning the intended batch column.

Value alignment
---------------

Sparse algebra is value-aligned rather than shape-only. Combining branches is
valid when the operator can identify the coordinate rows being combined. A branch
merge that silently assumes row order without checking coordinate identity is not
part of the stable semantics.
