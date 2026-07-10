Sparse operators
================

``reindex_sparse`` is the exact-support operation for decoder and context
branches. It keeps target row order and coordinate identity, drops source-only
rows, and fills target-only rows without routing through a pooling kernel.

``prune`` keeps explicit row indices in caller order. ``prune_mask`` is the
boolean-mask form and preserves feature gradients through selected rows.

``sparse_from_coordinates(..., duplicate_reduction='mean')`` performs an
unweighted feature mean for exact duplicate integer coordinates. Reduced
coordinates retain their first-occurrence order.

.. automodule:: torch_lattice.operators
   :members:
   :undoc-members: False
