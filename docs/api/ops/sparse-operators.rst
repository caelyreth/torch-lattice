Sparse operators
================

``reindex_sparse`` is the exact-support operation for decoder and context
branches. It keeps target row order and coordinate identity, drops source-only
rows, and fills target-only rows without routing through a pooling kernel.

``prune`` keeps explicit row indices in caller order. ``prune_mask`` is the
boolean-mask form and preserves feature gradients through selected rows.

.. automodule:: torch_lattice.operators
   :members:
   :undoc-members: False
