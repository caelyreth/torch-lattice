Sparse operators
================

``reindex_sparse`` is the exact-support operation for decoder and context
branches. It keeps target row order and coordinate identity, drops source-only
rows, and fills target-only rows without routing through a pooling kernel.

.. automodule:: torch_lattice.operators
   :members:
   :undoc-members: False
