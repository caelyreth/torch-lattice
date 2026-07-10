Sparse tensor
=============

Batch views are derived from the coordinate batch column, while declared
``batch_counts`` or ``spatial_range`` preserve empty batches. The decomposed
coordinate views omit the batch column and remain available after row-changing
layers.

.. automodule:: torch_lattice.tensor
   :members:
   :undoc-members: False
