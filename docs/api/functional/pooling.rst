Functional pooling
==================

Local pooling supports sum, maximum, and contributor-average reductions.
``pool_transpose3d`` performs contributor averaging on generated fine support
or on an explicit target tensor. The explicit route preserves target coordinate
order and emits zero for unmatched rows.

.. automodule:: torch_lattice.nn.functional.pooling
   :members:
   :undoc-members: False
