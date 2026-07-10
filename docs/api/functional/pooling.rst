Functional pooling
==================

Local pooling supports sum, maximum, and contributor-average reductions.
``pool_transpose3d`` performs contributor averaging on generated fine support
or on an explicit target tensor. The explicit route preserves target coordinate
order and emits zero for unmatched rows.
``trilinear_upsample3d`` uses separable linear weights and normalizes by the
weights present on sparse support. It accepts generated or explicit target
coordinates.

.. automodule:: torch_lattice.nn.functional.pooling
   :members:
   :undoc-members: False
