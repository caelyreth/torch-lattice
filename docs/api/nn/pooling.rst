Pooling modules
===============

``PoolTranspose3d`` is the sparse inverse-resolution module. Calling it with
only a coarse tensor generates fine support; passing a second sparse tensor
uses that tensor as exact output support. Both routes average all valid coarse
contributors per output row.

.. automodule:: torch_lattice.nn.modules.pooling
   :members:
   :undoc-members: False
