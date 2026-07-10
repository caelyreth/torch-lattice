Coordinate management
=====================

Coordinate managers make sparse support identity explicit. A
``CoordinateMapKey`` belongs to exactly one ``CoordinateManager``; relations
are cached by source key, target key, kernel geometry, semantic operation, and
execution policy. This prevents tensors with equal stride but different rows
from sharing a kernel map.

Feature-only operations preserve the input key. Support-changing operations
insert a new map into the same manager so later inverse convolution and branch
alignment can refer to the precise execution graph.

.. automodule:: torch_lattice.core.coords
   :members:
   :undoc-members: False
