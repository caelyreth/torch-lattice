Benchmarks
==========

The benchmark suite mirrors the MLX Lattice benchmark vocabulary so CUDA and
Metal results can be compared by case, layout, feature width, and mode.

Basic usage
-----------

.. code-block:: bash

   uv run bench --preset smoke --group conv --device cuda
   uv run bench --list

A fuller run can sweep sizes, layouts, channels, and modes:

.. code-block:: bash

   uv run bench \
     --preset standard \
     --group tensor \
     --group conv \
     --group nn \
     --size 8192 \
     --size 32768 \
     --channels 32 \
     --channels 64 \
     --layout grid \
     --layout block4 \
     --mode cold_op \
     --mode hot_op \
     --output cuda-standard.json

Case groups
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Group
     - Coverage
   * - ``tensor``
     - Sparse tensor construction, device/dtype paths, branch combination, global pooling, crop, activation, and normalization modules.
   * - ``hash``
     - Coordinate hashing, offset hashing, self-query, and count kernels.
   * - ``dense``
     - Dense materialization, voxelization, devoxelization, and interpolation weights.
   * - ``kmap``
     - Downsample/upsample helpers and kernel-map construction across CUDA dataflows.
   * - ``conv``
     - Pointwise, spatial, strided, Fetch-on-Demand, Gather-Scatter, target, and submanifold convolution paths.
   * - ``nn``
     - Composed sparse modules including residual, cat, classifier-style, and activation-chain graphs.
   * - ``train``
     - Forward/backward convolution paths.

Density layouts
---------------

Synthetic coordinate layouts include ``isolated``, ``line``, ``plane``, ``grid``,
and dense local blocks from ``block2`` through ``block8``. They intentionally
stress different kernel-map and memory-locality patterns.

Output
------

Reports contain environment metadata, per-case latency samples, median/min/p90
/p95 latency, throughput units, workload metrics, and skip/error notes. Relative
``--output`` paths are written under ``benchmarks/results`` with a text summary
next to the JSON file.
