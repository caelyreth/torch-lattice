Conformance
===========

Conformance tools validate the artifact boundary between Torch/CUDA and MLX/Metal.
They are intentionally separate from benchmarks.

Fixed fixtures
--------------

Fixed E2E fixtures are small deterministic models that cover important graph
patterns. They are useful when changing artifact lowering, graph loading, or core
operator semantics.

Random fuzz fixtures
--------------------

Fuzz generation builds random sparse model structures, runs them on CUDA, exports
the artifact bundle, and stores input/output tensors. The MLX side can replay the
same archive and report numerical error distributions.

Torch-side generation:

.. code-block:: bash

   uv run fuzz --cases 32 --device cuda --archive /tmp/torch_lattice_fuzz.tar.gz
   uv run conformance fuzz --cases 32 --device cuda

MLX-side replay:

.. code-block:: bash

   uv run conformance replay /tmp/torch_lattice_fuzz.tar.gz \
     --report /tmp/torch_lattice_fuzz_report.json

What the reports mean
---------------------

Use absolute and relative error percentiles to inspect numerical agreement.
Different CUDA and Metal accumulation orders can produce small differences, so a
healthy report is a distribution bounded by the tolerance for the dtype and graph
shape rather than literal bitwise equality.
