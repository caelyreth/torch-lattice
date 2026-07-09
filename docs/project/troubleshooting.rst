Troubleshooting
===============

CUDA environment
----------------

First check the CUDA stack from the same environment used to build or run tests:

.. code-block:: bash

   nvcc --version
   uv run python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"

If PyTorch reports CUDA support but no GPU is available, CUDA-only tests should
skip on CI-style CPU runners. On a real CUDA host, that usually means the NVIDIA
driver or container runtime is not exposing the device.

Build failures
--------------

Read the first compiler error, not only the final CMake failure. Common causes are
an incompatible NVCC/PyTorch CUDA pair, missing CUDA toolkit path, unsupported GPU
architecture flags, or memory pressure during parallel NVCC compilation.

Useful build controls:

.. code-block:: bash

   export CUDA_PATH=/usr/local/cuda-12.8
   export MAX_JOBS=2
   export TORCH_CUDA_ARCH_LIST="8.9"

Artifact replay failures
------------------------

When MLX replay does not match CUDA output, inspect the failure in this order:

#. confirm the same artifact archive is replayed;
#. inspect input coordinate ordering and dtype;
#. compare graph operation names and weight names;
#. check absolute and relative error percentiles;
#. reduce the case to the smallest failing fuzz fixture.

A single large relative error near zero-valued outputs is often less informative
than the absolute error distribution and the maximum output magnitude.
