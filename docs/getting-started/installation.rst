Installation
============

``torch-lattice`` is a CUDA extension package and therefore has stricter build
requirements than the MLX deployment package.

Environment requirements
------------------------

Use a Linux environment with:

* Python ``>= 3.14``;
* a CUDA toolkit compatible with the configured PyTorch wheel;
* PyTorch ``2.11.0+cu128`` from the official CUDA 12.8 wheel index;
* an NVIDIA driver capable of running the selected CUDA runtime;
* ``uv`` ``>= 0.11.25``.

The repository pins the CUDA 12.8 PyTorch index in ``pyproject.toml``. A normal
workspace setup is:

.. code-block:: bash

   uv sync --all-packages --extra test
   uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

If ``torch.cuda.is_available()`` is false, CPU-only import and documentation
builds may still work, but CUDA operator tests and benchmarks will skip or fail
when they intentionally require a real device.

Editable development
--------------------

For development on a CUDA host:

.. code-block:: bash

   export CUDA_PATH=/usr/local/cuda-12.8
   uv sync --all-packages --extra test
   uv run --all-packages --extra test pytest tests -q

The build system uses scikit-build-core and CMake. The CUDA compiler and toolkit
root can be overridden at build time:

.. code-block:: bash

   uv build \
     --sdist \
     --wheel \
     --config-setting=cmake.define.CMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
     --config-setting=cmake.define.CUDAToolkit_ROOT="$CUDA_PATH"

Documentation build
-------------------

The documentation uses the same Sphinx/Furo stack as MLX Lattice:

.. code-block:: bash

   uv sync --all-packages --extra test --group docs
   uv run --group docs sphinx-build -W -b html docs docs/_build/html

The API reference imports the Python package. Build it from an environment where
``torch-lattice`` can be imported successfully.
