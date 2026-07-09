Installation
============

``torch-lattice`` is published on PyPI as a CUDA extension package. Most users
should install the published wheel; a local CUDA toolkit is only required when
building from source or developing the native extension.

Install from PyPI
-----------------

For a project managed by ``uv``, add the package with the PyTorch CUDA 12.8
backend selected:

.. code-block:: bash

   uv add torch-lattice --torch-backend cu128

For an existing virtual environment, install directly with:

.. code-block:: bash

   uv pip install --torch-backend cu128 torch-lattice

Runtime requirements
--------------------

The published wheel currently targets:

* Linux ``x86_64``;
* Python ``3.14``;
* PyTorch ``2.11.0+cu128`` from the official CUDA 12.8 wheel index;
* an NVIDIA driver compatible with the CUDA runtime shipped through the PyTorch
  dependency stack.

In normal installed-wheel usage, you do not need ``nvcc`` or a local CUDA toolkit.
Those are build-time requirements, not runtime requirements.

Check the installed runtime with:

.. code-block:: bash

   uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

If ``torch.cuda.is_available()`` is false, import-only and some CPU-safe checks
may still work, but CUDA sparse operators, benchmarks, and training workflows
require a real CUDA device.

Development requirements
------------------------

For development from a checkout, use a Linux CUDA build environment with:

* Python ``>= 3.14``;
* ``uv`` ``>= 0.11.25``;
* a CUDA toolkit compatible with the configured PyTorch wheel;
* PyTorch ``2.11.0+cu128`` from the official CUDA 12.8 wheel index;
* an NVIDIA driver capable of running the selected CUDA runtime.

The repository pins the CUDA 12.8 PyTorch index in ``pyproject.toml``. A normal
workspace setup is:

.. code-block:: bash

   uv sync --all-packages --extra test
   uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

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

   uv sync --group docs --no-install-workspace
   uv run --no-sync sphinx-build -W -b html docs docs/_build/html

The documentation configuration reads the Python sources and mocks the native
CUDA extension for autodoc, so a local CUDA build is not required just to render
the site.
