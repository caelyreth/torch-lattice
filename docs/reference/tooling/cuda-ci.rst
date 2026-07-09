CUDA CI and release build
=========================

The CUDA workflow has two distinct jobs:

* compile and smoke-check CUDA wheels on Linux;
* run CUDA behavior tests when a real GPU is available.

A GitHub-hosted Linux runner can install the CUDA toolkit and build the extension,
but it does not provide an NVIDIA GPU. Tests that require a real CUDA device are
therefore guarded with a CUDA availability marker. This is expected: compile-time
coverage and runtime GPU coverage are different checks.

Release wheel shape
-------------------

The wheel contains the Python package and native CUDA extension. Build settings
come from ``pyproject.toml`` and can be overridden with CMake config settings.
For a release build, record the CUDA toolkit, PyTorch CUDA version, and target
architecture list used by the build.

Troubleshooting build time
--------------------------

CUDA extension builds are dominated by NVCC compilation. Useful hardening steps
include caching build directories, limiting full wheel builds to release or native
change paths, and keeping CPU-only quality checks separate from CUDA packaging.
Those are CI design choices; they should not change operator semantics.
