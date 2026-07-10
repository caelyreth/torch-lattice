from __future__ import annotations

try:
    from torch_lattice import _C
except ImportError as exc:
    raise ImportError(
        "torch-lattice native operators are unavailable; install a compatible "
        "CUDA wheel or build the project with its CUDA toolkit"
    ) from exc

_EXPECTED = (
    "GPUHashTable",
    "conv_forward_gather_scatter_cuda",
    "build_kernel_map_subm_hashmap",
)
_missing = tuple(name for name in _EXPECTED if not hasattr(_C, name))
if _missing:
    raise ImportError(
        "torch-lattice native extension is incomplete; missing: " + ", ".join(_missing)
    )

__all__ = tuple(name for name in dir(_C) if not name.startswith("_"))
globals().update({name: getattr(_C, name) for name in __all__})
