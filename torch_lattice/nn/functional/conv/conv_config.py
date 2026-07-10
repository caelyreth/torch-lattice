from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterator, Mapping

from .conv_mode import ConvMode


class Dataflow(Enum):
    ImplicitGEMM = 0
    GatherScatter = 1
    FetchOnDemand = 2


@dataclass(slots=True)
class ConvConfig(Mapping[str, Any]):
    """Validated sparse-convolution execution policy."""

    dataflow: Dataflow = Dataflow.ImplicitGEMM
    ifsort: bool = False
    kmap_mode: str = "hashmap_on_the_fly"
    downsample_mode: str = "spconv"
    split_mask_num: int = 1
    split_mask_num_bwd: int = 3
    wgrad_split_k: int | str = "auto"
    IGEMM_center_only: bool = False
    epsilon: float = 0.0
    mm_thresh: int = 0
    FOD_fusion: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dataflow, Dataflow):
            raise TypeError("dataflow must be a Dataflow value")
        if self.kmap_mode not in {"hashmap", "hashmap_on_the_fly"}:
            raise ValueError("kmap_mode must be 'hashmap' or 'hashmap_on_the_fly'")
        if self.downsample_mode not in {"spconv", "minkowski"}:
            raise ValueError("downsample_mode must be 'spconv' or 'minkowski'")
        if self.split_mask_num < 1 or self.split_mask_num_bwd < 1:
            raise ValueError("split mask counts must be positive")
        if self.wgrad_split_k != "auto" and int(self.wgrad_split_k) < 1:
            raise ValueError("wgrad_split_k must be 'auto' or a positive integer")

    def copy(self) -> ConvConfig:
        return replace(self)

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


_global_conv_config: ContextVar[ConvConfig | None] = ContextVar(
    "torch_lattice_conv_config", default=None
)


def get_global_conv_config() -> ConvConfig | None:
    config = _global_conv_config.get()
    return None if config is None else config.copy()


def set_global_conv_config(conv_config: ConvConfig | Mapping[str, Any]) -> None:
    _global_conv_config.set(_coerce_config(conv_config))


def clear_global_conv_config() -> None:
    _global_conv_config.set(None)


def get_default_conv_config(
    conv_mode: ConvMode = ConvMode.mode0,
    training: bool = False,
) -> ConvConfig:
    del training
    config = ConvConfig()
    if conv_mode == ConvMode.mode1:
        config.ifsort = True
    elif conv_mode == ConvMode.mode2:
        config.ifsort = True
        config.split_mask_num = 3
    elif conv_mode != ConvMode.mode0:
        raise ValueError(f"unsupported convolution mode: {conv_mode}")
    return config


def _coerce_config(value: ConvConfig | Mapping[str, Any]) -> ConvConfig:
    if isinstance(value, ConvConfig):
        return value.copy()
    known = set(ConvConfig.__dataclass_fields__)
    unknown = set(value) - known
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown convolution configuration fields: {names}")
    return ConvConfig(**dict(value))


__all__ = [
    "ConvConfig",
    "Dataflow",
    "clear_global_conv_config",
    "get_default_conv_config",
    "get_global_conv_config",
    "set_global_conv_config",
]
