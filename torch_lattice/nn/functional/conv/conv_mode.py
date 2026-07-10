from __future__ import annotations

from contextvars import ContextVar
from enum import Enum


class ConvMode(Enum):
    mode0 = 0
    mode1 = 1
    mode2 = 2


_conv_mode: ContextVar[ConvMode] = ContextVar(
    "torch_lattice_conv_mode", default=ConvMode.mode0
)


def get_conv_mode() -> ConvMode:
    return _conv_mode.get()


def set_conv_mode(conv_mode: int | ConvMode) -> None:
    if isinstance(conv_mode, int):
        try:
            conv_mode = ConvMode(conv_mode)
        except ValueError as exc:
            raise ValueError(f"unknown convolution mode: {conv_mode}") from exc
    if not isinstance(conv_mode, ConvMode):
        raise TypeError("conv_mode must be an int or ConvMode")
    _conv_mode.set(conv_mode)


__all__ = ["ConvMode", "get_conv_mode", "set_conv_mode"]
