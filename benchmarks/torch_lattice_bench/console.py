from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TextIO

type ColorMode = Literal["auto", "always", "never"]


@dataclass(slots=True)
class Console:
    use_color: bool
    quiet: bool = False
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    total: int | None = None
    current: int = 0

    def set_total(self, total: int) -> None:
        self.total = total
        self.current = 0

    def heading(self, text: str) -> None:
        if not self.quiet:
            self.write(self.style(text, "1", "36"))

    def report(self, json_path: Any, summary_path: Any) -> None:
        if self.quiet:
            return
        self.write(
            f"{self.style('report', '35')} {self.style(str(json_path), '2')} "
            f"{self.style('summary=', '2')}{self.style(str(summary_path), '36')}"
        )

    def start(
        self, case: Any, params: Mapping[str, Any], mode: str, device: str
    ) -> None:
        if self.quiet:
            return
        self.current += 1
        detail = self._format_params(params)
        self.write(
            f"{self._progress()}{self.style('run', '36')} "
            f"{self.style(f'{device:<5}', '33')} "
            f"{self.style(f'{mode:<12}', '35')} "
            f"{self.style(f'{case.group}/{case.name}', '1')} {detail}".rstrip()
        )

    def done(self, result: Any, *_: Any) -> None:
        if self.quiet:
            return
        if result.skipped:
            self.write(
                f"     {self.style('skip', '2')} {self.style(result.notes, '2')}"
            )
            return
        workload = self._format_workload(result.workload)
        self.write(
            f"     {self.style('ok ', '32')} "
            f"{self.style('median=', '2')}{self.style(f'{result.median_ms:.3f}ms', '1', '32')} "
            f"{self.style('p95=', '2')}{self.style(f'{result.p95_ms:.3f}ms', '32')} "
            f"{workload}".rstrip()
        )

    def failed(
        self,
        case: Any,
        params: Mapping[str, Any],
        mode: str,
        device: str,
        error: BaseException,
    ) -> None:
        if self.quiet:
            return
        self.write(
            f"     {self.style('fail', '31')} "
            f"{self.style(f'{device:<5}', '33')} "
            f"{self.style(f'{mode:<12}', '35')} "
            f"{self.style(f'{case.group}/{case.name}', '1')}: "
            f"{self.style(type(error).__name__ + ': ' + str(error), '31')}"
        )

    def skipped(
        self, case: Any, params: Mapping[str, Any], mode: str, device: str
    ) -> None:
        if not self.quiet:
            self.write(
                f"     {self.style('skip', '2')} "
                f"{self.style(f'{device:<5}', '33')} "
                f"{self.style(f'{mode:<12}', '35')} "
                f"{self.style(f'{case.group}/{case.name}', '1')}"
            )

    def write(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def style(self, text: str, *codes: str) -> str:
        if not self.use_color or not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def _progress(self) -> str:
        if self.total is None:
            return ""
        width = len(str(max(self.total, 1)))
        return self.style(f"[{self.current:{width}d}/{self.total:{width}d}] ", "2")

    def _format_params(self, params: Mapping[str, Any]) -> str:
        preferred = (
            "N",
            "points",
            "channels",
            "channels_in",
            "channels_out",
            "layout",
            "dtype",
            "kernel",
            "stride",
        )
        parts = [self._kv(key, params[key]) for key in preferred if key in params]
        parts.extend(
            self._kv(key, params[key]) for key in sorted(params) if key not in preferred
        )
        return " ".join(parts)

    def _format_workload(self, workload: Mapping[str, Any]) -> str:
        labels = (
            ("points", "P"),
            ("n_in", "Nin"),
            ("n_out", "Nout"),
            ("edges", "E"),
            ("channels_in", "Cin"),
            ("channels_out", "Cout"),
            ("kernel_volume", "K"),
            ("avg_neighbors", "avgN"),
            ("memory_mb", "memMB"),
        )
        parts = []
        for key, label in labels:
            value = workload.get(key)
            if isinstance(value, int):
                parts.append(self._kv(label, value, key_style="2", value_style="36"))
            elif isinstance(value, float):
                parts.append(
                    self._kv(label, f"{value:.2f}", key_style="2", value_style="36")
                )
        return " ".join(parts)

    def _kv(
        self, key: str, value: Any, *, key_style: str = "2", value_style: str = "33"
    ) -> str:
        return (
            f"{self.style(f'{key}=', key_style)}{self.style(str(value), value_style)}"
        )


def make_console(color: ColorMode, *, quiet: bool = False) -> Console:
    if color == "always":
        use_color = True
    elif color == "never":
        use_color = False
    else:
        use_color = sys.stdout.isatty()
    return Console(use_color=use_color, quiet=quiet)


__all__ = ["ColorMode", "Console", "make_console"]
