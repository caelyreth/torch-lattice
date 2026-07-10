from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("torch-lattice")
    group.addoption(
        "--device",
        action="append",
        default=None,
        help="Device(s) for behavior tests. Default: cpu. Use cuda to include CUDA-only coverage.",
    )
    group.addoption(
        "--dtype",
        action="append",
        default=None,
        help="Dtype(s) for dtype-parametrized cases. Default: float32.",
    )


def pytest_configure(config: pytest.Config) -> None:
    markers: Iterable[tuple[str, str]] = [
        ("ops", "public functional operator behavior"),
        ("nn", "public neural-network module behavior"),
        ("artifact", "MLIR artifact export behavior"),
        ("benchmark", "benchmark package behavior"),
        ("conformance", "cross-runtime fixture and migration tooling"),
        ("core", "sparse tensor and coordinate behavior"),
        ("conv", "sparse convolution behavior"),
        ("pool", "sparse pooling behavior"),
        ("feature", "feature-only sparse tensor behavior"),
        ("cuda", "CUDA-only behavior"),
        ("slow", "slow case that should be opt-in for tight loops"),
    ]
    for name, description in markers:
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "selected_device" in metafunc.fixturenames:
        metafunc.parametrize("selected_device", _device_params(metafunc.config))
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", _dtype_params(metafunc.config))


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _device_params(config: pytest.Config) -> list[object]:
    names = _option_names(config, "--device") or ["cpu"]
    params = []
    for name in names:
        if name == "cuda" and not torch.cuda.is_available():
            params.append(
                pytest.param(
                    name, id=name, marks=pytest.mark.skip(reason="CUDA is unavailable")
                )
            )
        else:
            marks = [pytest.mark.cuda] if name == "cuda" else []
            params.append(pytest.param(torch.device(name), id=name, marks=marks))
    return params


def _dtype_params(config: pytest.Config) -> list[object]:
    names = _option_names(config, "--dtype") or ["float32"]
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    return [pytest.param(mapping[name], id=name) for name in names]


def _option_names(config: pytest.Config, name: str) -> list[str]:
    raw = config.getoption(name) or []
    names: list[str] = []
    for item in raw:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    return names
