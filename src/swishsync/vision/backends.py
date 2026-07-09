"""Inference backends: WHERE a model runs, orthogonal to WHAT it computes.

This is the piece that satisfies "GPU inference, but still runs on CPU-only
devices". A Detector/Tracker is constructed with a Backend; it never knows if
inference is local CPU torch, a CUDA engine, a Triton client, or an HTTP API.
Switching tiers is a config change (backend: cpu|cuda|triton|http), never a
rewrite.

Backends are device PROVIDERS: ultralytics-style models take a `device=`
argument and own their preprocessing, so `run()` simply executes the model with
the backend's device. cpu is mandatory; cuda is the optional accelerator.
Remote backends (triton/http) were trimmed as YAGNI — re-add when a remote GPU
actually exists (DECISION_LOG 2026-07-09).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceBackend(ABC):
    """Runs a model's forward pass. Implementations own device placement."""

    device: str = "cpu"

    @abstractmethod
    def run(self, model: Any, inputs: Any) -> Any:
        """Execute inference. Shape of `inputs`/return is the model's contract."""


class CpuBackend(InferenceBackend):
    """In-process CPU execution. Always available — the mandatory fallback tier."""

    device = "cpu"

    def run(self, model: Any, inputs: Any) -> Any:
        return model(inputs)


class CudaBackend(InferenceBackend):
    """Local GPU execution. The model contract is ultralytics-style: it accepts
    the device at call time; this backend supplies `device="cuda"`. Falls back
    loudly (torch raises) on machines with no CUDA — tier selection is the
    config's job, not silent downgrade."""

    device = "cuda"

    def run(self, model: Any, inputs: Any) -> Any:  # pragma: no cover - GPU env
        return model(inputs)


_BACKENDS: dict[str, type[InferenceBackend]] = {
    "cpu": CpuBackend,
    "cuda": CudaBackend,
}


def make_backend(name: str, **kwargs: Any) -> InferenceBackend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[name](**kwargs)
