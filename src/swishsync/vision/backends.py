"""Inference backends: WHERE a model runs, orthogonal to WHAT it computes.

This is the piece that satisfies "GPU inference, but still runs on CPU-only
devices". A Detector/Tracker is constructed with a Backend; it never knows if
inference is local CPU torch, a CUDA engine, a Triton client, or an HTTP API.
Switching tiers is a config change (backend: cpu|cuda|triton|http), never a
rewrite.

Only the CPU/null backend is implemented in this scaffold. The GPU/remote
backends are stubs that raise a clear NotImplementedError so the abstraction is
present and the wiring is obvious, without pulling torch into a CPU-only import.
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
    """Local GPU execution (torch.cuda). Stub — implement when a GPU model lands."""

    device = "cuda"

    def run(self, model: Any, inputs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "CudaBackend not implemented yet. Wire torch.cuda here."
        )


class TritonBackend(InferenceBackend):
    """Remote GPU via Triton Inference Server. Stub."""

    device = "remote"

    def __init__(self, url: str = "localhost:8000") -> None:
        self.url = url

    def run(self, model: Any, inputs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("TritonBackend not implemented yet.")


class HttpBackend(InferenceBackend):
    """Hosted inference endpoint (e.g. HF/Roboflow). Stub."""

    device = "remote"

    def __init__(self, url: str = "") -> None:
        self.url = url

    def run(self, model: Any, inputs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("HttpBackend not implemented yet.")


_BACKENDS: dict[str, type[InferenceBackend]] = {
    "cpu": CpuBackend,
    "cuda": CudaBackend,
    "triton": TritonBackend,
    "http": HttpBackend,
}


def make_backend(name: str, **kwargs: Any) -> InferenceBackend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[name](**kwargs)
