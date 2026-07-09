"""Plugin registry: swap detectors / trackers / calibrators / event-detectors by
name from config, no code edits.

Discipline: don't register a second impl of an interface until you actually have
one. An abstraction with a single implementation is speculation, not modularity.
"""

from __future__ import annotations

from typing import Callable, TypeVar

_T = TypeVar("_T")

DETECTORS: dict[str, type] = {}
TRACKERS: dict[str, type] = {}
CALIBRATORS: dict[str, type] = {}
EVENT_DETECTORS: dict[str, type] = {}


def _register(table: dict[str, type], name: str) -> Callable[[type[_T]], type[_T]]:
    def deco(cls: type[_T]) -> type[_T]:
        if name in table:
            raise ValueError(f"{name!r} already registered in {table}")
        table[name] = cls
        return cls

    return deco


def register_detector(name: str):
    return _register(DETECTORS, name)


def register_tracker(name: str):
    return _register(TRACKERS, name)


def register_calibrator(name: str):
    return _register(CALIBRATORS, name)


def register_event_detector(name: str):
    return _register(EVENT_DETECTORS, name)


def get(table: dict[str, type], name: str) -> type:
    if name not in table:
        raise KeyError(f"{name!r} not registered. Available: {sorted(table)}")
    return table[name]
