"""Event layer interface. Event detectors turn tracks + world state into
basketball Events. This is the layer with NO off-the-shelf models — the research
lives here. Nothing in this layer imports a vision model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from swishsync.core.schemas import Event, Track, WorldState


class EventDetector(ABC):
    """tracks (+ optional world state) -> events."""

    @abstractmethod
    def detect(
        self,
        tracks: list[Track],
        world: list[WorldState] | None = None,
    ) -> list[Event]:
        ...
