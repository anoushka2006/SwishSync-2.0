"""Minimal pipeline orchestrator.

Runs the stages in order, each a pure function of its input IR. Stage outputs are
returned in a bundle and (optionally) cached to disk so any downstream stage can
be re-run without re-running detection — the research-iteration payoff.

This scaffold wires detection -> tracking -> events. World projection is a stub
(identity) until a CourtCalibrator lands; that is intentional and flagged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from swishsync.core.schemas import (
    CourtModel,
    Event,
    Track,
    WorldState,
    to_serializable,
)
from swishsync.events.interfaces import EventDetector
from swishsync.vision.interfaces import CourtCalibrator, Detector, Tracker


@dataclass
class PipelineResult:
    tracks: list[Track] = field(default_factory=list)
    court: CourtModel | None = None
    world: list[WorldState] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


@dataclass
class Pipeline:
    detector: Detector | None = None
    tracker: Tracker | None = None
    calibrator: CourtCalibrator | None = None
    event_detectors: list[EventDetector] = field(default_factory=list)

    def run(
        self,
        frames,  # iterable of (frame_index, t_ms, frame_pixels)
        cache_dir: Path | None = None,
    ) -> PipelineResult:
        result = PipelineResult()

        # vision: detect + track
        if self.detector is not None and self.tracker is not None:
            for frame_index, t_ms, pixels in frames:
                dets = self.detector.detect(pixels, frame_index, t_ms)
                self.tracker.update(dets, frame_index, t_ms, frame=pixels)
                if self.calibrator is not None and result.court is None:
                    result.court = self.calibrator.calibrate(pixels, frame_index)
            result.tracks = self.tracker.tracks()

        # world: identity projection stub (replace once calibration lands)
        # result.world stays empty in the scaffold; event detectors that only
        # need pixel-space tracks (shot) work without it.

        # events
        for ed in self.event_detectors:
            result.events.extend(ed.detect(result.tracks, result.world or None))

        if cache_dir is not None:
            self._cache(result, cache_dir)
        return result

    @staticmethod
    def _cache(result: PipelineResult, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "tracks.json").write_text(
            json.dumps(to_serializable(result.tracks), indent=2)
        )
        (cache_dir / "events.json").write_text(
            json.dumps(to_serializable(result.events), indent=2)
        )
