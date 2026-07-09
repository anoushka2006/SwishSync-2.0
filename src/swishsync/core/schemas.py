"""Intermediate representations (IR) for the SwishSync platform pipeline.

Every stage of the pipeline consumes and emits these types. They are the ONLY
things that cross stage boundaries. No stage imports another stage's model; they
speak IR.

Design rules:
- pixel space and court space are both explicit. Anything downstream of tracking
  should prefer court coordinates (WorldState), never raw pixels.
- schemas are versioned via SCHEMA_VERSION; bump it on any breaking change.
- plain dataclasses + stdlib only (no pydantic dep) so the IR stays cheap to
  import on a CPU-only device.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

SCHEMA_VERSION = "0.1.0"


class ObjectClass(str, Enum):
    """What a vision model may emit. Detectors emit these, NOT basketball roles.

    'shooter' / 'ball handler' are semantic roles assigned in the World/Events
    layers, never by a detector.
    """

    PLAYER = "player"
    BALL = "ball"
    HOOP = "hoop"
    BACKBOARD = "backboard"
    REFEREE = "referee"


class EventType(str, Enum):
    """Basketball events. Only SHOT is implemented in this scaffold."""

    SHOT = "shot"
    POSSESSION = "possession"  # placeholder — not implemented
    REBOUND = "rebound"  # placeholder — not implemented


class Team(str, Enum):
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# --------------------------------------------------------------------------- #
# Vision layer IR
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One detected object in one frame (pixel space)."""

    frame: int
    t_ms: float
    cls: ObjectClass
    bbox_xyxy: BBox
    conf: float
    # optional appearance embedding for team-classification / ReID. Riding on the
    # detection avoids a second inference pass later.
    embedding: tuple[float, ...] | None = None

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox_xyxy)


@dataclass
class TrackState:
    """One track's state in one frame."""

    frame: int
    t_ms: float
    bbox_xyxy: BBox
    conf: float
    interpolated: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox_xyxy)


@dataclass
class Track:
    """One object followed across frames. team/jersey are assigned downstream,
    never by the tracker."""

    track_id: int
    cls: ObjectClass
    states: list[TrackState] = field(default_factory=list)
    team: Team | None = None
    jersey: int | None = None


@dataclass
class CourtModel:
    """Image->court homography. The linchpin: every court-space analytic depends
    on this being accurate. `confidence` is honest self-report, not decoration."""

    homography: list[list[float]]  # 3x3
    method: str  # "manual_4pt" | "auto_keypoint" | "identity"
    confidence: float
    valid_frames: tuple[int, int]


# --------------------------------------------------------------------------- #
# World layer IR (court coordinates)
# --------------------------------------------------------------------------- #
@dataclass
class WorldEntity:
    track_id: int
    cls: ObjectClass
    x: float  # court coords
    y: float
    vx: float = 0.0
    vy: float = 0.0
    team: Team | None = None


@dataclass
class WorldState:
    """Everything in court space for one frame."""

    frame: int
    t_ms: float
    players: list[WorldEntity] = field(default_factory=list)
    ball: WorldEntity | None = None
    hoop: tuple[float, float] | None = None


# --------------------------------------------------------------------------- #
# Analytics primitives
# --------------------------------------------------------------------------- #
@dataclass
class ParabolaFit:
    """Quadratic shot fit y = a*x^2 + b*x + c. Field-compatible with the legacy
    swishsync_cv.data.ParabolaFit so the existing engine's output maps 1:1."""

    coefficients: tuple[float, float, float]
    r_squared: float
    apex_x: float
    apex_y: float
    x_min: float
    x_max: float
    weighted_r_squared: float = 0.0
    weighted_residual_rmse: float = 0.0


# --------------------------------------------------------------------------- #
# Event layer IR
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """A basketball event. `evidence` is mandatory: an event you can't explain is
    an event you can't debug or benchmark-gate."""

    type: EventType
    t_start_ms: float
    t_end_ms: float
    actors: list[int] = field(default_factory=list)  # track_ids
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    # typed payload per event kind (e.g. {"parabola_fit": ParabolaFit} for SHOT)
    payload: dict[str, Any] = field(default_factory=dict)


def to_serializable(obj: Any) -> Any:
    """Recursively convert IR dataclasses/enums to json-safe primitives."""

    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj
