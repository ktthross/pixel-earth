"""Which mirrored EPIC frames are worth decoding for a given viewpoint.

Two cheap-then-expensive stages, so an image is only ever opened and
segmented if it might actually contribute a pixel:

1. :func:`load_frame_index` reads every mirrored day's ``metadata.json`` --
   no PNGs touched -- and :func:`candidate_frames` narrows that down to the
   ones whose own sub-satellite point is close enough to a target viewpoint to
   possibly overlap it, using only the ``lat``/``lon`` already in that
   metadata.
2. :class:`GeometryCache` decodes and segments (:func:`pixel_earth.segment.segment`)
   only frames that survive stage 1, and remembers the result on disk so a
   second run -- or the next viewpoint that happens to share a candidate --
   never re-segments the same frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import UnidentifiedImageError

from pixel_earth.batch import Settings, load_rgb
from pixel_earth.epic import Frame, parse_frames
from pixel_earth.geometry import great_circle_distance_deg
from pixel_earth.segment import segment

_CACHE_DIR = ".cache"


def load_frame_index(root: Path, collection: str = "natural") -> list[Frame]:
    """Every frame described by a mirrored day's ``metadata.json``.

    Reads only JSON -- this is the metadata-only step, safe to call on the
    full mirror before deciding which frames are worth decoding.
    """
    frames: list[Frame] = []
    for meta_path in sorted((root / collection).glob("*/*/*/metadata.json")):
        try:
            raw = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        frames.extend(parse_frames(raw, collection))
    return frames


def candidate_frames(
    target_lat0: float,
    target_lon0: float,
    frames: list[Frame],
    *,
    cap_radius_deg: float,
    max_candidates: int | None = None,
) -> list[Frame]:
    """Frames whose own visibility cap could overlap the target's, nearest first.

    Two caps of radius ``cap_radius_deg`` -- one centred on the target
    viewpoint, one on a candidate frame's own sub-satellite point -- can only
    share a point if the centres are within ``2 * cap_radius_deg`` of each
    other. This is deliberately loose (it says nothing about any specific
    pixel); the exact per-pixel test lives in
    :func:`pixel_earth.mosaic.render_viewpoint`.
    """
    usable = [f for f in frames if f.lat is not None and f.lon is not None]
    if not usable:
        return []

    lats = np.array([f.lat for f in usable], dtype=np.float64)
    lons = np.array([f.lon for f in usable], dtype=np.float64)
    distance = great_circle_distance_deg(target_lat0, target_lon0, lats, lons)

    order = np.argsort(distance)
    kept = [usable[i] for i in order if distance[i] < 2 * cap_radius_deg]
    if max_candidates is not None:
        kept = kept[:max_candidates]
    return kept


@dataclass(frozen=True)
class FrameGeometry:
    """Where a decoded frame's disc sits in its own pixel grid."""

    frame: Frame
    cx: float
    cy: float
    radius: float
    looks_like_disc: bool


def _settings_key(settings: Settings) -> str:
    return json.dumps(settings.as_kwargs(), sort_keys=True)


class GeometryCache:
    """Decode + segment on demand, memoised in-process and on disk.

    Persisted beside the mirror at ``<root>/.cache/turntable-geometry.json``,
    keyed by ``(collection, image, settings)`` so different segmentation
    settings never collide -- same spirit as :func:`pixel_earth.batch.run_id`,
    but the cache holds many frames' worth of geometry rather than naming one
    run's output folder.
    """

    def __init__(self, root: Path, *, settings: Settings = Settings(), fmt: str = "png"):
        self._root = root
        self._settings = settings
        self._fmt = fmt
        self._path = root / _CACHE_DIR / "turntable-geometry.json"
        self._disk: dict = self._load_disk()
        self._mem: dict[tuple[str, str, str], FrameGeometry | None] = {}
        self._dirty = False

    def _load_disk(self) -> dict:
        try:
            return json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self, frame: Frame) -> FrameGeometry | None:
        key = (frame.collection, frame.image, _settings_key(self._settings))
        if key in self._mem:
            return self._mem[key]

        disk_key = " ".join(key)
        if disk_key in self._disk:
            record = self._disk[disk_key]
            geometry = None if record is None else FrameGeometry(frame=frame, **record)
            self._mem[key] = geometry
            return geometry

        geometry = self._compute(frame)
        self._mem[key] = geometry
        self._disk[disk_key] = None if geometry is None else _geometry_record(geometry)
        self._dirty = True
        return geometry

    def _compute(self, frame: Frame) -> FrameGeometry | None:
        path = frame.local_path(self._root, self._fmt)
        if not path.exists():
            return None
        try:
            rgb = load_rgb(path)
        except (UnidentifiedImageError, OSError, ValueError):
            return None

        result = segment(rgb, **self._settings.as_kwargs())
        if result.is_empty or result.tight_bbox is None:
            return None

        left, top, right, bottom = result.tight_bbox
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        radius = ((right - left) + (bottom - top)) / 4.0  # mean of the two half-extents
        return FrameGeometry(
            frame=frame, cx=cx, cy=cy, radius=radius, looks_like_disc=result.looks_like_disc()
        )

    def flush(self) -> None:
        """Write accumulated results to disk. Best effort; never fatal."""
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._disk))
            self._dirty = False
        except OSError:
            pass

    def __enter__(self) -> "GeometryCache":
        return self

    def __exit__(self, *exc_info) -> None:
        self.flush()


def _geometry_record(geometry: FrameGeometry) -> dict:
    record = asdict(geometry)
    del record["frame"]  # the frame is the lookup key, not part of the cached value
    return record
