"""Compose a cloud-free disc, or a reusable reference grid, from decoded frames.

No persistent global raster *of full-resolution disc renders*: a
:class:`Viewpoint` is rendered by reprojecting each candidate frame's own
pixels directly into the viewpoint's geometry (``inverse`` the viewpoint's
pixels to a sphere point, ``forward`` that point into each candidate's own
geometry, sample) and choosing among them -- one source-to-target resampling
per candidate, no round trip through an intermediate grid.

Selection never synthesises a colour by combining channels from different
frames: the least-cloudy candidate's whole ``(R, G, B)`` triplet is copied
verbatim. Blending (``blend_k > 1``) is a weighted *average of whole
triplets* among near-tied candidates -- legitimate anti-aliasing across
samples of similar quality, never a per-channel mix.

:func:`render_viewpoint` and :func:`build_reference_grid` share this
selection rule (:func:`_composite_at`) but differ in *where* it's decided.
Deciding independently per viewpoint means the very same physical point can
have its winning candidate flip between one rotation frame and the next
whenever two candidates' scores are close -- visible as flicker across an
animation, even though nothing about the scene actually changed. A
:class:`ReferenceGrid` decides once, in a fixed lat/lon frame, and every
viewpoint samples that fixed decision -- see :mod:`pixel_earth.turntable` for
how a full rotation is built from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pixel_earth.catalog import FrameGeometry
from pixel_earth.cloud_score import CloudScorer
from pixel_earth.geometry import DEFAULT_MIN_COS, Geometry, forward, inverse
from pixel_earth.segment import luminance

_LUMINANCE_FLOOR_SCORE = np.inf  # a sampled pixel darker than the floor can never win


@dataclass(frozen=True)
class Viewpoint:
    """The disc a single output frame is rendered into."""

    lat0: float
    lon0: float
    cx: float
    cy: float
    radius: float
    size: tuple[int, int]  # (height, width) of the output canvas

    @property
    def geometry(self) -> Geometry:
        return Geometry(lat0=self.lat0, lon0=self.lon0, cx=self.cx, cy=self.cy, radius=self.radius)


def make_viewpoint(lat0: float, lon0: float, *, radius: int = 800, margin: float = 1.08) -> Viewpoint:
    """A square canvas centred on ``(lat0, lon0)``, with room around the disc."""
    half = int(round(radius * margin))
    size = (2 * half, 2 * half)
    return Viewpoint(lat0=lat0, lon0=lon0, cx=half, cy=half, radius=float(radius), size=size)


@dataclass(frozen=True)
class Candidate:
    """One decoded, segmented frame available to draw from."""

    geometry: FrameGeometry
    rgb: np.ndarray  # (H, W, 3) uint8, this candidate's own decoded frame


@dataclass(frozen=True)
class RenderResult:
    rgba: np.ndarray  # (H, W, 4) uint8
    suspect: np.ndarray  # (H, W) bool -- on the disc, but no trustworthy candidate
    contributor: np.ndarray  # (H, W) int16 -- winning candidate index, -1 where suspect/off-disc
    suspect_fraction: float  # suspect area / on-disc area


def _composite_at(
    lat: np.ndarray,
    lon: np.ndarray,
    candidates: list[Candidate],
    *,
    scorer: CloudScorer,
    min_cos: float,
    luminance_floor: int,
    obliqueness_penalty: float,
    blend_k: int,
    blend_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick/blend the least-cloudy candidate at each ``(lat, lon)`` sample point.

    Shared by :func:`render_viewpoint` (points on one viewpoint's disc) and
    :func:`build_reference_grid` (points on a fixed equirectangular grid) --
    the selection rule doesn't care what shape the sample points came from.
    Returns ``(rgb, has_data, contributor)``, all shaped like ``lat``/``lon``
    (plus a trailing 3 on ``rgb``); ``rgb`` is only meaningful where
    ``has_data`` is True.
    """
    shape = lat.shape
    if not candidates:
        return (
            np.zeros((*shape, 3), dtype=np.float32),
            np.zeros(shape, dtype=bool),
            np.full(shape, -1, dtype=np.int16),
        )

    scores = np.empty((len(candidates), *shape), dtype=np.float32)
    rgb_stack = np.zeros((len(candidates), *shape, 3), dtype=np.float32)

    for index, candidate in enumerate(candidates):
        source_geometry = Geometry(
            lat0=candidate.geometry.frame.lat,
            lon0=candidate.geometry.frame.lon,
            cx=candidate.geometry.cx,
            cy=candidate.geometry.cy,
            radius=candidate.geometry.radius,
        )
        src_col, src_row, cos_c = forward(lat, lon, geometry=source_geometry)

        src_h, src_w = candidate.rgb.shape[:2]
        in_bounds = (
            (src_col >= 0) & (src_col <= src_w - 1) & (src_row >= 0) & (src_row <= src_h - 1)
        )
        valid = (cos_c >= min_cos) & in_bounds

        safe_row = np.where(valid, src_row, 0.0)
        safe_col = np.where(valid, src_col, 0.0)
        sample_coords = np.stack([safe_row, safe_col])

        sampled = np.stack(
            [
                ndimage.map_coordinates(
                    candidate.rgb[..., c].astype(np.float32), sample_coords, order=1, mode="nearest"
                )
                for c in range(3)
            ],
            axis=-1,
        )
        rgb_stack[index] = sampled

        pixel_score = scorer(sampled.clip(0, 255).astype(np.uint8)).astype(np.float32)
        too_dark = luminance(sampled.clip(0, 255).astype(np.uint8)) < luminance_floor
        pixel_score = np.where(too_dark, _LUMINANCE_FLOOR_SCORE, pixel_score)
        pixel_score = pixel_score + obliqueness_penalty * (1.0 - cos_c)
        scores[index] = np.where(valid, pixel_score, np.inf)

    keep = min(blend_k, len(candidates))
    order = np.argsort(scores, axis=0)
    top_idx = order[:keep]
    top_scores = np.take_along_axis(scores, top_idx, axis=0)
    best_score = top_scores[0]

    threshold = best_score * blend_margin + 1e-6
    finite = np.isfinite(top_scores)
    within_margin = finite & (top_scores <= threshold[np.newaxis])
    weight = np.where(within_margin, 1.0 / np.maximum(top_scores, 1e-6), 0.0)
    weight_sum = weight.sum(axis=0)

    top_rgb = np.take_along_axis(rgb_stack, top_idx[..., np.newaxis].repeat(3, axis=-1), axis=0)
    blended = (weight[..., np.newaxis] * top_rgb).sum(axis=0)

    has_data = weight_sum > 0
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    rgb[has_data] = blended[has_data] / weight_sum[has_data, np.newaxis]
    contributor = np.where(has_data, top_idx[0], -1).astype(np.int16)

    return rgb, has_data, contributor


def render_viewpoint(
    viewpoint: Viewpoint,
    candidates: list[Candidate],
    *,
    scorer: CloudScorer,
    min_cos: float = DEFAULT_MIN_COS,
    luminance_floor: int = 30,
    obliqueness_penalty: float = 0.5,
    blend_k: int = 1,
    blend_margin: float = 1.5,
) -> RenderResult:
    """Render one output disc by picking/blending the least-cloudy candidate pixels.

    Independent per viewpoint: two adjacent rotation frames can pick
    different winning candidates for the same physical point if their scores
    are close. For a temporally-stable rotation, render from a
    :class:`ReferenceGrid` instead (:func:`sample_reference_grid`), which
    decides each point's winner exactly once.
    """
    height, width = viewpoint.size
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    lat, lon, on_disc = inverse(cols, rows, geometry=viewpoint.geometry)

    empty = RenderResult(
        rgba=np.zeros((height, width, 4), dtype=np.uint8),
        suspect=on_disc.copy(),
        contributor=np.full((height, width), -1, dtype=np.int16),
        suspect_fraction=1.0 if on_disc.any() else 0.0,
    )
    if not candidates or not on_disc.any():
        return empty

    rgb, has_data, contributor = _composite_at(
        lat,
        lon,
        candidates,
        scorer=scorer,
        min_cos=min_cos,
        luminance_floor=luminance_floor,
        obliqueness_penalty=obliqueness_penalty,
        blend_k=blend_k,
        blend_margin=blend_margin,
    )

    suspect = on_disc & ~has_data
    alpha = np.where(on_disc & has_data, 255, 0).astype(np.uint8)
    contributor = np.where(on_disc & has_data, contributor, -1).astype(np.int16)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.round().clip(0, 255).astype(np.uint8)
    rgba[..., 3] = alpha

    on_disc_area = float(on_disc.sum())
    suspect_fraction = float(suspect.sum()) / on_disc_area if on_disc_area > 0 else 0.0

    return RenderResult(
        rgba=rgba, suspect=suspect, contributor=contributor, suspect_fraction=suspect_fraction
    )


@dataclass(frozen=True)
class ReferenceGrid:
    """A fixed equirectangular composite: each cell's winning candidate is
    decided exactly once, so every viewpoint that samples it sees the same
    colour for the same physical point -- see :func:`sample_reference_grid`.
    """

    rgb: np.ndarray  # (height, width, 3) uint8
    has_data: np.ndarray  # (height, width) bool
    width: int
    height: int


def build_reference_grid(
    width: int,
    height: int,
    candidates: list[Candidate],
    *,
    scorer: CloudScorer,
    min_cos: float = DEFAULT_MIN_COS,
    luminance_floor: int = 30,
    obliqueness_penalty: float = 0.5,
    blend_k: int = 1,
    blend_margin: float = 1.5,
) -> ReferenceGrid:
    """Composite every candidate onto one lat/lon grid, once.

    Row 0 is the north pole, row ``height - 1`` the south pole; column 0 is
    -180 degrees longitude, wrapping up to (not including) +180. Every cell
    is sampled at its centre. This is the one-time decision every rotation
    frame later samples from (:func:`sample_reference_grid`), which is what
    makes a physical point's colour stop flickering between frames -- it was
    only ever decided once, not independently re-decided per viewpoint.
    """
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    lat = 90.0 - (rows + 0.5) / height * 180.0
    lon = -180.0 + (cols + 0.5) / width * 360.0

    rgb, has_data, _contributor = _composite_at(
        lat,
        lon,
        candidates,
        scorer=scorer,
        min_cos=min_cos,
        luminance_floor=luminance_floor,
        obliqueness_penalty=obliqueness_penalty,
        blend_k=blend_k,
        blend_margin=blend_margin,
    )
    return ReferenceGrid(
        rgb=rgb.round().clip(0, 255).astype(np.uint8), has_data=has_data, width=width, height=height
    )


def sample_reference_grid(
    grid: ReferenceGrid, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-sample a :class:`ReferenceGrid` at arbitrary ``(lat, lon)`` points.

    Longitude wraps (a viewpoint straddling +/-180 degrees samples across the
    seam correctly, via a one-column wraparound pad rather than
    ``map_coordinates``'s per-axis ``mode``, which this scipy build doesn't
    accept); latitude does not (there is no wraparound at the poles --
    ``mode="nearest"`` just holds the nearest row, which is what "off the top
    of the grid" should do anyway).

    Known wart: ``rgb`` is interpolated bilinearly but ``has_data`` is sampled
    nearest, so a point whose four surrounding cells are not all covered gets
    a colour pulled toward the ``(0, 0, 0)`` stored in the empty ones while
    still reading as opaque -- a one-cell dark fringe around a coverage hole.
    Only visible at all because coverage is never quite 100% (97.9% on the
    shipped run); the fix is to fill empty cells from their nearest covered
    neighbour before sampling.
    """
    row = (90.0 - lat) / 180.0 * grid.height - 0.5
    col = (lon + 180.0) / 360.0 * grid.width - 0.5
    col = np.mod(col, grid.width)  # into [0, width) so the ghost column below covers the seam
    coords = np.stack([row, col])

    padded_rgb = np.concatenate([grid.rgb, grid.rgb[:, :1]], axis=1)  # column width == column 0
    padded_has_data = np.concatenate([grid.has_data, grid.has_data[:, :1]], axis=1)

    sampled_rgb = np.stack(
        [
            ndimage.map_coordinates(
                padded_rgb[..., c].astype(np.float32), coords, order=1, mode="nearest"
            )
            for c in range(3)
        ],
        axis=-1,
    )
    sampled_has_data = (
        ndimage.map_coordinates(
            padded_has_data.astype(np.float32), coords, order=0, mode="nearest"
        )
        > 0.5
    )
    return sampled_rgb, sampled_has_data


def render_viewpoint_from_reference(viewpoint: Viewpoint, grid: ReferenceGrid) -> RenderResult:
    """Render one output disc by sampling a pre-built :class:`ReferenceGrid`.

    Unlike :func:`render_viewpoint`, this makes *no new decisions* -- every
    physical point's winning candidate was already decided once when the
    grid was built, so any two viewpoints covering the same point render it
    identically. This is what a temporally-stable rotation should use;
    :func:`render_viewpoint` remains for one-off renders where a persistent
    grid isn't worth building.
    """
    height, width = viewpoint.size
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    lat, lon, on_disc = inverse(cols, rows, geometry=viewpoint.geometry)

    rgb, has_data = sample_reference_grid(grid, lat, lon)

    suspect = on_disc & ~has_data
    alpha = np.where(on_disc & has_data, 255, 0).astype(np.uint8)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.round().clip(0, 255).astype(np.uint8)
    rgba[..., 3] = alpha

    on_disc_area = float(on_disc.sum())
    suspect_fraction = float(suspect.sum()) / on_disc_area if on_disc_area > 0 else 0.0

    return RenderResult(
        rgba=rgba,
        suspect=suspect,
        contributor=np.full((height, width), -1, dtype=np.int16),
        suspect_fraction=suspect_fraction,
    )
