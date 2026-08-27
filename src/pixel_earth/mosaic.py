"""Compose one cloud-free disc from many already-decoded candidate frames.

No persistent global raster: a :class:`Viewpoint` is rendered by reprojecting
each candidate frame's own pixels directly into the viewpoint's geometry
(``inverse`` the viewpoint's pixels to a sphere point, ``forward`` that point
into each candidate's own geometry, sample) and choosing among them --
one source-to-target resampling per candidate, no round trip through an
intermediate grid.

Selection never synthesises a colour by combining channels from different
frames: the least-cloudy candidate's whole ``(R, G, B)`` triplet is copied
verbatim. Blending (``blend_k > 1``) is a weighted *average of whole
triplets* among near-tied candidates -- legitimate anti-aliasing across
samples of similar quality, never a per-channel mix.
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
    """Render one output disc by picking/blending the least-cloudy candidate pixels."""
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

    scores = np.empty((len(candidates), height, width), dtype=np.float32)
    rgb_stack = np.zeros((len(candidates), height, width, 3), dtype=np.float32)

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
        valid = on_disc & (cos_c >= min_cos) & in_bounds

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
    top_idx = order[:keep]  # (keep, H, W)
    top_scores = np.take_along_axis(scores, top_idx, axis=0)
    best_score = top_scores[0]

    threshold = best_score * blend_margin + 1e-6
    finite = np.isfinite(top_scores)
    within_margin = finite & (top_scores <= threshold[np.newaxis])
    weight = np.where(within_margin, 1.0 / np.maximum(top_scores, 1e-6), 0.0)
    weight_sum = weight.sum(axis=0)

    top_rgb = np.take_along_axis(
        rgb_stack, top_idx[..., np.newaxis].repeat(3, axis=-1), axis=0
    )  # (keep, H, W, 3)
    blended = (weight[..., np.newaxis] * top_rgb).sum(axis=0)

    has_data = weight_sum > 0
    output_rgb = np.zeros((height, width, 3), dtype=np.float32)
    output_rgb[has_data] = blended[has_data] / weight_sum[has_data, np.newaxis]

    suspect = on_disc & ~has_data
    alpha = np.where(on_disc & has_data, 255, 0).astype(np.uint8)
    contributor = np.where(on_disc & has_data, top_idx[0], -1).astype(np.int16)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = output_rgb.round().clip(0, 255).astype(np.uint8)
    rgba[..., 3] = alpha

    on_disc_area = float(on_disc.sum())
    suspect_fraction = float(suspect.sum()) / on_disc_area if on_disc_area > 0 else 0.0

    return RenderResult(
        rgba=rgba, suspect=suspect, contributor=contributor, suspect_fraction=suspect_fraction
    )
