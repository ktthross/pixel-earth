"""Orthographic projection between a sphere and a disc image.

No UI, no I/O. Everything here is a pure function on numbers or numpy arrays,
tested independently of image decoding or the EPIC archive.

DSCOVR sits at the Earth-Sun L1 point, about 1.46 million km out. Earth
subtends roughly 0.24 degrees from there, so the rays reaching the camera are
parallel to a very good approximation -- an orthographic (parallel-projection)
model is essentially exact, unlike a full perspective camera model which would
need the exact observer distance.

Every frame -- whether a real EPIC photo or a synthetic output viewpoint -- is
described by five numbers: the sub-satellite point it is centred on
(``lat0``, ``lon0``, degrees) and the disc's location in its own pixel grid
(``cx``, ``cy``, ``radius``, pixels). :func:`forward` turns a point on the
sphere into that frame's pixel coordinates (plus how obliquely the frame sees
it); :func:`inverse` turns one of that frame's pixels back into a point on the
sphere. Reprojecting frame A's pixels into frame B's geometry is
``forward(*inverse(col_b, row_b, geometry=B), geometry=A)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# A point past this cosine-of-angular-distance from a frame's own
# sub-satellite point is over the limb, or too oblique to trust: shared by the
# candidate prefilter (pixel_earth.catalog) and the per-pixel visibility gate
# (pixel_earth.mosaic), so both agree on what "visible" means. Measured by the
# prior compositing work as a reasonable obliqueness cutoff.
DEFAULT_MIN_COS = 0.35


def cap_radius_deg(min_cos: float = DEFAULT_MIN_COS) -> float:
    """Angular radius, in degrees, of the visibility cap for ``min_cos``."""
    return float(np.degrees(np.arccos(min_cos)))


@dataclass(frozen=True)
class Geometry:
    """Where a disc sits in its own pixel grid, and what it is centred on."""

    lat0: float  # sub-satellite latitude, degrees
    lon0: float  # sub-satellite longitude, degrees
    cx: float  # disc centre column, pixels
    cy: float  # disc centre row, pixels
    radius: float  # disc radius, pixels


class Projected(NamedTuple):
    col: np.ndarray
    row: np.ndarray
    cos_c: np.ndarray  # cosine of angular distance from (lat0, lon0); <0 is the far hemisphere


class Unprojected(NamedTuple):
    lat: np.ndarray
    lon: np.ndarray
    visible: np.ndarray  # bool: within the unit disc, i.e. the near hemisphere


def great_circle_distance_deg(
    lat0: float, lon0: float, lat1: np.ndarray, lon1: np.ndarray
) -> np.ndarray:
    """Angular distance in degrees between (lat0, lon0) and (lat1, lon1)."""
    cos_c = _cos_angular_distance(lat0, lon0, lat1, lon1)
    return np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0)))


def _cos_angular_distance(
    lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return np.sin(lat0_r) * np.sin(lat_r) + np.cos(lat0_r) * np.cos(lat_r) * np.cos(
        lon_r - lon0_r
    )


def forward(lat: np.ndarray, lon: np.ndarray, *, geometry: Geometry) -> Projected:
    """(lat, lon) in degrees -> pixel (col, row) in ``geometry``'s own frame.

    ``cos_c`` is computed from the spherical formula, not recovered from the
    projected (x, y) -- (x, y) alone cannot distinguish the near hemisphere
    from the far one, since both project to the same disc. Callers should
    treat a point with ``cos_c`` below some visibility floor (0 at the very
    least; the real limb is oblique well before that) as not usable.
    """
    lat0_r, lon0_r = np.radians(geometry.lat0), np.radians(geometry.lon0)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dlon = lon_r - lon0_r

    cos_c = np.sin(lat0_r) * np.sin(lat_r) + np.cos(lat0_r) * np.cos(lat_r) * np.cos(dlon)
    x = np.cos(lat_r) * np.sin(dlon)
    y = np.cos(lat0_r) * np.sin(lat_r) - np.sin(lat0_r) * np.cos(lat_r) * np.cos(dlon)

    col = geometry.cx + geometry.radius * x
    row = geometry.cy - geometry.radius * y  # minus: images are north-up, row grows downward
    return Projected(col, row, cos_c)


def inverse(col: np.ndarray, row: np.ndarray, *, geometry: Geometry) -> Unprojected:
    """Pixel (col, row) in ``geometry``'s own frame -> (lat, lon) in degrees.

    Points outside the unit disc (``rho > 1``) have no sphere location;
    ``visible`` marks which inputs were actually on the disc. Where
    ``visible`` is False, ``lat``/``lon`` hold whatever the formula produced
    (typically NaN from the outer arcsin) and must not be used.
    """
    lat0_r, lon0_r = np.radians(geometry.lat0), np.radians(geometry.lon0)
    x = (np.asarray(col, dtype=np.float64) - geometry.cx) / geometry.radius
    y = (geometry.cy - np.asarray(row, dtype=np.float64)) / geometry.radius
    rho = np.hypot(x, y)
    visible = rho <= 1.0

    # rho == 0 (disc centre) would divide by zero in the y/rho, x/rho terms
    # below; substitute a harmless 1 there since sin_c is 0 anyway, so those
    # terms vanish regardless of what rho is replaced with.
    safe_rho = np.where(rho > 0, rho, 1.0)
    sin_c = np.clip(rho, 0.0, 1.0)
    cos_c = np.sqrt(np.clip(1.0 - sin_c**2, 0.0, 1.0))

    lat = np.arcsin(
        cos_c * np.sin(lat0_r) + (y / safe_rho) * sin_c * np.cos(lat0_r)
    )
    lon_r = lon0_r + np.arctan2(
        x * sin_c, safe_rho * np.cos(lat0_r) * cos_c - y * np.sin(lat0_r) * sin_c
    )
    lon = (np.degrees(lon_r) + 180.0) % 360.0 - 180.0  # normalise to (-180, 180]

    return Unprojected(np.degrees(lat), lon, visible)
