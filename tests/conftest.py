"""Synthetic test images, shared by the segment and batch tests."""

from pathlib import Path

import numpy as np
from PIL import Image


def synthetic_earth(
    size: int = 200,
    center: tuple[int, int] = (100, 100),
    radius: int = 50,
    *,
    night_fraction: float = 0.0,
    dark_spot: int = 0,
    stars: int = 0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Bright disc on near-black space.

    ``dark_spot`` puts a dark blob strictly inside the disc (a dark ocean) --
    an enclosed hole. ``night_fraction`` darkens a slice that runs out to the
    limb -- an open notch, which is a different problem entirely.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size]
    cy, cx = center
    dist_sq = (rows - cy) ** 2 + (cols - cx) ** 2
    disc = dist_sq <= radius**2

    img = np.zeros((size, size, 3), dtype=np.float32)
    img[disc] = (60, 110, 190)

    if dark_spot > 0:
        img[dist_sq <= dark_spot**2] = (4, 5, 8)

    if night_fraction > 0:
        # Darken the right-hand slice of the disc to mimic the terminator.
        night = disc & (cols > cx + radius * (1 - 2 * night_fraction))
        img[night] = (4, 5, 8)

    for _ in range(stars):
        sy, sx = rng.integers(0, size, size=2)
        img[sy, sx] = (230, 230, 230)

    if noise > 0:
        img += rng.normal(0, noise, img.shape)

    return img.clip(0, 255).astype(np.uint8)


def write_earth(path: Path, **kwargs) -> Path:
    """Render a synthetic earth to disk, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(synthetic_earth(**kwargs), "RGB").save(path)
    return path


_CLOUD_COLOR = (250, 250, 250)


def synthetic_viewpoint_frame(
    size: tuple[int, int],
    cx: float,
    cy: float,
    radius: float,
    *,
    quadrant_colors: dict[str, tuple[int, int, int]],
    cloudy_quadrants: tuple[str, ...] = (),
) -> np.ndarray:
    """A disc split into NW/NE/SW/SE quadrants, each a flat colour.

    Models several frames of "the same view" where clouds happen to sit over
    a different quadrant in each -- exactly the case a compositor needs to
    recombine into one clear disc. Quadrants named in ``cloudy_quadrants`` are
    painted bright white instead of their assigned colour. Pixels outside the
    disc are black, as a real EPIC frame's background is.
    """
    rows, cols = np.mgrid[0 : size[0], 0 : size[1]]
    dist_sq = (rows - cy) ** 2 + (cols - cx) ** 2
    on_disc = dist_sq <= radius**2
    north = rows <= cy
    west = cols <= cx

    img = np.zeros((*size, 3), dtype=np.uint8)
    quadrants = {
        "NW": north & west,
        "NE": north & ~west,
        "SW": ~north & west,
        "SE": ~north & ~west,
    }
    for name, region in quadrants.items():
        mask = region & on_disc
        color = _CLOUD_COLOR if name in cloudy_quadrants else quadrant_colors[name]
        img[mask] = color
    return img
