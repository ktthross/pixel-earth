"""Shared output writers: a grid contact sheet and a looping GIF built from a
list of RGBA frame PNGs, flattened onto a solid background.

Used by both :mod:`pixel_earth.turntable` and :mod:`pixel_earth.sprites` so
the two pipelines' review artifacts look and behave the same way.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


def flatten_on_background(
    path: Path, *, background: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Composite one RGBA frame onto a solid background (GIF has no true alpha)."""
    with Image.open(path) as opened:
        img = opened.convert("RGBA")
    canvas = Image.new("RGB", img.size, background)
    canvas.paste(img, mask=img.split()[3])
    return canvas


def write_contact_sheet(
    frame_paths: list[Path], out: Path, *, background: tuple[int, int, int] = (0, 0, 0)
) -> None:
    """Every frame tiled into one roughly-square grid, for a one-glance look."""
    flattened = [flatten_on_background(p, background=background) for p in frame_paths]
    if not flattened:
        return
    width, height = flattened[0].size
    columns = max(1, math.ceil(math.sqrt(len(flattened))))
    rows = math.ceil(len(flattened) / columns)

    sheet = Image.new("RGB", (columns * width, rows * height), background)
    for i, img in enumerate(flattened):
        x, y = (i % columns) * width, (i // columns) * height
        sheet.paste(img, (x, y))
    sheet.save(out)


def write_rotation_gif(
    frame_paths: list[Path],
    out: Path,
    *,
    duration_ms: int = 80,
    background: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """A looping GIF cycling through every frame in order."""
    flattened = [flatten_on_background(p, background=background) for p in frame_paths]
    if not flattened:
        return
    flattened[0].save(
        out, save_all=True, append_images=flattened[1:], duration=duration_ms, loop=0
    )
