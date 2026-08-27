"""Turn a photo-real disc into pixel art: grade, downsample, quantize.

No UI, no I/O. Every step is a pure function on numpy arrays, in the order
:func:`pixelate` applies them:

1. :func:`grade`        colour-grade toward a punchier "expected Earth" look
2. :func:`downsample_rgba`  nearest-neighbour downsample to the working grid
3. :func:`quantize_palette` snap to a small, fixed-size palette
4. :func:`upscale_nearest`  nearest-neighbour blow-up for viewing

Every resampling step (2 and 4) is nearest-neighbour on purpose: no
averaging means every output pixel is a colour that was actually in the
source, and a boundary between two regions is one hard step rather than a
many-pixel gradient -- the bold, blocky look pixel art is supposed to have,
not a shrunk-down photo.

Grading happens on the full-resolution image, before downsampling -- the
saturation/contrast curves are local per-pixel operations, and running them
at native resolution avoids compounding two different kinds of averaging
(box-downsample, then a nonlinear curve) into a muddier result than doing
either alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# ------------------------------------------------------------------- colour


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(..., 3) uint8 -> (h, s, v), each float32 (...) in [0, 1]."""
    arr = rgb[..., :3].astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    delta = maxc - minc

    v = maxc
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1.0), 0.0)

    safe_delta = np.where(delta > 0, delta, 1.0)
    hue = np.select(
        [maxc == r, maxc == g, maxc == b],
        [
            ((g - b) / safe_delta) % 6.0,
            (b - r) / safe_delta + 2.0,
            (r - g) / safe_delta + 4.0,
        ],
        default=0.0,
    )
    h = np.where(delta > 0, hue / 6.0, 0.0) % 1.0
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(h, s, v) float in [0, 1] -> (..., 3) float32 in [0, 255]."""
    h6 = (np.asarray(h, dtype=np.float32) % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)

    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1) * 255.0


def _hue_delta(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest signed distance from hue ``a`` to hue ``b`` on the hue circle."""
    return (b - a + 0.5) % 1.0 - 0.5


def grade(
    rgb: np.ndarray,
    amount: float,
    *,
    saturation_boost: float = 1.8,
    gamma: float = 2.4,
    contrast: float = 0.35,
    black_point: float = 0.05,
    land_green: float = 1.0,
    land_hue_center: float = 35.0 / 360.0,
    land_hue_width: float = 35.0 / 360.0,
    land_hue_rotation: float = 90.0 / 360.0,
    land_min_saturation: float = 0.12,
    land_bright_low: float = 0.28,
    land_bright_high: float = 0.55,
) -> np.ndarray:
    """Blend ``rgb`` toward a punchier, "expected Earth" version of itself.

    ``amount`` is 0 (untouched -- the true, muted colour the camera actually
    saw) to 1 (fully graded).

    A real EPIC frame reads as muted rather than the vivid "blue marble"
    people expect mostly because it is genuinely **dark** -- deep ocean well
    off the glint point measures a median brightness around 0.3, not because
    it is washed out toward grey (measured median saturation is a fairly
    normal ~0.4). So the main lever is a **gamma lift** on brightness
    (``v ** (1/gamma)``), which brightens shadows and midtones much more than
    highlights. A small black-point stretch before that removes a thin grey
    haze veil, and a **contrast** S-curve *after* the gamma lift spreads the
    now-brighter tones back out around the midpoint -- this is what actually
    separates ocean from land from cloud into distinct-looking bands rather
    than one undifferentiated bright wash, i.e. the "pop." A *vibrance* curve
    on saturation (``1 - (1-s)**(1+boost)``) makes the colour read as colour
    on top of that, moving muted pixels much more than already-vivid or
    already-neutral ones (bright white cloud stays white, not tinted).

    None of that can make land look *green*, though: measured on a real
    frame, land pixels cluster almost entirely between 20-50 degrees of hue
    (tan/brown/orange -- deserts and dry-season vegetation dominate what
    DSCOVR actually sees), with next to nothing in the 90-150 degree range
    people picture when they think "green continents". Saturating or
    brightening a hue that was never there cannot produce it. So this is the
    one deliberately *not* accurate step: land-toned pixels (picked by hue
    alone, in a band around ``land_hue_center``, and gated on saturation so
    cloud/ice/night are never touched) get rotated by up to
    ``land_hue_rotation`` toward green, weighted by how central their hue is
    to that band. Rotating rather than snapping every land pixel to one
    target hue keeps the desert-to-forest tonal *variation* that makes it
    read as texture instead of a flat green fill.

    That rotation is also weighted down by brightness (the pixel's *original*
    value, before the gamma lift above): real deserts -- bare sand and rock --
    reflect far more light than any vegetation, so a land pixel brighter than
    ``land_bright_high`` is far more likely to be Sahara-type desert than
    forest, and keeps most of its true tan/orange rather than being rotated
    to match the same green as genuinely darker, plausibly-vegetated land
    below ``land_bright_low``. Rotating purely by hue, with no brightness
    term, over-greened deserts specifically -- they *are* the biggest, most
    saturated cluster in that hue band, so they picked up the most rotation
    of anything on the disc, exactly backwards from how a desert should read.

    The thresholds themselves are set from the actual distribution: measured
    on a real frame, land-hued pixels have a median brightness around 0.41 and
    a 90th percentile around 0.57, so a naive "midtone" cutoff (e.g. 0.42)
    excludes most land, not just deserts -- ``land_bright_high`` sits closer
    to the 75th percentile instead, so only the brightest quarter (the actual
    desert/outback cluster) is held back.
    """
    if amount <= 0:
        return rgb[..., :3].astype(np.uint8)

    h, s, v = rgb_to_hsv(rgb)

    graded_s = np.clip(1.0 - (1.0 - s) ** (1.0 + saturation_boost), 0.0, 1.0)

    lifted_v = np.clip((v - black_point) / max(1e-6, 1.0 - black_point), 0.0, 1.0)
    brightened_v = lifted_v ** (1.0 / max(1e-6, gamma))
    graded_v = np.clip(0.5 + (brightened_v - 0.5) * (1.0 + contrast), 0.0, 1.0)

    hue_distance = np.abs(_hue_delta(h, land_hue_center))
    hue_weight = np.clip(1.0 - hue_distance / max(1e-6, land_hue_width), 0.0, 1.0)
    saturation_weight = np.clip((s - land_min_saturation) / 0.15, 0.0, 1.0)
    brightness_weight = np.clip(
        (land_bright_high - v) / max(1e-6, land_bright_high - land_bright_low), 0.0, 1.0
    )
    land_weight = hue_weight * saturation_weight * brightness_weight * land_green
    graded_h = (h + land_hue_rotation * land_weight) % 1.0

    graded_rgb = hsv_to_rgb(graded_h, graded_s, graded_v)
    original_rgb = rgb[..., :3].astype(np.float32)
    blended = original_rgb * (1.0 - amount) + graded_rgb * amount
    return blended.round().clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------- resampling


def _block_reduce_rgba(rgba: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping ``factor`` x ``factor`` blocks, alpha-aware.

    Unlike a full-cell box filter, ``factor`` is small and fixed (a handful
    of samples) regardless of how far the overall image is being shrunk --
    this is a light anti-alias for a handful of raw nearest-neighbour
    samples, not a shrink-the-whole-photo blur.
    """
    height, width = rgba.shape[:2]
    out_h, out_w = height // factor, width // factor
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32)
    premultiplied = rgb * (alpha / 255.0)

    premult_blocks = premultiplied.reshape(out_h, factor, out_w, factor, 3).mean(axis=(1, 3))
    alpha_blocks = alpha.reshape(out_h, factor, out_w, factor, 1).mean(axis=(1, 3))

    safe_alpha = np.where(alpha_blocks > 0, alpha_blocks, 1.0)
    rgb_out = np.where(alpha_blocks > 0, premult_blocks / safe_alpha * 255.0, 0.0)

    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    out[..., :3] = rgb_out.round().clip(0, 255).astype(np.uint8)
    out[..., 3] = alpha_blocks[..., 0].round().clip(0, 255).astype(np.uint8)
    return out


def downsample_rgba(
    rgba: np.ndarray, size: int, *, method: str = "nearest", supersample: int = 1
) -> np.ndarray:
    """Downsample to a ``size`` x ``size`` grid.

    ``method="nearest"`` (the default) picks one source pixel per output
    cell -- no averaging, so every output pixel is a colour that actually
    existed in the source, and edges between regions land as one hard step,
    not a many-pixel-wide gradient. This is what gives pixel art its bold,
    graphic quality; a box/area average is the "photo shrunk down" look,
    which reads as fuzzy rather than deliberate at pixel-art resolutions.

    A single nearest-neighbour sample is also, on its own, why a *sequence*
    can flicker along hard edges (coastlines) even when the underlying
    source is itself perfectly stable frame to frame: each output cell
    covers many source pixels (dozens, at typical pixel-art sizes), and
    exactly which one lands on a coastline shifts by a fraction of a pixel
    as the viewpoint rotates -- enough to flip that one cell between
    confidently-ocean and confidently-land, a full-strength colour swap,
    every couple of frames. ``supersample > 1`` takes a small fixed grid of
    nearest-neighbour samples per output cell (e.g. 3x3 = 9) and averages
    just those -- interior cells are still one flat colour (all 9 samples
    agree), but a cell straddling a boundary now blends a few samples
    instead of aliasing between two extremes, and that blend moves smoothly
    as the boundary sweeps through instead of snapping.

    ``method="box"`` area-averages the *entire* cell instead (alpha-aware:
    premultiplied before resampling and un-premultiplied after, so a
    transparent neighbour can't darken an opaque edge pixel's colour) --
    the "shrunk-down photo" look, kept as an option if that's ever wanted.
    """
    if method == "nearest":
        working_size = size * max(1, supersample)
        img = Image.fromarray(rgba, mode="RGBA")
        big = np.asarray(img.resize((working_size, working_size), Image.Resampling.NEAREST))
        if supersample <= 1:
            return big
        return _block_reduce_rgba(big, supersample)
    if method != "box":
        raise ValueError(f"unknown downsample method: {method!r}")

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32)
    premultiplied = rgb * (alpha / 255.0)

    # PIL has no multi-channel float mode; resize each premultiplied channel
    # (and alpha) separately, all with the same box filter.
    channels = [
        Image.fromarray(premultiplied[..., c], mode="F").resize(
            (size, size), Image.Resampling.BOX
        )
        for c in range(3)
    ]
    alpha_small = Image.fromarray(alpha[..., 0], mode="F").resize(
        (size, size), Image.Resampling.BOX
    )

    small_premult = np.stack([np.asarray(c) for c in channels], axis=-1)
    small_alpha = np.asarray(alpha_small)

    safe_alpha = np.where(small_alpha > 0, small_alpha, 1.0)
    small_rgb = (small_premult / safe_alpha[..., np.newaxis]) * 255.0
    small_rgb = np.where(small_alpha[..., np.newaxis] > 0, small_rgb, 0.0)

    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = small_rgb.round().clip(0, 255).astype(np.uint8)
    out[..., 3] = small_alpha.round().clip(0, 255).astype(np.uint8)
    return out


def upscale_nearest(rgba: np.ndarray, factor: int) -> np.ndarray:
    """Blow up by an integer factor with nearest-neighbour -- keeps pixels
    looking like pixels instead of blurring them into a smooth gradient."""
    if factor <= 1:
        return rgba
    return np.repeat(np.repeat(rgba, factor, axis=0), factor, axis=1)


def display_scale_for(size: int, *, target: int = 512) -> int:
    """A nearest-neighbour blow-up factor that gets a tiny grid to a
    reasonable on-screen size, without shrinking anything already big."""
    return max(1, round(target / size))


# ---------------------------------------------------------------- palette


def quantize_palette(rgb: np.ndarray, *, colors: int = 32, dither: bool = False) -> np.ndarray:
    """Snap to a ``colors``-size palette via PIL's median-cut quantizer,
    discovered fresh from this image alone.

    Fine for a single still image; for a sequence, this is exactly what
    makes still frames flicker even once the underlying colour is stable --
    two frames showing slightly different parts of the globe discover
    slightly different palettes, so the same true colour can snap to a
    different swatch from one frame to the next. Use
    :func:`build_shared_palette` + :func:`quantize_to_palette` for a
    sequence instead.
    """
    img = Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB")
    quantized = img.quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE,
    )
    return np.asarray(quantized.convert("RGB"))


def build_shared_palette(rgb_images: list[np.ndarray], *, colors: int = 32) -> Image.Image:
    """One palette, discovered once from every image in ``rgb_images`` combined.

    Quantizing each frame of a sequence against this same palette (via
    :func:`quantize_to_palette`) instead of each discovering its own is what
    keeps a stable true colour mapped to the same final swatch in every
    frame -- removing exactly the quantization-level flicker
    :func:`quantize_palette` warns about.
    """
    pixels = np.concatenate([img.reshape(-1, 3) for img in rgb_images], axis=0)
    width = 1024
    height = -(-len(pixels) // width)  # ceil
    padded = np.zeros((height * width, 3), dtype=np.uint8)
    padded[: len(pixels)] = pixels
    combined = Image.fromarray(padded.reshape(height, width, 3), mode="RGB")
    return combined.quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )


def quantize_to_palette(
    rgb: np.ndarray, palette_image: Image.Image, *, dither: bool = False
) -> np.ndarray:
    """Snap to a palette built by :func:`build_shared_palette`, not a fresh one."""
    img = Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB")
    quantized = img.quantize(
        palette=palette_image, dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    )
    return np.asarray(quantized.convert("RGB"))


# --------------------------------------------------------------- orchestrator


@dataclass(frozen=True)
class PixelArtSettings:
    """Everything that determines one pixel-art rendering of a frame."""

    size: int = 64
    stylize: float = 1.0  # 0 = true colour, 1 = fully graded
    saturation_boost: float = 1.8
    gamma: float = 2.4
    contrast: float = 0.35
    black_point: float = 0.05
    land_green: float = 1.0  # 0 = true land hue, 1 = fully rotated toward green
    colors: int = 32
    dither: bool = False
    downsample_method: str = "nearest"  # "nearest" for bold/blocky, "box" for a softer average
    supersample: int = 8  # anti-alias for `method="nearest"`; see downsample_rgba
    display_scale: int | None = None  # None -> derived from `size`, see display_scale_for


def grade_and_downsample(rgba: np.ndarray, settings: PixelArtSettings) -> np.ndarray:
    """The two steps before quantization: colour-grade, then shrink to the
    working grid. Split out from :func:`pixelate` so a sequence can grade
    and downsample every frame first, build one shared palette from all of
    them (:func:`build_shared_palette`), and only then quantize each --
    see :mod:`pixel_earth.sprites`.
    """
    graded_rgb = grade(
        rgba[..., :3],
        settings.stylize,
        saturation_boost=settings.saturation_boost,
        gamma=settings.gamma,
        land_green=settings.land_green,
        contrast=settings.contrast,
        black_point=settings.black_point,
    )
    graded = np.dstack([graded_rgb, rgba[..., 3]])
    return downsample_rgba(
        graded, settings.size, method=settings.downsample_method, supersample=settings.supersample
    )


def finish(
    small: np.ndarray,
    settings: PixelArtSettings,
    *,
    palette_image: Image.Image | None = None,
) -> np.ndarray:
    """Quantize an already-graded-and-downsampled disc and blow it back up.

    ``palette_image``, when given (from :func:`build_shared_palette`), is
    used instead of discovering a fresh palette from ``small`` alone -- see
    :func:`quantize_palette`'s docstring for why that matters for a sequence.
    """
    if palette_image is not None:
        quantized_rgb = quantize_to_palette(small[..., :3], palette_image, dither=settings.dither)
    else:
        quantized_rgb = quantize_palette(
            small[..., :3], colors=settings.colors, dither=settings.dither
        )

    result = np.zeros_like(small)
    result[..., :3] = quantized_rgb
    result[..., 3] = small[..., 3]

    scale = settings.display_scale
    if scale is None:
        scale = display_scale_for(settings.size)
    return upscale_nearest(result, scale)


def pixelate(rgba: np.ndarray, settings: PixelArtSettings) -> np.ndarray:
    """Grade, downsample, and quantize one RGBA disc into pixel art.

    For a single still image. For a sequence, use :func:`grade_and_downsample`
    + :func:`build_shared_palette` + :func:`finish` instead, so every frame
    quantizes against the same palette -- see :mod:`pixel_earth.sprites`.
    """
    small = grade_and_downsample(rgba, settings)
    return finish(small, settings)
