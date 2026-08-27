"""Pixel-value segmentation of a bright disc (the Earth) against dark space.

No UI, no I/O. Everything here is a pure function on numpy arrays so the
algorithm can be tested and swapped independently of the app shell.

Pipeline
--------
1. luminance      RGB -> single channel
2. gaussian blur  suppress sensor noise / JPEG ringing before thresholding
3. threshold      Otsu by default, manual override available
4. largest blob   drops stars, the moon, lens flare, caption text
5. fill holes     recovers the night side and dark ocean interior
6. edge adjust    erode away the JPEG halo, or dilate to keep the atmosphere
7. bbox           tight crop box, optional padding
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Rec. 709 luma weights.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# A disc fills pi/4 of its bounding box. Deviation from this is the cheapest
# signal that a mask is a clipped crescent or has leaked into the background.
DISC_FILL_RATIO = float(np.pi / 4)

# 3x3 all-ones neighbourhood; iterated it approximates a disc better than the
# default 4-connected cross.
_STRUCT = ndimage.generate_binary_structure(2, 2)


@dataclass(frozen=True)
class SegmentResult:
    """Mask plus the numbers needed to judge whether the mask is any good."""

    mask: np.ndarray  # bool, (H, W)
    threshold: int  # luminance cutoff actually used, 0-255
    bbox: tuple[int, int, int, int] | None  # left, top, right, bottom (excl.)
    tight_bbox: tuple[int, int, int, int] | None  # bbox before padding
    coverage: float  # fraction of the frame inside the mask
    fill_ratio: float  # mask area / *unpadded* bbox area; pi/4 for a full disc

    @property
    def is_empty(self) -> bool:
        return self.bbox is None

    @property
    def aspect_ratio(self) -> float:
        """Width / height of the unpadded box. 1.0 for a full disc."""
        if self.tight_bbox is None:
            return 0.0
        left, top, right, bottom = self.tight_bbox
        return (right - left) / max(1, bottom - top)

    @property
    def disc_deviation(self) -> float:
        """How far the mask is from a full disc, as a relative error.

        Two independent signals, because either alone is fooled:

        * ``fill_ratio`` catches a mask that leaked into the background, but
          barely moves for a clipped crescent -- a 35% terminator shifts it
          only ~4%;
        * ``aspect_ratio`` catches exactly that clipping, since a disc with a
          slice missing stops being square.
        """
        return max(
            abs(self.fill_ratio - DISC_FILL_RATIO) / DISC_FILL_RATIO,
            abs(self.aspect_ratio - 1.0),
        )

    def looks_like_disc(self, tolerance: float = 0.12) -> bool:
        """Whether the mask plausibly is a full disc, and so worth trusting."""
        return not self.is_empty and self.disc_deviation < tolerance


def luminance(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3|4) uint8 -> (H, W) uint8. Alpha, if present, is ignored."""
    if rgb.ndim == 2:
        return rgb
    return (rgb[..., :3].astype(np.float32) @ _LUMA).round().clip(0, 255).astype(np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    """Threshold maximising between-class variance of the luminance histogram.

    Every cutoff inside an empty histogram gap scores identically, and a clean
    space photo has a very wide gap. ``argmax`` would return the low end of it,
    parking the threshold on the background noise floor; the midpoint of the
    tied plateau sits in the middle of the gap instead.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0

    levels = np.arange(256, dtype=np.float64)
    weight_fg = np.cumsum(hist) / total  # omega(t)
    mean_fg = np.cumsum(hist * levels) / total  # mu(t)
    mean_total = mean_fg[-1]

    spread = weight_fg * (1.0 - weight_fg)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(spread > 0, (mean_total * weight_fg - mean_fg) ** 2 / spread, 0.0)

    best = np.flatnonzero(between >= between.max() - 1e-12)
    return int(round(float(best.mean())))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest 8-connected blob."""
    labels, count = ndimage.label(mask, structure=_STRUCT)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    return labels == int(np.argmax(sizes))


def bounding_box(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    """Tight box around the True pixels, padded and clipped to the frame."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    height, width = mask.shape
    left = max(0, int(cols[0]) - pad)
    top = max(0, int(rows[0]) - pad)
    right = min(width, int(cols[-1]) + 1 + pad)
    bottom = min(height, int(rows[-1]) + 1 + pad)
    return left, top, right, bottom


def segment(
    rgb: np.ndarray,
    *,
    threshold: int | None = None,
    blur_sigma: float = 1.0,
    edge_adjust: int = 0,
    fill_holes: bool = True,
    keep_largest: bool = True,
    pad: int = 0,
    min_area: float = 0.0,
) -> SegmentResult:
    """Segment the bright disc out of a dark background.

    Args:
        rgb: (H, W, 3) or (H, W, 4) uint8 image.
        threshold: manual luminance cutoff; ``None`` runs Otsu.
        blur_sigma: gaussian pre-blur in pixels; 0 disables it.
        edge_adjust: positive dilates the mask, negative erodes it.
        fill_holes: fill interior gaps (the night side, dark ocean).
        keep_largest: discard everything but the biggest blob (stars, moon).
        pad: extra pixels around the reported bounding box.
        min_area: reject masks covering less than this fraction of the frame.
            Otsu always finds *something* -- on an empty frame it latches onto
            a hot pixel or a caption dot. Unattended runs want a floor here.
    """
    gray = luminance(rgb)
    smoothed = ndimage.gaussian_filter(gray, blur_sigma) if blur_sigma > 0 else gray

    cutoff = otsu_threshold(smoothed) if threshold is None else int(threshold)
    mask = smoothed > cutoff

    if keep_largest:
        mask = largest_component(mask)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if edge_adjust > 0:
        mask = ndimage.binary_dilation(mask, structure=_STRUCT, iterations=edge_adjust)
    elif edge_adjust < 0:
        mask = ndimage.binary_erosion(mask, structure=_STRUCT, iterations=-edge_adjust)
        # Erosion can split the blob or wipe it out entirely.
        if keep_largest and mask.any():
            mask = largest_component(mask)

    area = float(mask.sum())
    if min_area > 0 and area < min_area * mask.size:
        mask = np.zeros_like(mask)
        area = 0.0

    # Fill ratio and aspect ratio are measured against the *tight* box: padding
    # would otherwise dilute them and break the is-this-a-disc check.
    tight = bounding_box(mask)
    box = tight if pad == 0 else bounding_box(mask, pad=pad)
    if tight is None:
        fill_ratio = 0.0
    else:
        left, top, right, bottom = tight
        fill_ratio = area / max(1, (right - left) * (bottom - top))

    return SegmentResult(
        mask=mask,
        threshold=cutoff,
        bbox=box,
        tight_bbox=tight,
        coverage=area / mask.size,
        fill_ratio=fill_ratio,
    )


def cutout(rgb: np.ndarray, result: SegmentResult) -> np.ndarray | None:
    """Crop to the bbox and make everything outside the mask transparent."""
    if result.bbox is None:
        return None
    left, top, right, bottom = result.bbox
    patch = rgb[top:bottom, left:right, :3]
    alpha = result.mask[top:bottom, left:right]
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = patch
    rgba[..., 3] = alpha * 255
    # Zero the RGB of transparent pixels so viewers that ignore alpha, and
    # downstream resizes that bleed colour across the edge, stay clean.
    rgba[~alpha, :3] = 0
    return rgba


def overlay(
    rgb: np.ndarray,
    result: SegmentResult,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    dim: float = 0.45,
) -> np.ndarray:
    """Original image with the background dimmed and the mask outline drawn.

    This is the diagnostic view: a mask that clips the terminator or eats the
    limb is obvious here and invisible in the cutout alone.
    """
    out = rgb[..., :3].astype(np.float32)
    mask = result.mask
    out[~mask] *= 1.0 - dim

    if mask.any():
        outline = mask & ~ndimage.binary_erosion(mask, structure=_STRUCT, iterations=2)
        out[outline] = color

    return out.round().clip(0, 255).astype(np.uint8)
