"""Pluggable per-pixel cloudiness scoring.

No UI, no I/O. A score is a float array where 0 means "trust this pixel" and
larger means "less trustworthy" -- callers (:mod:`pixel_earth.mosaic`) use it
with ``argmin``, so scores from different scorers only need to be internally
consistent, not on a shared absolute scale.

``SCORERS`` is a name -> callable registry so the rest of the pipeline never
imports a specific scorer by name; :mod:`pixel_earth.turntable` resolves the
configured name through it. This is what makes it possible to swap in a
better cloud signal later without touching :mod:`pixel_earth.mosaic` at all.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CloudScorer(Protocol):
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """(..., 3) uint8 or float -> (...) float32, 0 = clear, larger = cloudier."""
        ...


def rgb_heuristic_score(rgb: np.ndarray) -> np.ndarray:
    """Cloud is bright and colourless; score = brightness * whiteness**2.

    Snow and sea ice are also bright and colourless, so this is a known,
    visible-light-only approximation -- it cannot distinguish the two. Squaring
    whiteness weights it more heavily than brightness alone, so a merely bright
    but still-saturated scene (a sunlit ocean glint, a desert) does not score as
    high as an actually white one.
    """
    channels = rgb[..., :3].astype(np.float32) / 255.0
    value = channels.max(axis=-1)
    chroma = value - channels.min(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturation = np.where(value > 0, chroma / np.where(value > 0, value, 1.0), 0.0)
    whiteness = 1.0 - saturation
    return (value * whiteness**2).astype(np.float32)


def cloudfraction_score(cloud_quicklook: np.ndarray) -> np.ndarray:
    """Score derived from NASA's ``epic_cloudfraction_`` quicklook image.

    Not implemented -- ``scripts/spike_cloudfraction.py`` ran and found a
    no-go, on both of its stated criteria:

    * **Not geometrically compatible.** The quicklook is 640x480, not
      2048x2048, and is a rendered *figure* -- a title, a coastline overlay,
      and a 4-swatch legend strip -- not a bare disc on black. The inset
      disc's fill ratio/aspect ratio inside that canvas varies run to run
      (measured 0.50-0.77 fill, 0.91-1.43 aspect across 6 sampled frames),
      because the legend's height isn't fixed. There is no single crop or
      scale factor that lines a quicklook up with its ``natural`` sibling.
    * **Categorical, not continuous.** It is a 4-class confidence map (High/
      Low Confidence Clear, Low/High Confidence Cloudy), flat-shaded from a
      legend, not a grayscale or continuous fraction -- confirmed both
      visually and by the couple-hundred distinct colours found inside the
      detected disc (legend swatches plus anti-aliased edges/coastlines/text,
      not a smooth ramp).

    That said, the underlying classification looks genuinely good -- the
    cloud/clear boundaries visibly track real cloud structure in the paired
    natural-colour frames (see the spike's contact sheet). A real
    implementation is plausible, just bigger than a plug-in scorer: detect
    and crop the inset disc out of the chrome (a tuned :func:`pixel_earth.segment.segment`
    could do this), classify each pixel by nearest legend colour, then
    reproject through :mod:`pixel_earth.geometry` same as any other frame.
    Worth a follow-up if the RGB heuristic's snow/ice confusion turns out to
    matter in practice; not attempted here since the data alignment work
    needed is out of scope for a "plug in a better score" spike.
    """
    raise NotImplementedError(
        "cloudfraction scoring is unimplemented -- scripts/spike_cloudfraction.py "
        "recorded a no-go (see this function's docstring); use scorer='rgb'"
    )


SCORERS: dict[str, CloudScorer] = {
    "rgb": rgb_heuristic_score,
    "cloudfraction": cloudfraction_score,
}
