"""Spike: is NASA's ``epic_cloudfraction_`` quicklook usable as a cloud mask?

Throwaway, not part of the ``pixel_earth`` package -- see the go/no-go
criteria in the project plan. Run after mirroring a handful of ``cloud``
frames that share a day with already-mirrored ``natural`` frames, e.g.::

    uv run pixel-earth-fetch --collection cloud --format png \\
        --date 2024-01-01 --date 2024-06-01 --date 2024-12-31 \\
        --date 2025-01-01 --date 2025-02-20 --max-bytes 3GiB
    uv run python scripts/spike_cloudfraction.py

Checks, cheapest first:

1. **Format/geometry** -- is the quicklook even the same shape of thing as a
   ``natural`` frame (a disc on black, same aspect ratio), or does it carry
   its own chrome (title, legend, coastline overlay) that would need
   detecting and stripping before any per-pixel comparison is possible?
2. **Encoding** -- how many distinct colours appear inside the disc? A
   continuous 0-1 cloud fraction rendered as grayscale would show close to
   256; a handful of exact, repeated colours means a categorical legend
   instead.
3. **The test that actually matters** -- a snow/ice frame (a Southern-summer
   day, when Antarctica is prominently lit) is where the RGB heuristic is
   known to false-positive. Does the quicklook visibly read differently over
   the ice cap than over actual cloud?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pixel_earth.batch import find_repo_root, load_rgb
from pixel_earth.segment import segment

REPO_ROOT = find_repo_root()
MIRROR = REPO_ROOT / "data" / "epic"
OUT_DIR = REPO_ROOT / "outputs" / "spike-cloudfraction"


def _matching_pairs() -> list[tuple[Path, Path]]:
    """Cloud quicklooks that share an exact timestamp with a mirrored natural frame."""
    pairs = []
    for cloud_path in sorted(MIRROR.glob("cloud/*/*/*/png/epic_cloudfraction_*.png")):
        stamp = cloud_path.stem.removeprefix("epic_cloudfraction_")
        year, month, day = stamp[:4], stamp[4:6], stamp[6:8]
        natural_path = MIRROR / "natural" / year / month / day / "png" / f"epic_1b_{stamp}.png"
        if natural_path.exists():
            pairs.append((cloud_path, natural_path))
    return pairs


def _describe(cloud_path: Path, natural_path: Path) -> None:
    cloud = load_rgb(cloud_path)
    natural = load_rgb(natural_path)
    print(f"\n{cloud_path.name}")
    print(f"  natural   {natural.shape[1]}x{natural.shape[0]}")
    print(f"  cloud     {cloud.shape[1]}x{cloud.shape[0]}")

    # 1. Format/geometry -- run the same disc detector used everywhere else.
    # A real disc fills pi/4 of its bounding box and is square (aspect ~1.0);
    # chrome (title text, a legend strip) around a smaller inset disc throws
    # both numbers off.
    result = segment(cloud)
    print(
        f"  disc detector on the quicklook: fill_ratio={result.fill_ratio:.3f} "
        f"(want ~0.785), aspect={result.aspect_ratio:.3f} (want ~1.0), "
        f"looks_like_disc={result.looks_like_disc()}"
    )

    # 2. Encoding -- count distinct colours inside whatever the detector found.
    if result.bbox is not None:
        left, top, right, bottom = result.bbox
        patch = cloud[top:bottom, left:right][result.mask[top:bottom, left:right]]
        distinct = np.unique(patch.reshape(-1, 3), axis=0)
        print(f"  distinct colours inside the detected region: {len(distinct)}")
        if len(distinct) <= 16:
            print(f"    {[tuple(c) for c in distinct[:16]]}")


def _contact_sheet(pairs: list[tuple[Path, Path]], out: Path) -> None:
    thumbs = []
    for cloud_path, natural_path in pairs:
        cloud = Image.open(cloud_path).convert("RGB").resize((320, 240))
        natural = Image.open(natural_path).convert("RGB").resize((320, 320))
        row = Image.new("RGB", (320, 320 + 240), (0, 0, 0))
        row.paste(natural, (0, 0))
        row.paste(cloud, (0, 320))
        thumbs.append(row)

    if not thumbs:
        return
    sheet = Image.new("RGB", (320 * len(thumbs), 320 + 240), (0, 0, 0))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (i * 320, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nwrote {out}  (top: natural, bottom: cloud quicklook)")


def main() -> None:
    pairs = _matching_pairs()
    print(f"{len(pairs)} natural/cloud pairs sharing an exact timestamp")
    if not pairs:
        print("nothing to check -- mirror some `cloud` frames first, see module docstring")
        return

    # A spread across the year, so at least one pair sits near Southern
    # summer (prominent, brightly-lit Antarctic ice -- the actual case this
    # spike exists to check).
    sample = pairs[:: max(1, len(pairs) // 6)][:6]
    for cloud_path, natural_path in sample:
        _describe(cloud_path, natural_path)

    _contact_sheet(sample, OUT_DIR / "contact-sheet.png")


if __name__ == "__main__":
    main()
