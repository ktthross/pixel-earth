"""Batch cutouts: point at a directory, get a hashed run folder in the repo.

The run folder name is a short hash of the *source directory plus the
segmentation settings*, not a timestamp or a random value. Consequences, all
deliberate:

* rerunning the same command reuses the folder and skips work already done,
  so an interrupted batch resumes;
* changing any setting writes a new folder, so two settings can be compared
  side by side instead of clobbering each other.

Layout::

    outputs/<run_id>/
        manifest.json      settings, per-image stats, which images need review
        cutouts/<mirrored input path>.png
        overlays/<mirrored input path>.png
    outputs/latest -> <run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from pixel_earth.segment import DISC_FILL_RATIO, cutout, overlay, segment

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
)

# Directory names we create ourselves; never treat them as input.
_OUTPUT_DIR_NAME = "outputs"


@dataclass(frozen=True)
class Settings:
    """Segmentation settings for a run. Frozen so it can be hashed."""

    threshold: int | None = None  # None -> Otsu
    blur_sigma: float = 1.0
    edge_adjust: int = 0
    fill_holes: bool = True
    keep_largest: bool = True
    pad: int = 0
    # Unattended runs need a floor: Otsu latches onto a hot pixel on an empty
    # frame, and a 3x3 "cutout" is worse than an honest miss.
    min_area: float = 0.001

    def as_kwargs(self) -> dict:
        return asdict(self)


@dataclass
class ImageOutcome:
    """What happened to one input image."""

    input: str  # path relative to the source directory
    status: str  # "ok" | "empty" | "skipped" | "failed"
    output: str | None = None  # path relative to the run directory
    threshold: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    coverage: float | None = None
    fill_ratio: float | None = None
    aspect_ratio: float | None = None
    needs_review: bool = False
    error: str | None = None


@dataclass
class RunReport:
    run_id: str
    run_dir: Path
    source: Path
    settings: Settings
    outcomes: list[ImageOutcome]

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def needs_review(self) -> list[ImageOutcome]:
        return [o for o in self.outcomes if o.needs_review]


def run_id(source: Path, settings: Settings, *, recursive: bool) -> str:
    """Short stable hash of everything that determines a run's output."""
    payload = json.dumps(
        {
            "source": str(source.resolve()),
            "recursive": recursive,
            "settings": settings.as_kwargs(),
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode(), digest_size=4).hexdigest()


def find_repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor containing .git, else the starting directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def find_images(source: Path, *, recursive: bool) -> list[Path]:
    """Image files under ``source``, sorted, excluding our own output folders."""
    paths = source.rglob("*") if recursive else source.glob("*")
    found = [
        path
        for path in paths
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and _OUTPUT_DIR_NAME not in path.relative_to(source).parts
        and not path.name.startswith(".")
    ]
    return sorted(found)


def load_rgb(path: Path) -> np.ndarray:
    """Read an image as (H, W, 3) uint8, honouring EXIF orientation.

    Phone and camera photos carry rotation in EXIF rather than in the pixel
    data; without the transpose the bounding box would be for the wrong
    orientation.
    """
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def process_image(
    path: Path,
    *,
    relative: Path,
    run_dir: Path,
    settings: Settings,
    write_overlay: bool,
    force: bool,
    dry_run: bool,
) -> ImageOutcome:
    """Segment one image and write its cutout (and overlay) into ``run_dir``."""
    cutout_path = run_dir / "cutouts" / relative.with_suffix(".png")
    rel_out = str(cutout_path.relative_to(run_dir))

    if cutout_path.exists() and not force:
        return ImageOutcome(str(relative), "skipped", output=rel_out)

    try:
        rgb = load_rgb(path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageOutcome(str(relative), "failed", error=f"{type(exc).__name__}: {exc}")

    result = segment(rgb, **settings.as_kwargs())
    if result.is_empty:
        return ImageOutcome(
            str(relative),
            "empty",
            threshold=result.threshold,
            coverage=result.coverage,
            needs_review=True,
        )

    outcome = ImageOutcome(
        str(relative),
        "ok",
        output=rel_out,
        threshold=result.threshold,
        bbox=result.bbox,
        coverage=result.coverage,
        fill_ratio=result.fill_ratio,
        aspect_ratio=result.aspect_ratio,
        needs_review=not result.looks_like_disc(),
    )
    if dry_run:
        outcome.status = "dry-run"
        return outcome

    cutout_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cutout(rgb, result), "RGBA").save(cutout_path)

    if write_overlay:
        overlay_path = run_dir / "overlays" / relative.with_suffix(".png")
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay(rgb, result), "RGB").save(overlay_path)

    return outcome


def write_manifest(report: RunReport, *, images_found: int) -> Path:
    manifest = {
        "run_id": report.run_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(report.source.resolve()),
        "settings": report.settings.as_kwargs(),
        "disc_fill_ratio": DISC_FILL_RATIO,
        "counts": {
            "found": images_found,
            "ok": report.count("ok"),
            "skipped": report.count("skipped"),
            "empty": report.count("empty"),
            "failed": report.count("failed"),
            "needs_review": len(report.needs_review),
        },
        "images": [asdict(o) for o in report.outcomes],
    }
    path = report.run_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def link_latest(run_dir: Path) -> None:
    """Point ``outputs/latest`` at this run. Best effort; never fatal."""
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        pass


def process_directory(
    source: Path,
    *,
    out_root: Path,
    settings: Settings,
    recursive: bool = False,
    write_overlays: bool = True,
    force: bool = False,
    dry_run: bool = False,
    on_progress=None,
) -> RunReport:
    """Segment every image under ``source`` into a hashed run folder."""
    if not source.is_dir():
        raise NotADirectoryError(source)

    identifier = run_id(source, settings, recursive=recursive)
    run_dir = out_root / identifier
    images = find_images(source, recursive=recursive)

    outcomes = []
    for index, path in enumerate(images, start=1):
        outcome = process_image(
            path,
            relative=path.relative_to(source),
            run_dir=run_dir,
            settings=settings,
            write_overlay=write_overlays,
            force=force,
            dry_run=dry_run,
        )
        outcomes.append(outcome)
        if on_progress is not None:
            on_progress(index, len(images), outcome)

    report = RunReport(identifier, run_dir, source, settings, outcomes)
    if not dry_run:
        write_manifest(report, images_found=len(images))
        link_latest(run_dir)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel-earth-batch",
        description="Cut the Earth out of every image in a directory.",
    )
    parser.add_argument("source", type=Path, help="directory of input images")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help=f"output root (default: <repo root>/{_OUTPUT_DIR_NAME})",
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="recurse into subdirectories")
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=None,
        help="manual luminance cutoff 0-255 (default: Otsu per image)",
    )
    parser.add_argument("--blur", type=float, default=1.0, metavar="SIGMA", help="pre-blur sigma in px")
    parser.add_argument(
        "--edge-adjust", type=int, default=0, metavar="PX", help="negative erodes, positive dilates"
    )
    parser.add_argument("--pad", type=int, default=0, metavar="PX", help="padding around the crop box")
    parser.add_argument(
        "--min-area",
        type=float,
        default=Settings.min_area,
        metavar="FRAC",
        help="reject masks smaller than this fraction of the frame (default: %(default)s)",
    )
    parser.add_argument("--no-fill-holes", action="store_true", help="leave interior dark regions out")
    parser.add_argument("--all-blobs", action="store_true", help="keep every blob, not just the largest")
    parser.add_argument("--no-overlays", action="store_true", help="skip the QC overlay images")
    parser.add_argument("--force", action="store_true", help="redo images already written")
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    return parser


def _review_reason(outcome: ImageOutcome) -> str:
    """Name the signal that actually flagged this image, not just the first one."""
    if outcome.error:
        return outcome.error
    if outcome.status == "empty":
        return "no object found"

    fill_error = abs((outcome.fill_ratio or 0.0) - DISC_FILL_RATIO) / DISC_FILL_RATIO
    aspect_error = abs((outcome.aspect_ratio or 0.0) - 1.0)
    if aspect_error >= fill_error:
        return f"box is not square (aspect {outcome.aspect_ratio:.2f}) — clipped disc?"
    return f"fill ratio {outcome.fill_ratio:.3f}, want {DISC_FILL_RATIO:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    settings = Settings(
        threshold=args.threshold,
        blur_sigma=args.blur,
        edge_adjust=args.edge_adjust,
        fill_holes=not args.no_fill_holes,
        keep_largest=not args.all_blobs,
        pad=args.pad,
        min_area=args.min_area,
    )
    out_root = args.out or (find_repo_root() / _OUTPUT_DIR_NAME)

    def progress(index: int, total: int, outcome: ImageOutcome) -> None:
        flag = " REVIEW" if outcome.needs_review else ""
        print(f"[{index}/{total}] {outcome.status:<8} {outcome.input}{flag}")

    try:
        report = process_directory(
            args.source,
            out_root=out_root,
            settings=settings,
            recursive=args.recursive,
            write_overlays=not args.no_overlays,
            force=args.force,
            dry_run=args.dry_run,
            on_progress=progress,
        )
    except NotADirectoryError as exc:
        print(f"not a directory: {exc}")
        return 2

    if not report.outcomes:
        print(f"no images found in {args.source}")
        return 1

    print(f"\nrun {report.run_id} -> {report.run_dir}")
    print(
        f"  ok {report.count('ok')}"
        f"  skipped {report.count('skipped')}"
        f"  empty {report.count('empty')}"
        f"  failed {report.count('failed')}"
    )
    if report.needs_review:
        print(f"  {len(report.needs_review)} need review (mask is not a full disc):")
        for outcome in report.needs_review[:10]:
            print(f"    {outcome.input}  {_review_reason(outcome)}")
        if len(report.needs_review) > 10:
            print(f"    ... and {len(report.needs_review) - 10} more, see manifest.json")

    return 1 if report.count("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
