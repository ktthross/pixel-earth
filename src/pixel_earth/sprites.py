"""Render a turntable rotation as pixel art, at one or more resolutions.

Takes the RGBA frames a :mod:`pixel_earth.turntable` run already produced
(or any directory of RGBA PNGs) and, for each requested size, grades +
downsamples + palette-quantizes every frame (:mod:`pixel_earth.pixelart`),
then writes the same kind of review artifacts turntable does
(:mod:`pixel_earth.sheets`), into a hashed run folder::

    outputs/pixelart-<run_id>/
        32px/frames/frame_000.png ... frame_<N-1>.png
        32px/sheet.png
        32px/rotation.gif
        64px/...
        manifest.json
    outputs/latest-pixelart -> pixelart-<run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from pixel_earth import sheets
from pixel_earth.batch import find_repo_root
from pixel_earth.pixelart import (
    PixelArtSettings,
    build_shared_palette,
    finish,
    grade_and_downsample,
)

_OUTPUT_DIR_NAME = "outputs"
_RUN_PREFIX = "pixelart-"


@dataclass(frozen=True)
class SpriteSettings:
    """Everything that determines a pixel-art run's output. Frozen so it can be hashed."""

    sizes: tuple[int, ...] = (16, 32, 64, 128)
    stylize: float = 1.0
    saturation_boost: float = 1.8
    gamma: float = 2.4
    contrast: float = 0.35
    black_point: float = 0.05
    land_green: float = 1.0
    colors: int = 32
    dither: bool = False
    downsample_method: str = "nearest"
    supersample: int = 8  # anti-alias against coastline-style flicker; see pixelart.downsample_rgba
    shared_palette: bool = True  # one palette across the whole sequence, not one per frame
    display_scale: int | None = None  # None -> derived per size, see pixelart.display_scale_for

    def as_dict(self) -> dict:
        return asdict(self)

    def art_settings(self, size: int) -> PixelArtSettings:
        return PixelArtSettings(
            size=size,
            stylize=self.stylize,
            saturation_boost=self.saturation_boost,
            gamma=self.gamma,
            contrast=self.contrast,
            black_point=self.black_point,
            land_green=self.land_green,
            colors=self.colors,
            dither=self.dither,
            downsample_method=self.downsample_method,
            supersample=self.supersample,
            display_scale=self.display_scale,
        )


@dataclass
class SizeReport:
    size: int
    frame_count: int
    output_size: int  # the actual pixel dimensions written, after display-scale blow-up


@dataclass
class SpriteReport:
    run_id: str
    run_dir: Path
    source: Path
    settings: SpriteSettings
    sizes: list[SizeReport] = field(default_factory=list)


def run_id(source: Path, settings: SpriteSettings) -> str:
    """Short stable hash of everything that determines a run's output."""
    payload = json.dumps(
        {"source": str(source.resolve()), "settings": settings.as_dict()}, sort_keys=True
    )
    return hashlib.blake2b(payload.encode(), digest_size=4).hexdigest()


def resolve_frames_dir(path: Path) -> Path:
    """A turntable run root, or an already-a-frames directory -- either works."""
    candidate = path / "turntable" / "frames"
    return candidate if candidate.is_dir() else path


def find_frames(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.png"))


def render_all(
    source: Path,
    out_root: Path,
    settings: SpriteSettings,
    *,
    write_gif: bool = True,
    write_sheet: bool = True,
    on_progress=None,
) -> SpriteReport:
    """Render every frame at every requested size and write the run folder above."""
    frames_dir = resolve_frames_dir(source)
    frame_paths = find_frames(frames_dir)

    identifier = run_id(source, settings)
    run_dir = out_root / f"{_RUN_PREFIX}{identifier}"

    size_reports: list[SizeReport] = []
    total_steps = len(settings.sizes) * len(frame_paths)
    done = 0

    for size in settings.sizes:
        art_settings = settings.art_settings(size)
        size_dir = run_dir / f"{size}px"
        out_frames_dir = size_dir / "frames"
        out_frames_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: grade + downsample every frame first, without quantizing --
        # a shared palette needs to see all of them before any one frame can
        # be finished. Splitting this way is what keeps a stable true colour
        # mapped to the same swatch across the whole sequence instead of each
        # frame's independently-discovered palette flickering it between
        # near-identical swatches; see pixel_earth.pixelart.quantize_palette.
        graded: list[tuple[Path, np.ndarray]] = []
        for frame_path in frame_paths:
            try:
                rgba = _load_rgba(frame_path)
            except (UnidentifiedImageError, OSError, ValueError):
                rgba = None  # a corrupt frame is skipped, not fatal
            if rgba is not None:
                graded.append((frame_path, grade_and_downsample(rgba, art_settings)))
            done += 1
            if on_progress is not None:
                on_progress(done, total_steps, size, frame_path)

        palette_image = None
        if settings.shared_palette and graded:
            palette_image = build_shared_palette(
                [small[..., :3] for _, small in graded], colors=settings.colors
            )

        # Pass 2: quantize (against the shared palette, if built) and write.
        out_paths: list[Path] = []
        output_size = None
        for frame_path, small in graded:
            pixel_art = finish(small, art_settings, palette_image=palette_image)
            output_size = pixel_art.shape[0]
            out_path = out_frames_dir / frame_path.name
            Image.fromarray(pixel_art, "RGBA").save(out_path)
            out_paths.append(out_path)

        if write_sheet and out_paths:
            sheets.write_contact_sheet(out_paths, size_dir / "sheet.png")
        if write_gif and out_paths:
            sheets.write_rotation_gif(out_paths, size_dir / "rotation.gif")

        size_reports.append(
            SizeReport(size=size, frame_count=len(out_paths), output_size=output_size or 0)
        )

    report = SpriteReport(identifier, run_dir, source, settings, size_reports)
    _write_manifest(report)
    _link_latest(run_dir)
    return report


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGBA"))


def _write_manifest(report: SpriteReport) -> Path:
    manifest = {
        "run_id": report.run_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(report.source.resolve()),
        "settings": report.settings.as_dict(),
        "sizes": [asdict(s) for s in report.sizes],
    }
    path = report.run_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _link_latest(run_dir: Path) -> None:
    """Point ``outputs/latest-pixelart`` at this run. Best effort; never fatal."""
    latest = run_dir.parent / "latest-pixelart"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel-earth-pixelart",
        description="Render a turntable rotation as pixel art, at one or more resolutions.",
    )
    parser.add_argument(
        "source", type=Path, help="a turntable run directory, or a directory of RGBA frame PNGs"
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="output root (default: <repo root>/outputs)",
    )
    parser.add_argument(
        "--sizes", default=",".join(map(str, SpriteSettings.sizes)),
        help="comma-separated pixel-art grid sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--stylize", type=float, default=SpriteSettings.stylize,
        help="0 = true colour, 1 = fully colour-graded (default: %(default)s)",
    )
    parser.add_argument(
        "--saturation-boost", type=float, default=SpriteSettings.saturation_boost,
        help="vibrance curve strength (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma", type=float, default=SpriteSettings.gamma,
        help="brightness lift, >1 brightens shadows/midtones (default: %(default)s)",
    )
    parser.add_argument(
        "--contrast", type=float, default=SpriteSettings.contrast,
        help="post-lift contrast S-curve, separating ocean/land/cloud "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--black-point", type=float, default=SpriteSettings.black_point,
        help="shadow cutoff stretched back to true black, 0-1 (default: %(default)s)",
    )
    parser.add_argument(
        "--land-green", type=float, default=SpriteSettings.land_green,
        help="rotate tan/brown land hues toward green, 0 = true hue, "
        "1 = fully rotated (default: %(default)s)",
    )
    parser.add_argument(
        "--colors", type=int, default=SpriteSettings.colors,
        help="palette size (default: %(default)s)",
    )
    parser.add_argument("--dither", action="store_true", help="Floyd-Steinberg dither the palette")
    parser.add_argument(
        "--downsample", choices=["nearest", "box"], default=SpriteSettings.downsample_method,
        dest="downsample_method",
        help="nearest = bold/blocky (default), box = softer area-averaged blend",
    )
    parser.add_argument(
        "--supersample", type=int, default=SpriteSettings.supersample,
        help="nearest-neighbour samples averaged per output pixel "
        "(1 = none, default: %(default)s) -- stabilizes hard edges like coastlines "
        "across a rotating sequence without softening interior colour",
    )
    parser.add_argument(
        "--display-scale", type=int, default=None,
        help="nearest-neighbour blow-up factor (default: derived per size, ~512px)",
    )
    parser.add_argument(
        "--no-shared-palette", action="store_false", dest="shared_palette",
        help="quantize each frame against its own palette instead of one shared "
        "across the sequence (faster, but the same true colour can flicker between "
        "near-identical swatches frame to frame)",
    )
    parser.add_argument("--no-gif", action="store_true", help="skip rotation.gif per size")
    parser.add_argument("--no-sheet", action="store_true", help="skip sheet.png per size")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    except ValueError:
        print(f"could not parse --sizes: {args.sizes!r}")
        return 2
    if not sizes:
        print("no sizes given")
        return 2

    settings = SpriteSettings(
        sizes=sizes,
        stylize=args.stylize,
        saturation_boost=args.saturation_boost,
        gamma=args.gamma,
        contrast=args.contrast,
        black_point=args.black_point,
        land_green=args.land_green,
        colors=args.colors,
        dither=args.dither,
        downsample_method=args.downsample_method,
        supersample=args.supersample,
        shared_palette=args.shared_palette,
        display_scale=args.display_scale,
    )
    out_root = args.out or (find_repo_root() / _OUTPUT_DIR_NAME)

    frames_dir = resolve_frames_dir(args.source)
    if not frames_dir.is_dir():
        print(f"not a directory: {args.source}")
        return 2
    if not find_frames(frames_dir):
        print(f"no frame PNGs found in {frames_dir}")
        return 1

    def progress(done: int, total: int, size: int, frame_path: Path) -> None:
        print(f"[{done}/{total}] {size}px  {frame_path.name}")

    report = render_all(
        args.source,
        out_root,
        settings,
        write_gif=not args.no_gif,
        write_sheet=not args.no_sheet,
        on_progress=progress,
    )

    print(f"\nrun {report.run_id} -> {report.run_dir}")
    for size_report in report.sizes:
        print(
            f"  {size_report.size:>4}px grid  "
            f"{size_report.frame_count} frames  "
            f"({size_report.output_size}x{size_report.output_size} written)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
