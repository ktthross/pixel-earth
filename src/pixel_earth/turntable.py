"""Render a full rotation of cloud-free Earth from the mirrored EPIC archive.

Ties the rest of the package together: :mod:`pixel_earth.catalog` picks which
mirrored frames are worth decoding for each of ``frame_count`` evenly-spaced
longitudes, :mod:`pixel_earth.mosaic` composites each one, and this module
writes the results into a hashed run folder, following the same convention as
:mod:`pixel_earth.batch`::

    outputs/<run_id>/turntable/
        frames/frame_000.png ... frame_<N-1>.png   RGBA, transparent off-disc
        contact_sheet.png                           every frame, one glance
        rotation.gif                                looping, black background
        manifest.json                                settings + per-frame stats
    outputs/latest -> <run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from pixel_earth.batch import Settings, find_repo_root, link_latest, load_rgb
from pixel_earth.catalog import GeometryCache, candidate_frames, load_frame_index
from pixel_earth.cloud_score import SCORERS
from pixel_earth.epic import Frame
from pixel_earth.geometry import DEFAULT_MIN_COS, cap_radius_deg
from pixel_earth.mosaic import Candidate, Viewpoint, make_viewpoint, render_viewpoint

_OUTPUT_DIR_NAME = "outputs"


@dataclass(frozen=True)
class TurntableSettings:
    """Everything that determines a turntable run's output. Frozen so it can be hashed."""

    lat0: float = 0.0  # sub-satellite latitude every output viewpoint shares
    frame_count: int = 72
    radius: int = 800  # output disc radius, pixels
    min_cos: float = DEFAULT_MIN_COS
    luminance_floor: int = 30
    obliqueness_penalty: float = 0.5
    blend_k: int = 1
    blend_margin: float = 1.5
    max_candidates: int = 48
    scorer: str = "rgb"
    collection: str = "natural"
    fmt: str = "png"
    segment_settings: Settings = field(default_factory=Settings)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameReport:
    index: int
    lon: float
    output: str  # path relative to the run directory
    candidate_count: int
    suspect_fraction: float


@dataclass
class TurntableReport:
    run_id: str
    run_dir: Path
    mirror_root: Path
    settings: TurntableSettings
    frames: list[FrameReport]

    @property
    def mean_suspect_fraction(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.suspect_fraction for f in self.frames) / len(self.frames)


def run_id(mirror_root: Path, settings: TurntableSettings) -> str:
    """Short stable hash of everything that determines a run's output."""
    payload = json.dumps(
        {"mirror_root": str(mirror_root.resolve()), "settings": settings.as_dict()},
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode(), digest_size=4).hexdigest()


def build_viewpoints(settings: TurntableSettings) -> list[Viewpoint]:
    """``frame_count`` viewpoints evenly spaced around a full rotation."""
    longitudes = np.linspace(0.0, 360.0, settings.frame_count, endpoint=False)
    return [
        make_viewpoint(settings.lat0, float(lon), radius=settings.radius) for lon in longitudes
    ]


class _DecodedFrameCache:
    """Bounded LRU of decoded frames, since adjacent viewpoints share most of
    their candidates -- without a cache each shared frame would be re-decoded
    once per viewpoint that uses it."""

    def __init__(self, root: Path, fmt: str, *, capacity: int = 64):
        self._root = root
        self._fmt = fmt
        self._capacity = capacity
        self._store: OrderedDict[str, np.ndarray | None] = OrderedDict()

    def get(self, frame: Frame) -> np.ndarray | None:
        key = frame.image
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]

        path = frame.local_path(self._root, self._fmt)
        try:
            rgb = load_rgb(path) if path.exists() else None
        except (UnidentifiedImageError, OSError, ValueError):
            rgb = None

        self._store[key] = rgb
        self._store.move_to_end(key)
        if len(self._store) > self._capacity:
            self._store.popitem(last=False)
        return rgb


def render_all(
    root: Path,
    out_root: Path,
    settings: TurntableSettings,
    *,
    write_gif: bool = True,
    write_contact_sheet: bool = True,
    on_progress=None,
) -> TurntableReport:
    """Render every viewpoint and write the run folder described above."""
    identifier = run_id(root, settings)
    run_dir = out_root / identifier
    turntable_dir = run_dir / "turntable"
    frames_dir = turntable_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scorer = SCORERS[settings.scorer]
    index = load_frame_index(root, settings.collection)
    cap = cap_radius_deg(settings.min_cos)
    viewpoints = build_viewpoints(settings)

    frame_reports: list[FrameReport] = []
    frame_paths: list[Path] = []

    with GeometryCache(root, settings=settings.segment_settings, fmt=settings.fmt) as geometry_cache:
        decoded = _DecodedFrameCache(root, settings.fmt)

        for i, viewpoint in enumerate(viewpoints):
            candidate_meta = candidate_frames(
                viewpoint.lat0,
                viewpoint.lon0,
                index,
                cap_radius_deg=cap,
                max_candidates=settings.max_candidates,
            )

            candidates: list[Candidate] = []
            for frame in candidate_meta:
                geometry = geometry_cache.get(frame)
                if geometry is None or not geometry.looks_like_disc:
                    continue
                rgb = decoded.get(frame)
                if rgb is None:
                    continue
                candidates.append(Candidate(geometry=geometry, rgb=rgb))

            result = render_viewpoint(
                viewpoint,
                candidates,
                scorer=scorer,
                min_cos=settings.min_cos,
                luminance_floor=settings.luminance_floor,
                obliqueness_penalty=settings.obliqueness_penalty,
                blend_k=settings.blend_k,
                blend_margin=settings.blend_margin,
            )

            frame_path = frames_dir / f"frame_{i:03d}.png"
            Image.fromarray(result.rgba, "RGBA").save(frame_path)
            frame_paths.append(frame_path)

            frame_reports.append(
                FrameReport(
                    index=i,
                    lon=viewpoint.lon0,
                    output=str(frame_path.relative_to(run_dir)),
                    candidate_count=len(candidates),
                    suspect_fraction=result.suspect_fraction,
                )
            )
            if on_progress is not None:
                on_progress(i + 1, len(viewpoints), frame_reports[-1])

    if write_contact_sheet and frame_paths:
        _write_contact_sheet(frame_paths, turntable_dir / "contact_sheet.png")
    if write_gif and frame_paths:
        _write_rotation_gif(frame_paths, turntable_dir / "rotation.gif")

    report = TurntableReport(identifier, run_dir, root, settings, frame_reports)
    _write_manifest(report)
    link_latest(run_dir)
    return report


def _flatten_on_black(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGB", img.size, (0, 0, 0))
    canvas.paste(img, mask=img.split()[3])
    return canvas


def _write_contact_sheet(frame_paths: list[Path], out: Path) -> None:
    flattened = [_flatten_on_black(p) for p in frame_paths]
    width, height = flattened[0].size
    columns = max(1, math.ceil(math.sqrt(len(flattened))))
    rows = math.ceil(len(flattened) / columns)

    sheet = Image.new("RGB", (columns * width, rows * height), (0, 0, 0))
    for i, img in enumerate(flattened):
        x, y = (i % columns) * width, (i // columns) * height
        sheet.paste(img, (x, y))
    sheet.save(out)


def _write_rotation_gif(frame_paths: list[Path], out: Path, *, duration_ms: int = 80) -> None:
    flattened = [_flatten_on_black(p) for p in frame_paths]
    flattened[0].save(
        out, save_all=True, append_images=flattened[1:], duration=duration_ms, loop=0
    )


def _write_manifest(report: TurntableReport) -> Path:
    manifest = {
        "run_id": report.run_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mirror_root": str(report.mirror_root.resolve()),
        "settings": report.settings.as_dict(),
        "mean_suspect_fraction": report.mean_suspect_fraction,
        "frames": [asdict(f) for f in report.frames],
    }
    path = report.run_dir / "turntable" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel-earth-turntable",
        description="Render a cloud-free rotation of Earth from mirrored EPIC frames.",
    )
    parser.add_argument("mirror", type=Path, help="EPIC mirror root, e.g. data/epic")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output root (default: <repo root>/outputs)")
    parser.add_argument("--lat0", type=float, default=TurntableSettings.lat0, help="sub-satellite latitude, degrees")
    parser.add_argument("--frames", type=int, default=TurntableSettings.frame_count, dest="frame_count")
    parser.add_argument("--radius", type=int, default=TurntableSettings.radius, help="output disc radius, px")
    parser.add_argument("--min-cos", type=float, default=TurntableSettings.min_cos)
    parser.add_argument("--luminance-floor", type=int, default=TurntableSettings.luminance_floor)
    parser.add_argument("--obliqueness-penalty", type=float, default=TurntableSettings.obliqueness_penalty)
    parser.add_argument("--blend-k", type=int, default=TurntableSettings.blend_k)
    parser.add_argument("--blend-margin", type=float, default=TurntableSettings.blend_margin)
    parser.add_argument("--max-candidates", type=int, default=TurntableSettings.max_candidates)
    parser.add_argument("--scorer", choices=sorted(SCORERS), default=TurntableSettings.scorer)
    parser.add_argument("--collection", default=TurntableSettings.collection)
    parser.add_argument("--fmt", default=TurntableSettings.fmt)
    parser.add_argument("--no-gif", action="store_true", help="skip rotation.gif")
    parser.add_argument("--no-contact-sheet", action="store_true", help="skip contact_sheet.png")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    settings = TurntableSettings(
        lat0=args.lat0,
        frame_count=args.frame_count,
        radius=args.radius,
        min_cos=args.min_cos,
        luminance_floor=args.luminance_floor,
        obliqueness_penalty=args.obliqueness_penalty,
        blend_k=args.blend_k,
        blend_margin=args.blend_margin,
        max_candidates=args.max_candidates,
        scorer=args.scorer,
        collection=args.collection,
        fmt=args.fmt,
    )
    out_root = args.out or (find_repo_root() / _OUTPUT_DIR_NAME)

    if not args.mirror.is_dir():
        print(f"not a directory: {args.mirror}")
        return 2

    def progress(done: int, total: int, frame: FrameReport) -> None:
        flag = " REVIEW" if frame.suspect_fraction > 0.05 else ""
        print(
            f"[{done}/{total}] lon {frame.lon:6.1f}  "
            f"{frame.candidate_count:3d} candidates  "
            f"suspect {frame.suspect_fraction:5.1%}{flag}"
        )

    report = render_all(
        args.mirror,
        out_root,
        settings,
        write_gif=not args.no_gif,
        write_contact_sheet=not args.no_contact_sheet,
        on_progress=progress,
    )

    print(f"\nrun {report.run_id} -> {report.run_dir}")
    print(f"  mean suspect fraction: {report.mean_suspect_fraction:.1%}")
    review = [f for f in report.frames if f.suspect_fraction > 0.05]
    if review:
        print(f"  {len(review)} frame(s) over 5% suspect, see manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
