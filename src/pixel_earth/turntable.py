"""Render a full rotation of cloud-free Earth from the mirrored EPIC archive.

Ties the rest of the package together: :mod:`pixel_earth.catalog` picks which
mirrored frames are worth decoding across the whole rotation,
:mod:`pixel_earth.mosaic` composites them all onto one
:class:`~pixel_earth.mosaic.ReferenceGrid` *once*, and every output viewpoint
then samples that same fixed grid, before this module writes the results into
a hashed run folder, following the same convention as :mod:`pixel_earth.batch`::

    outputs/<run_id>/turntable/
        frames/frame_000.png ... frame_<N-1>.png   RGBA, transparent off-disc
        contact_sheet.png                           every frame, one glance
        rotation.gif                                looping, black background
        manifest.json                                settings + per-frame stats
    outputs/latest -> <run_id>

The one-shared-grid design matters beyond tidiness: rendering each viewpoint
independently (as an earlier version of this module did) lets the *same*
physical location flip between two different candidate photographs from one
rotation frame to the next, whenever their cloud scores are close -- visible
as flicker/sparkle across the animation, even though nothing in the scene
changed. Deciding every location's winner exactly once, in a fixed lat/lon
frame, removes that by construction: two viewpoints covering the same point
now sample the identical decision. It is also considerably cheaper -- each
candidate is reprojected once against the grid instead of once per viewpoint
that happens to use it.
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
from pixel_earth.batch import Settings, find_repo_root, link_latest, load_rgb
from pixel_earth.catalog import GeometryCache, candidate_frames, load_frame_index
from pixel_earth.cloud_score import SCORERS
from pixel_earth.epic import Frame
from pixel_earth.geometry import DEFAULT_MIN_COS, cap_radius_deg
from pixel_earth.mosaic import (
    Candidate,
    Viewpoint,
    build_reference_grid,
    make_viewpoint,
    render_viewpoint_from_reference,
)

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
    # The shared reference grid's resolution, as a multiple of `radius` --
    # bigger keeps more native detail when a viewpoint samples back out of
    # it, at the cost of reprojecting every candidate onto more cells.
    reference_scale: float = 4.0
    segment_settings: Settings = field(default_factory=Settings)

    def as_dict(self) -> dict:
        return asdict(self)

    def reference_size(self) -> tuple[int, int]:
        """(width, height) of the shared equirectangular reference grid."""
        width = max(2, int(round(self.radius * self.reference_scale)))
        return width, width // 2


@dataclass
class FrameReport:
    index: int
    lon: float
    output: str  # path relative to the run directory
    suspect_fraction: float


@dataclass
class TurntableReport:
    run_id: str
    run_dir: Path
    mirror_root: Path
    settings: TurntableSettings
    reference_candidate_count: int  # frames composited into the shared grid
    reference_coverage: float  # fraction of the reference grid's cells with data
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


def _decode(frame: Frame, root: Path, fmt: str) -> np.ndarray | None:
    path = frame.local_path(root, fmt)
    try:
        return load_rgb(path) if path.exists() else None
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _gather_candidates(
    root: Path,
    settings: TurntableSettings,
    viewpoints: list[Viewpoint],
    index: list[Frame],
    geometry_cache: GeometryCache,
    *,
    on_decode_progress=None,
) -> list[Candidate]:
    """Every frame that's a candidate for *any* viewpoint in the rotation,
    decoded once each. This is deliberately the union across all viewpoints,
    not per-viewpoint -- the whole point of the reference grid is that a
    given physical point's winner is decided once from everything that could
    ever see it, not re-decided per rotation angle.

    The cost of that is memory, and it is the one number to keep an eye on
    here. The union is held decoded all at once (an earlier per-viewpoint
    version could get away with a bounded LRU cache, because it only ever
    needed one viewpoint's candidates at a time), and
    :func:`~pixel_earth.mosaic.build_reference_grid` then stacks a sample of
    every one of them over the grid. Roughly, in bytes:

        3 * H * W * len(candidates)              decoded frames, uint8
        + 24 * height * width * len(candidates)  the score/rgb/argsort stacks

    -- so the shipped ``--frames 72 --radius 360`` run (620 candidates of
    2048^2, onto a 1440x720 grid) peaks near 23 GiB. ``--max-candidates``
    bounds the first term per viewpoint but not the union; ``--reference-scale``
    is the direct lever on the second."""
    cap = cap_radius_deg(settings.min_cos)
    seen: dict[str, Frame] = {}
    for viewpoint in viewpoints:
        for frame in candidate_frames(
            viewpoint.lat0,
            viewpoint.lon0,
            index,
            cap_radius_deg=cap,
            max_candidates=settings.max_candidates,
        ):
            seen.setdefault(frame.image, frame)

    candidates: list[Candidate] = []
    for done, frame in enumerate(seen.values(), start=1):
        geometry = geometry_cache.get(frame)
        if geometry is not None and geometry.looks_like_disc:
            rgb = _decode(frame, root, settings.fmt)
            if rgb is not None:
                candidates.append(Candidate(geometry=geometry, rgb=rgb))
        if on_decode_progress is not None:
            on_decode_progress(done, len(seen))
    return candidates


def render_all(
    root: Path,
    out_root: Path,
    settings: TurntableSettings,
    *,
    write_gif: bool = True,
    write_contact_sheet: bool = True,
    on_progress=None,
    on_reference_progress=None,
) -> TurntableReport:
    """Build the shared reference grid, then render every viewpoint from it."""
    identifier = run_id(root, settings)
    run_dir = out_root / identifier
    turntable_dir = run_dir / "turntable"
    frames_dir = turntable_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scorer = SCORERS[settings.scorer]
    index = load_frame_index(root, settings.collection)
    viewpoints = build_viewpoints(settings)

    with GeometryCache(root, settings=settings.segment_settings, fmt=settings.fmt) as geometry_cache:
        candidates = _gather_candidates(
            root,
            settings,
            viewpoints,
            index,
            geometry_cache,
            on_decode_progress=on_reference_progress,
        )

    width, height = settings.reference_size()
    grid = build_reference_grid(
        width,
        height,
        candidates,
        scorer=scorer,
        min_cos=settings.min_cos,
        luminance_floor=settings.luminance_floor,
        obliqueness_penalty=settings.obliqueness_penalty,
        blend_k=settings.blend_k,
        blend_margin=settings.blend_margin,
    )

    frame_reports: list[FrameReport] = []
    frame_paths: list[Path] = []

    for i, viewpoint in enumerate(viewpoints):
        result = render_viewpoint_from_reference(viewpoint, grid)

        frame_path = frames_dir / f"frame_{i:03d}.png"
        Image.fromarray(result.rgba, "RGBA").save(frame_path)
        frame_paths.append(frame_path)

        frame_reports.append(
            FrameReport(
                index=i,
                lon=viewpoint.lon0,
                output=str(frame_path.relative_to(run_dir)),
                suspect_fraction=result.suspect_fraction,
            )
        )
        if on_progress is not None:
            on_progress(i + 1, len(viewpoints), frame_reports[-1])

    if write_contact_sheet and frame_paths:
        sheets.write_contact_sheet(frame_paths, turntable_dir / "contact_sheet.png")
    if write_gif and frame_paths:
        sheets.write_rotation_gif(frame_paths, turntable_dir / "rotation.gif")

    report = TurntableReport(
        identifier,
        run_dir,
        root,
        settings,
        reference_candidate_count=len(candidates),
        reference_coverage=float(grid.has_data.mean()),
        frames=frame_reports,
    )
    _write_manifest(report)
    link_latest(run_dir)
    return report


def _write_manifest(report: TurntableReport) -> Path:
    manifest = {
        "run_id": report.run_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mirror_root": str(report.mirror_root.resolve()),
        "settings": report.settings.as_dict(),
        "reference_candidate_count": report.reference_candidate_count,
        "reference_coverage": report.reference_coverage,
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
    parser.add_argument(
        "--reference-scale", type=float, default=TurntableSettings.reference_scale,
        help="shared reference grid width, as a multiple of --radius (default: %(default)s)",
    )
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
        reference_scale=args.reference_scale,
        scorer=args.scorer,
        collection=args.collection,
        fmt=args.fmt,
    )
    out_root = args.out or (find_repo_root() / _OUTPUT_DIR_NAME)

    if not args.mirror.is_dir():
        print(f"not a directory: {args.mirror}")
        return 2

    width, height = settings.reference_size()
    print(f"building {width}x{height} reference grid...")

    def reference_progress(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"  decoded {done}/{total} candidate frames")

    def progress(done: int, total: int, frame: FrameReport) -> None:
        flag = " REVIEW" if frame.suspect_fraction > 0.05 else ""
        print(f"[{done}/{total}] lon {frame.lon:6.1f}  suspect {frame.suspect_fraction:5.1%}{flag}")

    report = render_all(
        args.mirror,
        out_root,
        settings,
        write_gif=not args.no_gif,
        write_contact_sheet=not args.no_contact_sheet,
        on_progress=progress,
        on_reference_progress=reference_progress,
    )

    print(f"\nrun {report.run_id} -> {report.run_dir}")
    print(
        f"  reference grid: {report.reference_candidate_count} candidates, "
        f"{report.reference_coverage:.1%} coverage"
    )
    print(f"  mean suspect fraction: {report.mean_suspect_fraction:.1%}")
    review = [f for f in report.frames if f.suspect_fraction > 0.05]
    if review:
        print(f"  {len(review)} frame(s) over 5% suspect, see manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
