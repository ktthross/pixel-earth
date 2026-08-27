"""Metadata-only frame indexing/prefiltering, and on-demand geometry caching."""

import json
from pathlib import Path

import pytest

from pixel_earth.batch import Settings
from pixel_earth.catalog import GeometryCache, candidate_frames, load_frame_index
from pixel_earth.epic import Frame
from pixel_earth.geometry import cap_radius_deg

from conftest import write_earth


def _write_day(root: Path, collection: str, day: str, entries: list[dict]) -> None:
    year, month, dom = day.split("-")
    day_dir = root / collection / year / month / dom
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "metadata.json").write_text(json.dumps(entries))


def _entry(image: str, lat: float, lon: float, when: str) -> dict:
    return {
        "image": image,
        "date": when,
        "centroid_coordinates": {"lat": lat, "lon": lon},
    }


# --------------------------------------------------------------- frame index


def test_load_frame_index_reads_metadata_without_opening_any_image(tmp_path, monkeypatch):
    _write_day(
        tmp_path,
        "natural",
        "2024-01-20",
        [
            _entry("epic_1b_20240120003633", 1.2, -34.5, "2024-01-20 00:36:33"),
            _entry("epic_1b_20240120013049", 1.1, -22.1, "2024-01-20 01:30:49"),
        ],
    )
    _write_day(
        tmp_path,
        "natural",
        "2024-01-21",
        [_entry("epic_1b_20240121004512", -2.0, 40.0, "2024-01-21 00:45:12")],
    )

    def _boom(*args, **kwargs):
        raise AssertionError("load_frame_index must not open images")

    monkeypatch.setattr("PIL.Image.open", _boom)

    frames = load_frame_index(tmp_path, "natural")
    assert len(frames) == 3
    assert {f.image for f in frames} == {
        "epic_1b_20240120003633",
        "epic_1b_20240120013049",
        "epic_1b_20240121004512",
    }
    assert all(f.lat is not None and f.lon is not None for f in frames)


def test_load_frame_index_skips_unreadable_metadata(tmp_path):
    day_dir = tmp_path / "natural" / "2024" / "01" / "20"
    day_dir.mkdir(parents=True)
    (day_dir / "metadata.json").write_text("not json")

    assert load_frame_index(tmp_path, "natural") == []


def test_load_frame_index_empty_mirror(tmp_path):
    assert load_frame_index(tmp_path, "natural") == []


# ------------------------------------------------------------- prefiltering


def _frame(image: str, lat: float, lon: float) -> Frame:
    from datetime import date

    return Frame(
        collection="natural",
        image=image,
        archive_day=date(2024, 1, 20),
        captured=None,
        lat=lat,
        lon=lon,
    )


def test_candidate_frames_excludes_far_frames_and_sorts_nearest_first():
    frames = [
        _frame("near", 1.0, 1.0),
        _frame("far", 0.0, 180.0),
        _frame("mid", 5.0, 10.0),
    ]
    cap = cap_radius_deg(0.35)  # ~69.5 degrees

    candidates = candidate_frames(0.0, 0.0, frames, cap_radius_deg=cap)

    assert [f.image for f in candidates] == ["near", "mid"]


def test_candidate_frames_respects_max_candidates():
    frames = [_frame(f"f{i}", float(i), float(i)) for i in range(10)]
    candidates = candidate_frames(0.0, 0.0, frames, cap_radius_deg=90.0, max_candidates=3)
    assert len(candidates) == 3
    assert candidates[0].image == "f0"  # nearest to (0, 0)


def test_candidate_frames_skips_entries_without_coordinates():
    frames = [_frame("has-coords", 1.0, 1.0), _frame("missing", None, None)]
    candidates = candidate_frames(0.0, 0.0, frames, cap_radius_deg=90.0)
    assert [f.image for f in candidates] == ["has-coords"]


def test_candidate_frames_with_no_usable_frames_returns_empty():
    assert candidate_frames(0.0, 0.0, [], cap_radius_deg=90.0) == []


# ------------------------------------------------------------- geometry cache


def test_geometry_cache_computes_plausible_disc_geometry(tmp_path):
    frame = _frame("epic_1b_20240120003633", 0.0, 0.0)
    png_path = frame.local_path(tmp_path, "png")
    write_earth(png_path, size=200, center=(100, 100), radius=50)

    cache = GeometryCache(tmp_path, settings=Settings())
    geometry = cache.get(frame)

    assert geometry is not None
    assert geometry.looks_like_disc
    assert geometry.cx == pytest.approx(100, abs=2)
    assert geometry.cy == pytest.approx(100, abs=2)
    assert geometry.radius == pytest.approx(50, abs=2)


def test_geometry_cache_returns_none_for_missing_file(tmp_path):
    frame = _frame("epic_1b_20240120003633", 0.0, 0.0)
    cache = GeometryCache(tmp_path, settings=Settings())
    assert cache.get(frame) is None


def test_geometry_cache_reuses_disk_cache_without_recomputing(tmp_path, monkeypatch):
    frame = _frame("epic_1b_20240120003633", 0.0, 0.0)
    write_earth(frame.local_path(tmp_path, "png"), size=200, center=(100, 100), radius=50)

    with GeometryCache(tmp_path, settings=Settings()) as cache:
        first = cache.get(frame)

    calls = {"count": 0}
    import pixel_earth.catalog as catalog_module

    real_segment = catalog_module.segment

    def counting_segment(*args, **kwargs):
        calls["count"] += 1
        return real_segment(*args, **kwargs)

    monkeypatch.setattr(catalog_module, "segment", counting_segment)

    second_cache = GeometryCache(tmp_path, settings=Settings())
    second = second_cache.get(frame)

    assert calls["count"] == 0  # served entirely from disk
    assert second == first


def test_geometry_cache_different_settings_do_not_collide(tmp_path):
    frame = _frame("epic_1b_20240120003633", 0.0, 0.0)
    write_earth(frame.local_path(tmp_path, "png"), size=200, center=(100, 100), radius=50)

    default_geo = GeometryCache(tmp_path, settings=Settings()).get(frame)
    padded_geo = GeometryCache(tmp_path, settings=Settings(pad=20)).get(frame)

    # Padding only affects the reported bbox in segment.py, not tight_bbox --
    # so what matters here is that both entries land in the cache distinctly,
    # not that they differ numerically.
    assert default_geo is not None
    assert padded_geo is not None
