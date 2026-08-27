"""End-to-end: a tiny fabricated mirror all the way to a rendered rotation."""

import json
from pathlib import Path

import pytest

from conftest import write_earth
from pixel_earth.turntable import TurntableSettings, main, render_all, run_id


def _write_day(root: Path, day: str, lons: list[float], *, hour_start: int = 0) -> None:
    year, month, dom = day.split("-")
    entries = []
    for i, lon in enumerate(lons):
        image = f"epic_1b_{year}{month}{dom}{hour_start + i:02d}0000"
        entries.append(
            {
                "image": image,
                "date": f"{day} {hour_start + i:02d}:00:00",
                "centroid_coordinates": {"lat": 5.0, "lon": lon},
            }
        )
        write_earth(root / "natural" / year / month / dom / "png" / f"{image}.png", size=120, center=(60, 60), radius=50)
    (root / "natural" / year / month / dom / "metadata.json").write_text(json.dumps(entries))


@pytest.fixture
def fake_mirror(tmp_path) -> Path:
    root = tmp_path / "mirror"
    _write_day(root, "2024-01-01", [-170, -100, -30, 40, 110, 170])
    _write_day(root, "2024-06-01", [-150, -80, -10, 60, 130])
    return root


def test_render_all_produces_a_full_run(fake_mirror, tmp_path):
    out_root = tmp_path / "outputs"
    settings = TurntableSettings(frame_count=6, radius=40, max_candidates=10)

    report = render_all(fake_mirror, out_root, settings)

    assert len(report.frames) == 6
    turntable_dir = report.run_dir / "turntable"
    for i in range(6):
        assert (turntable_dir / "frames" / f"frame_{i:03d}.png").exists()
    assert (turntable_dir / "contact_sheet.png").exists()
    assert (turntable_dir / "rotation.gif").exists()
    assert (turntable_dir / "manifest.json").exists()
    assert (out_root / "latest").resolve() == report.run_dir.resolve()

    manifest = json.loads((turntable_dir / "manifest.json").read_text())
    assert manifest["run_id"] == report.run_id
    assert len(manifest["frames"]) == 6


def test_rerun_with_same_settings_reuses_the_run_id(fake_mirror, tmp_path):
    out_root = tmp_path / "outputs"
    settings = TurntableSettings(frame_count=4, radius=40, max_candidates=10)

    first = render_all(fake_mirror, out_root, settings)
    second = render_all(fake_mirror, out_root, settings)

    assert first.run_id == second.run_id


def test_different_settings_land_in_a_different_folder(fake_mirror, tmp_path):
    out_root = tmp_path / "outputs"
    a = render_all(fake_mirror, out_root, TurntableSettings(frame_count=4, radius=40, max_candidates=10))
    b = render_all(fake_mirror, out_root, TurntableSettings(frame_count=4, radius=60, max_candidates=10))

    assert a.run_id != b.run_id
    assert a.run_dir != b.run_dir


def test_run_id_depends_on_mirror_root_and_settings(tmp_path):
    settings_a = TurntableSettings()
    settings_b = TurntableSettings(frame_count=100)

    assert run_id(tmp_path / "one", settings_a) != run_id(tmp_path / "two", settings_a)
    assert run_id(tmp_path / "one", settings_a) != run_id(tmp_path / "one", settings_b)
    assert run_id(tmp_path / "one", settings_a) == run_id(tmp_path / "one", settings_a)


def test_no_gif_and_no_contact_sheet_flags_skip_their_outputs(fake_mirror, tmp_path):
    out_root = tmp_path / "outputs"
    settings = TurntableSettings(frame_count=4, radius=40, max_candidates=10)

    report = render_all(fake_mirror, out_root, settings, write_gif=False, write_contact_sheet=False)

    turntable_dir = report.run_dir / "turntable"
    assert not (turntable_dir / "rotation.gif").exists()
    assert not (turntable_dir / "contact_sheet.png").exists()


def test_cli_end_to_end(fake_mirror, tmp_path, capsys):
    out_root = tmp_path / "outputs"
    exit_code = main(["--frames", "4", "--radius", "40", "--max-candidates", "10", "-o", str(out_root), str(fake_mirror)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "mean suspect fraction" in captured.out
    run_dirs = [p for p in out_root.iterdir() if p.name != "latest"]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "turntable" / "frames" / "frame_000.png").exists()


def test_cli_rejects_a_missing_mirror_directory(tmp_path, capsys):
    exit_code = main(["-o", str(tmp_path / "outputs"), str(tmp_path / "does-not-exist")])
    assert exit_code == 2
