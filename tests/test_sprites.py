"""End-to-end: a few RGBA frames -> pixel-art sheets/GIFs at multiple sizes."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_earth.sprites import SpriteSettings, main, render_all, resolve_frames_dir, run_id


def _write_frame(path: Path, *, size: int = 120, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size * 0.4) ** 2
    rgba[disc, :3] = rng.integers(0, 256, size=(disc.sum(), 3))
    rgba[disc, 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path)


@pytest.fixture
def fake_frames_dir(tmp_path) -> Path:
    frames_dir = tmp_path / "turntable" / "frames"
    for i in range(4):
        _write_frame(frames_dir / f"frame_{i:03d}.png", seed=i)
    return tmp_path  # the turntable run root, not the frames dir itself


def test_resolve_frames_dir_finds_turntable_layout(fake_frames_dir):
    resolved = resolve_frames_dir(fake_frames_dir)
    assert resolved == fake_frames_dir / "turntable" / "frames"


def test_resolve_frames_dir_falls_back_to_the_path_itself(tmp_path):
    plain = tmp_path / "plain_frames"
    plain.mkdir()
    assert resolve_frames_dir(plain) == plain


def test_render_all_produces_every_size(fake_frames_dir, tmp_path):
    out_root = tmp_path / "outputs"
    settings = SpriteSettings(sizes=(8, 16), stylize=0.5, colors=8, display_scale=2)

    report = render_all(fake_frames_dir, out_root, settings)

    assert [s.size for s in report.sizes] == [8, 16]
    for size_report in report.sizes:
        assert size_report.frame_count == 4
        size_dir = report.run_dir / f"{size_report.size}px"
        for i in range(4):
            assert (size_dir / "frames" / f"frame_{i:03d}.png").exists()
        assert (size_dir / "sheet.png").exists()
        assert (size_dir / "rotation.gif").exists()

    manifest = json.loads((report.run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == report.run_id
    assert (out_root / "latest-pixelart").resolve() == report.run_dir.resolve()


def test_rerun_with_same_settings_reuses_the_run_id(fake_frames_dir, tmp_path):
    out_root = tmp_path / "outputs"
    settings = SpriteSettings(sizes=(8,), colors=8, display_scale=2)

    first = render_all(fake_frames_dir, out_root, settings)
    second = render_all(fake_frames_dir, out_root, settings)

    assert first.run_id == second.run_id


def test_different_stylize_lands_in_a_different_folder(fake_frames_dir, tmp_path):
    out_root = tmp_path / "outputs"
    a = render_all(fake_frames_dir, out_root, SpriteSettings(sizes=(8,), stylize=0.0, colors=8))
    b = render_all(fake_frames_dir, out_root, SpriteSettings(sizes=(8,), stylize=1.0, colors=8))

    assert a.run_id != b.run_id
    assert a.run_dir != b.run_dir


def test_run_id_depends_on_source_and_settings(tmp_path):
    settings_a = SpriteSettings()
    settings_b = SpriteSettings(stylize=0.9)

    assert run_id(tmp_path / "one", settings_a) != run_id(tmp_path / "two", settings_a)
    assert run_id(tmp_path / "one", settings_a) != run_id(tmp_path / "one", settings_b)


def test_pixel_art_output_is_smaller_than_source_before_display_scale(fake_frames_dir, tmp_path):
    out_root = tmp_path / "outputs"
    settings = SpriteSettings(sizes=(16,), colors=8, display_scale=1)
    report = render_all(fake_frames_dir, out_root, settings)

    out_path = report.run_dir / "16px" / "frames" / "frame_000.png"
    with Image.open(out_path) as img:
        assert img.size == (16, 16)


def test_cli_end_to_end(fake_frames_dir, tmp_path, capsys):
    out_root = tmp_path / "outputs"
    exit_code = main(
        ["--sizes", "8,16", "--colors", "8", "--display-scale", "2", "-o", str(out_root), str(fake_frames_dir)]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "8px grid" in captured.out and "16px grid" in captured.out
    run_dirs = [p for p in out_root.iterdir() if p.name != "latest-pixelart"]
    assert len(run_dirs) == 1


def test_cli_rejects_a_directory_with_no_frames(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    exit_code = main(["-o", str(tmp_path / "outputs"), str(empty)])
    assert exit_code == 1


def test_cli_rejects_unparseable_sizes(fake_frames_dir, tmp_path):
    exit_code = main(["--sizes", "not-a-number", "-o", str(tmp_path / "outputs"), str(fake_frames_dir)])
    assert exit_code == 2
