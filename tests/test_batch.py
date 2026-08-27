import json
from pathlib import Path

import pytest
from conftest import write_earth
from PIL import Image

from pixel_earth.batch import (
    ImageOutcome,
    Settings,
    _review_reason,
    find_images,
    find_repo_root,
    load_rgb,
    main,
    process_directory,
    run_id,
)


@pytest.fixture
def source(tmp_path):
    """Three good earths, one nested, plus a non-image and a hidden file."""
    root = tmp_path / "in"
    write_earth(root / "a.png")
    write_earth(root / "b.jpg", radius=30)
    write_earth(root / "nested" / "a.png", radius=40)  # same stem as a.png
    (root / "notes.txt").write_text("not an image")
    (root / ".hidden.png").write_bytes(b"")
    return root


def run(source, tmp_path, **kwargs):
    kwargs.setdefault("settings", Settings())
    return process_directory(source, out_root=tmp_path / "outputs", **kwargs)


def test_run_id_is_short_and_stable():
    first = run_id(Path("/data"), Settings(), recursive=False)
    second = run_id(Path("/data"), Settings(), recursive=False)

    assert first == second
    assert len(first) == 8
    assert all(c in "0123456789abcdef" for c in first)


def test_run_id_changes_with_settings_and_recursion(tmp_path):
    base = run_id(tmp_path, Settings(), recursive=False)

    assert run_id(tmp_path, Settings(threshold=90), recursive=False) != base
    assert run_id(tmp_path, Settings(pad=5), recursive=False) != base
    assert run_id(tmp_path, Settings(), recursive=True) != base


def test_find_images_skips_non_images_and_hidden_files(source):
    names = [p.name for p in find_images(source, recursive=False)]

    assert names == ["a.png", "b.jpg"]
    assert "notes.txt" not in names
    assert ".hidden.png" not in names


def test_find_images_recursive_includes_nested(source):
    found = find_images(source, recursive=True)

    assert [str(p.relative_to(source)) for p in found] == [
        "a.png",
        "b.jpg",
        "nested/a.png",
    ]


def test_find_images_ignores_our_own_output_folder(source):
    write_earth(source / "outputs" / "abc123" / "cutouts" / "a.png")

    assert all("outputs" not in p.parts for p in find_images(source, recursive=True))


def test_process_directory_writes_cutouts_and_overlays(source, tmp_path):
    report = run(source, tmp_path)

    assert report.count("ok") == 2
    assert (report.run_dir / "cutouts" / "a.png").exists()
    assert (report.run_dir / "overlays" / "a.png").exists()

    with Image.open(report.run_dir / "cutouts" / "a.png") as img:
        assert img.mode == "RGBA"
        # A radius-50 disc is 101px across; the default blur_sigma=1.0 softens
        # the limb and costs about a pixel per side.
        assert img.size == pytest.approx((101, 101), abs=3)


def test_recursive_mirrors_input_tree_so_stems_cannot_collide(source, tmp_path):
    report = run(source, tmp_path, recursive=True)

    assert report.count("ok") == 3
    assert (report.run_dir / "cutouts" / "a.png").exists()
    assert (report.run_dir / "cutouts" / "nested" / "a.png").exists()


def test_no_overlays_flag_skips_them(source, tmp_path):
    report = run(source, tmp_path, write_overlays=False)

    assert (report.run_dir / "cutouts").is_dir()
    assert not (report.run_dir / "overlays").exists()


def test_rerun_is_idempotent_and_resumes(source, tmp_path):
    first = run(source, tmp_path)
    second = run(source, tmp_path)

    assert second.run_dir == first.run_dir  # same hash, same folder
    assert second.count("skipped") == 2
    assert second.count("ok") == 0


def test_force_redoes_finished_images(source, tmp_path):
    run(source, tmp_path)
    forced = run(source, tmp_path, force=True)

    assert forced.count("ok") == 2
    assert forced.count("skipped") == 0


def test_different_settings_land_in_a_different_folder(source, tmp_path):
    default = run(source, tmp_path)
    tighter = run(source, tmp_path, settings=Settings(threshold=90))

    assert tighter.run_dir != default.run_dir
    assert default.run_dir.exists() and tighter.run_dir.exists()


def test_dry_run_writes_nothing(source, tmp_path):
    report = run(source, tmp_path, dry_run=True)

    assert not report.run_dir.exists()
    assert all(o.status == "dry-run" for o in report.outcomes)


def test_manifest_records_settings_and_per_image_stats(source, tmp_path):
    report = run(source, tmp_path, settings=Settings(pad=4))
    manifest = json.loads((report.run_dir / "manifest.json").read_text())

    assert manifest["run_id"] == report.run_dir.name
    assert manifest["settings"]["pad"] == 4
    assert manifest["counts"]["found"] == 2
    assert manifest["counts"]["ok"] == 2
    assert {img["input"] for img in manifest["images"]} == {"a.png", "b.jpg"}
    assert all(img["bbox"] is not None for img in manifest["images"])


def test_latest_symlink_points_at_the_run(source, tmp_path):
    report = run(source, tmp_path)
    latest = report.run_dir.parent / "latest"

    assert latest.resolve() == report.run_dir.resolve()


def test_blank_image_is_reported_empty_not_crashed(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    Image.new("RGB", (60, 60), (0, 0, 0)).save(root / "blank.png")
    report = run(root, tmp_path)

    assert report.count("empty") == 1
    assert report.needs_review
    assert not (report.run_dir / "cutouts").exists()


def test_single_hot_pixel_does_not_become_a_cutout(tmp_path):
    """Otsu always finds something; min_area is what stops it being written."""
    root = tmp_path / "in"
    root.mkdir()
    frame = Image.new("RGB", (200, 200), (0, 0, 0))
    frame.putpixel((100, 100), (255, 255, 255))
    frame.save(root / "hotpixel.png")

    assert run(root, tmp_path).count("empty") == 1
    # Without the floor, the same frame yields a few-pixel "cutout".
    assert run(root, tmp_path, settings=Settings(min_area=0.0)).count("ok") == 1


def test_corrupt_image_is_reported_failed_and_batch_continues(source, tmp_path):
    (source / "corrupt.png").write_bytes(b"not a png at all")
    report = run(source, tmp_path)

    assert report.count("failed") == 1
    assert report.count("ok") == 2  # the good images still went through


def test_clipped_terminator_is_flagged_for_review(tmp_path):
    root = tmp_path / "in"
    write_earth(root / "good.png")
    write_earth(root / "terminator.png", night_fraction=0.35)
    report = run(root, tmp_path)

    flagged = {o.input for o in report.needs_review}
    assert flagged == {"terminator.png"}


def test_exif_orientation_is_applied_on_load(tmp_path):
    """A phone photo carries its rotation in EXIF, not in the pixel data."""
    path = tmp_path / "rotated.jpg"
    wide = Image.new("RGB", (80, 40), (60, 110, 190))  # 80 wide, 40 tall
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 90 degrees clockwise
    wide.save(path, exif=exif)

    # Transposed, so the loaded array is (H, W) = (80, 40) -- tall, not wide.
    assert load_rgb(path).shape[:2] == (80, 40)


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(NotADirectoryError):
        run(tmp_path / "nope", tmp_path)


def test_find_repo_root_finds_the_git_directory(tmp_path):
    (tmp_path / ".git").mkdir()
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)

    assert find_repo_root(deep) == tmp_path.resolve()


def test_cli_end_to_end(source, tmp_path, capsys):
    code = main([str(source), "-o", str(tmp_path / "outputs"), "--recursive", "--pad", "3"])
    out = capsys.readouterr().out

    assert code == 0
    assert "ok 3" in out
    # Resolve first: the `latest` symlink matches the glob too.
    runs = {p.resolve() for p in (tmp_path / "outputs").glob("*/manifest.json")}
    assert len(runs) == 1


def test_cli_reports_empty_directory(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert main([str(empty), "-o", str(tmp_path / "outputs")]) == 1
    assert "no images found" in capsys.readouterr().out


def test_cli_returns_nonzero_when_an_image_fails(source, tmp_path):
    (source / "corrupt.png").write_bytes(b"junk")

    assert main([str(source), "-o", str(tmp_path / "outputs")]) == 1


def test_review_reason_names_the_signal_that_fired(tmp_path):
    root = tmp_path / "in"
    write_earth(root / "terminator.png", night_fraction=0.35)
    root_blank = root / "blank.png"
    Image.new("RGB", (60, 60), (0, 0, 0)).save(root_blank)
    report = run(root, tmp_path)

    reasons = {o.input: _review_reason(o) for o in report.needs_review}
    assert "aspect" in reasons["terminator.png"]
    assert reasons["blank.png"] == "no object found"


def test_review_reason_prefers_the_error_message(tmp_path):
    outcome = ImageOutcome("x.png", "failed", error="OSError: broken")

    assert _review_reason(outcome) == "OSError: broken"
