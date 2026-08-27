import numpy as np
import pytest

from conftest import synthetic_earth
from pixel_earth.segment import (
    DISC_FILL_RATIO,
    bounding_box,
    cutout,
    otsu_threshold,
    segment,
)


def test_otsu_lands_in_the_gap_between_two_modes():
    gray = np.concatenate([np.zeros(500, np.uint8), np.full(500, 200, np.uint8)])
    # Every cutoff in 0..199 ties on between-class variance; we want the middle
    # of that plateau, not its noise-floor-hugging low end.
    assert otsu_threshold(gray.reshape(-1, 1)) == pytest.approx(100, abs=2)


def test_bbox_of_empty_mask_is_none():
    assert bounding_box(np.zeros((10, 10), bool)) is None


def test_segment_finds_disc_bbox():
    img = synthetic_earth(radius=50, center=(100, 100))
    result = segment(img, blur_sigma=0)

    assert result.bbox == (50, 50, 151, 151)
    # A disc fills pi/4 of its bounding box.
    assert result.fill_ratio == pytest.approx(np.pi / 4, abs=0.02)


def test_enclosed_dark_region_is_recovered_by_hole_filling():
    img = synthetic_earth(dark_spot=20)

    filled = segment(img, blur_sigma=0, fill_holes=True)
    unfilled = segment(img, blur_sigma=0, fill_holes=False)

    assert filled.fill_ratio == pytest.approx(np.pi / 4, abs=0.02)
    assert unfilled.coverage < filled.coverage


def test_terminator_reaching_the_limb_defeats_thresholding():
    """Documents the known limit of pixel thresholding, not a bug.

    A dark slice that runs out to the limb is an open notch, not an enclosed
    hole, so ``binary_fill_holes`` cannot recover it and the crop clips the
    night side. Fixing this needs a shape prior (Hough circle), not a better
    threshold.
    """
    img = synthetic_earth(night_fraction=0.35)
    result = segment(img, blur_sigma=0, fill_holes=True)

    left, _, right, _ = result.bbox
    assert right - left < 101  # true disc is 101px wide
    assert result.fill_ratio > np.pi / 4  # a circular segment, not a disc


def test_stars_are_dropped_by_largest_component():
    img = synthetic_earth(stars=60, seed=1)

    kept = segment(img, blur_sigma=0, keep_largest=True)
    everything = segment(img, blur_sigma=0, keep_largest=False)

    assert kept.bbox == (50, 50, 151, 151)
    assert everything.bbox != kept.bbox  # stars blow the box out to the frame


def test_blur_survives_sensor_noise():
    img = synthetic_earth(noise=12.0, seed=2)
    result = segment(img, blur_sigma=1.5)

    left, top, right, bottom = result.bbox
    assert (left, top, right, bottom) == pytest.approx((50, 50, 151, 151), abs=3)


def test_edge_adjust_shrinks_and_grows_the_mask():
    img = synthetic_earth()
    base = segment(img, blur_sigma=0).coverage

    assert segment(img, blur_sigma=0, edge_adjust=-3).coverage < base
    assert segment(img, blur_sigma=0, edge_adjust=3).coverage > base


def test_cutout_is_rgba_and_transparent_outside_the_disc():
    img = synthetic_earth()
    result = segment(img, blur_sigma=0)
    rgba = cutout(img, result)

    assert rgba.shape == (101, 101, 4)
    assert rgba[0, 0, 3] == 0  # corner of the box is outside the disc
    assert rgba[50, 50, 3] == 255  # centre is inside
    assert (rgba[..., :3][rgba[..., 3] == 0] == 0).all()


def test_all_black_image_yields_no_object():
    result = segment(np.zeros((50, 50, 3), np.uint8))

    assert result.is_empty
    assert cutout(np.zeros((50, 50, 3), np.uint8), result) is None


def test_padding_grows_the_box_without_skewing_the_disc_check():
    img = synthetic_earth()
    tight = segment(img, blur_sigma=0, pad=0)
    padded = segment(img, blur_sigma=0, pad=10)

    assert padded.bbox == (40, 40, 161, 161)
    assert padded.fill_ratio == pytest.approx(tight.fill_ratio)


def test_full_disc_passes_the_disc_check():
    result = segment(synthetic_earth(), blur_sigma=0)

    assert result.aspect_ratio == pytest.approx(1.0)
    assert result.looks_like_disc()


def test_aspect_ratio_catches_a_clipped_terminator_that_fill_ratio_misses():
    result = segment(synthetic_earth(night_fraction=0.35), blur_sigma=0)

    # Fill ratio barely budges -- this is why it cannot be the only signal.
    assert result.disc_deviation > 0.12
    assert abs(result.fill_ratio - DISC_FILL_RATIO) / DISC_FILL_RATIO < 0.12
    assert result.aspect_ratio < 0.8
    assert not result.looks_like_disc()


def test_min_area_rejects_a_speck():
    img = np.zeros((200, 200, 3), np.uint8)
    img[100, 100] = (255, 255, 255)

    assert not segment(img, blur_sigma=0, min_area=0.0).is_empty
    assert segment(img, blur_sigma=0, min_area=0.001).is_empty


def test_min_area_keeps_a_real_disc():
    result = segment(synthetic_earth(), blur_sigma=0, min_area=0.001)

    assert result.bbox == (50, 50, 151, 151)


def test_aspect_ratio_of_empty_result_is_zero():
    result = segment(np.zeros((50, 50, 3), np.uint8))

    assert result.aspect_ratio == 0.0
    assert not result.looks_like_disc()
