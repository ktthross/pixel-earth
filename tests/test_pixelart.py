"""Colour grading, alpha-aware downsampling, palette quantization, upscaling."""

import numpy as np
import pytest

from pixel_earth.pixelart import (
    PixelArtSettings,
    build_shared_palette,
    display_scale_for,
    downsample_rgba,
    finish,
    grade,
    grade_and_downsample,
    hsv_to_rgb,
    pixelate,
    quantize_palette,
    quantize_to_palette,
    rgb_to_hsv,
    upscale_nearest,
)

# ------------------------------------------------------------------- colour


def test_hsv_round_trip():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    h, s, v = rgb_to_hsv(rgb)
    back = hsv_to_rgb(h, s, v).round().clip(0, 255).astype(np.uint8)
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1  # float round-trip slack


def test_hsv_known_values():
    red = np.array([[[255, 0, 0]]], dtype=np.uint8)
    h, s, v = rgb_to_hsv(red)
    assert h[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert s[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert v[0, 0] == pytest.approx(1.0, abs=1e-5)

    gray = np.array([[[128, 128, 128]]], dtype=np.uint8)
    _, s_gray, _ = rgb_to_hsv(gray)
    assert s_gray[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_grade_amount_zero_is_identity():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    assert np.array_equal(grade(rgb, 0.0), rgb)


def test_grade_amount_one_brightens_and_saturates_a_muted_dark_pixel():
    muted_ocean = np.array([[[25, 45, 70]]], dtype=np.uint8)  # dark, moderately saturated
    graded = grade(muted_ocean, 1.0)

    h0, s0, v0 = rgb_to_hsv(muted_ocean)
    h1, s1, v1 = rgb_to_hsv(graded)
    assert v1[0, 0] > v0[0, 0]  # gamma lift brightens
    assert s1[0, 0] > s0[0, 0]  # vibrance boosts saturation
    assert h1[0, 0] == pytest.approx(h0[0, 0], abs=1e-3)  # hue itself is untouched


def test_grade_never_tints_a_neutral_pixel():
    white = np.array([[[240, 240, 240]]], dtype=np.uint8)
    graded = grade(white, 1.0)
    r, g, b = graded[0, 0].astype(int)
    assert abs(r - g) <= 1 and abs(g - b) <= 1  # still grey, not colour-cast


def test_grade_rotates_tan_land_hue_toward_green():
    tan_desert = np.array([[[90, 70, 45]]], dtype=np.uint8)  # hue ~33 degrees, darkish -- plausibly-vegetated
    h0, _, _ = rgb_to_hsv(tan_desert)
    graded = grade(tan_desert, 1.0, land_green=1.0)
    h1, _, _ = rgb_to_hsv(graded)

    assert h1[0, 0] > h0[0, 0]  # moved toward green (higher hue), not just brighter/more saturated
    assert h1[0, 0] < 150 / 360  # but not spun past green into cyan


def test_grade_land_green_zero_leaves_hue_untouched():
    tan_desert = np.array([[[195, 150, 90]]], dtype=np.uint8)
    h0, _, _ = rgb_to_hsv(tan_desert)
    graded = grade(tan_desert, 1.0, land_green=0.0)
    h1, _, _ = rgb_to_hsv(graded)
    assert h1[0, 0] == pytest.approx(h0[0, 0], abs=1e-3)


def test_grade_land_green_does_not_touch_ocean_blue():
    ocean = np.array([[[25, 60, 130]]], dtype=np.uint8)  # hue ~215 degrees, far from the land band
    h0, _, _ = rgb_to_hsv(ocean)
    graded = grade(ocean, 1.0, land_green=1.0)
    h1, _, _ = rgb_to_hsv(graded)
    assert h1[0, 0] == pytest.approx(h0[0, 0], abs=1e-3)


def test_grade_land_green_does_not_tint_desaturated_cloud():
    cloud = np.array([[[235, 230, 225]]], dtype=np.uint8)  # bright, nearly grey, s ~ 0.04
    with_rotation = grade(cloud, 1.0, land_green=1.0)
    without_rotation = grade(cloud, 1.0, land_green=0.0)
    # Low enough saturation that the land-hue gate excludes it regardless of
    # land_green strength -- any remaining difference is the (unrelated)
    # saturation/contrast grading, not a hue rotation.
    assert np.array_equal(with_rotation, without_rotation)


def test_grade_bright_sand_rotates_less_than_dark_land_of_the_same_hue():
    # Bare sand/rock reflects far more light than vegetation, so a bright
    # land pixel is far more likely to be desert -- it should keep most of
    # its true tan/orange rather than getting the same green push as a
    # darker, plausibly-vegetated pixel at the identical hue.
    bright_sand = np.array([[[210, 163, 100]]], dtype=np.uint8)  # hue ~33 deg, v ~ 0.82
    dark_land = np.array([[[90, 70, 45]]], dtype=np.uint8)  # hue ~33 deg, v ~ 0.35

    h_sand_before, _, _ = rgb_to_hsv(bright_sand)
    h_land_before, _, _ = rgb_to_hsv(dark_land)
    h_sand_after, _, _ = rgb_to_hsv(grade(bright_sand, 1.0, land_green=1.0))
    h_land_after, _, _ = rgb_to_hsv(grade(dark_land, 1.0, land_green=1.0))

    sand_shift = abs(float(h_sand_after[0, 0] - h_sand_before[0, 0]))
    land_shift = abs(float(h_land_after[0, 0] - h_land_before[0, 0]))
    assert sand_shift < land_shift


def test_grade_amount_interpolates_monotonically_in_brightness():
    muted = np.array([[[25, 45, 70]]], dtype=np.uint8)
    values = []
    for amount in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, _, v = rgb_to_hsv(grade(muted, amount))
        values.append(float(v[0, 0]))
    assert values == sorted(values)


# --------------------------------------------------------------- resampling


@pytest.mark.parametrize("method", ["nearest", "box"])
def test_downsample_alpha_aware_transparent_neighbour_does_not_darken_opaque_block(method):
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[0:2, 0:2] = [255, 0, 0, 255]  # opaque red quadrant
    # remaining three quadrants stay fully transparent black

    small = downsample_rgba(rgba, 2, method=method)

    assert tuple(small[0, 0]) == (255, 0, 0, 255)
    assert small[0, 1, 3] == 0
    assert small[1, 0, 3] == 0
    assert small[1, 1, 3] == 0


@pytest.mark.parametrize("method", ["nearest", "box"])
def test_downsample_output_shape(method):
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    small = downsample_rgba(rgba, 8, method=method)
    assert small.shape == (8, 8, 4)


def test_downsample_nearest_defaults_and_never_invents_a_blended_colour():
    # Two flat-colour halves with no gradient between them in the source --
    # nearest downsampling must not introduce any colour that wasn't there.
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :4] = [20, 60, 140, 255]  # left half: ocean blue
    rgba[:, 4:] = [180, 140, 90, 255]  # right half: land tan

    small = downsample_rgba(rgba, 4)  # default method

    present = {tuple(px) for row in small.reshape(-1, 4) for px in [row]}
    assert present <= {(20, 60, 140, 255), (180, 140, 90, 255)}


def test_downsample_box_blends_across_a_hard_edge_while_nearest_does_not():
    rgba = np.zeros((9, 9, 4), dtype=np.uint8)
    rgba[:, :4] = [20, 60, 140, 255]
    rgba[:, 4:] = [180, 140, 90, 255]

    # size=3 doesn't evenly divide 9 the way the colour split does, so a box
    # filter's sampling regions straddle the boundary; nearest never does.
    nearest = downsample_rgba(rgba, 3, method="nearest")
    box = downsample_rgba(rgba, 3, method="box")

    nearest_colours = {tuple(px) for row in nearest.reshape(-1, 4) for px in [row]}
    box_colours = {tuple(px) for row in box.reshape(-1, 4) for px in [row]}
    assert nearest_colours <= {(20, 60, 140, 255), (180, 140, 90, 255)}
    assert box_colours - nearest_colours  # box invents at least one in-between colour


def test_downsample_rejects_unknown_method():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        downsample_rgba(rgba, 2, method="bicubic")


def test_upscale_nearest_repeats_blocks():
    rgba = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8)  # 1x2
    big = upscale_nearest(rgba, 3)
    assert big.shape == (3, 6, 4)
    assert np.all(big[:, :3] == [1, 2, 3, 4])
    assert np.all(big[:, 3:] == [5, 6, 7, 8])


def test_upscale_nearest_factor_one_is_a_no_op():
    rgba = np.zeros((3, 3, 4), dtype=np.uint8)
    assert upscale_nearest(rgba, 1) is rgba


def test_display_scale_for_shrinks_toward_target_without_upscaling_already_big():
    assert display_scale_for(16, target=512) == 32
    assert display_scale_for(512, target=512) == 1
    assert display_scale_for(1024, target=512) == 1  # never below 1


def test_supersample_default_of_one_is_unchanged_single_sample_nearest():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :4] = [20, 60, 140, 255]
    rgba[:, 4:] = [180, 140, 90, 255]

    plain = downsample_rgba(rgba, 4, method="nearest")
    explicit = downsample_rgba(rgba, 4, method="nearest", supersample=1)
    assert np.array_equal(plain, explicit)


def test_supersample_leaves_a_uniform_interior_untouched():
    # Nowhere near an edge -- every one of the supersampled sub-pixel
    # samples agrees, so the averaged result must be pixel-identical to a
    # single nearest-neighbour sample. This is what keeps interior colour
    # blocks flat and bold rather than softened.
    rgba = np.zeros((60, 60, 4), dtype=np.uint8)
    rgba[...] = [90, 140, 60, 255]

    plain = downsample_rgba(rgba, 6, method="nearest", supersample=1)
    supersampled = downsample_rgba(rgba, 6, method="nearest", supersample=3)
    assert np.array_equal(plain, supersampled)


def test_supersample_blends_a_cell_straddling_a_hard_edge():
    # A boundary that lands inside a single output cell: supersampling
    # should produce something *between* the two colours there, not one
    # colour or the other outright -- proof it's actually averaging several
    # samples, not just picking a different single one.
    rgba = np.zeros((60, 60, 4), dtype=np.uint8)
    rgba[:, :33] = [20, 60, 140, 255]  # ocean
    rgba[:, 33:] = [180, 140, 90, 255]  # land
    # Output cell 2 of 6 (columns 20-30) straddles the boundary at column 33? no --
    # pick geometry so the boundary at col 33 falls inside cell index 3 (cols 30-40).
    small = downsample_rgba(rgba, 6, method="nearest", supersample=4)
    straddling = small[0, 3, :3].astype(int)
    assert not np.array_equal(straddling, [20, 60, 140])
    assert not np.array_equal(straddling, [180, 140, 90])


def test_supersample_reduces_frame_to_frame_flicker_at_a_shifting_edge():
    # The actual bug: as a boundary sweeps a fraction of a pixel between
    # frames, single-sample nearest can flip a cell fully between two
    # colours; supersampling should change that cell's colour far less.
    def make(edge_col):
        rgba = np.zeros((60, 60, 4), dtype=np.uint8)
        rgba[:, :edge_col] = [20, 60, 140, 255]
        rgba[:, edge_col:] = [180, 140, 90, 255]
        return rgba

    cell = 3  # covers source columns 30-40 at size=6 on a 60-wide source
    plain_deltas = []
    super_deltas = []
    for edge_col in (34, 35, 36, 37, 38):
        base_plain = downsample_rgba(make(34), 6, method="nearest", supersample=1)[0, cell, :3].astype(int)
        base_super = downsample_rgba(make(34), 6, method="nearest", supersample=4)[0, cell, :3].astype(int)
        plain = downsample_rgba(make(edge_col), 6, method="nearest", supersample=1)[0, cell, :3].astype(int)
        supersampled = downsample_rgba(make(edge_col), 6, method="nearest", supersample=4)[0, cell, :3].astype(int)
        plain_deltas.append(np.abs(plain - base_plain).max())
        super_deltas.append(np.abs(supersampled - base_super).max())

    assert max(super_deltas) <= max(plain_deltas)


# ------------------------------------------------------------------ palette


def test_quantize_palette_respects_color_count():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    quantized = quantize_palette(rgb, colors=8)
    assert len(np.unique(quantized.reshape(-1, 3), axis=0)) <= 8


def test_quantize_palette_output_shape_matches_input():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    quantized = quantize_palette(rgb, colors=4)
    assert quantized.shape == rgb.shape


def test_shared_palette_maps_the_same_colour_identically_across_frames():
    # The actual bug this exists to fix: two frames showing different parts
    # of the globe discover slightly different optimal palettes
    # independently, so an identical true colour can snap to a different
    # swatch in each -- flicker, even with nothing about the colour itself
    # changing. A shared palette must not let that happen.
    rng = np.random.default_rng(4)
    shared_pixel = np.array([90, 140, 60], dtype=np.uint8)  # present in every frame

    frame_a = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    frame_a[0, 0] = shared_pixel
    frame_b = rng.integers(100, 256, size=(20, 20, 3), dtype=np.uint8)  # different colour range
    frame_b[0, 0] = shared_pixel

    palette = build_shared_palette([frame_a, frame_b], colors=16)
    quantized_a = quantize_to_palette(frame_a, palette)
    quantized_b = quantize_to_palette(frame_b, palette)

    assert tuple(quantized_a[0, 0]) == tuple(quantized_b[0, 0])


def test_independent_palettes_can_map_the_same_colour_differently():
    # The contrast case: without a shared palette, the same true colour can
    # legitimately quantize to different swatches in different images --
    # confirms the shared-palette test above is actually testing something.
    rng = np.random.default_rng(4)
    shared_pixel = np.array([90, 140, 60], dtype=np.uint8)

    frame_a = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    frame_a[0, 0] = shared_pixel
    frame_b = rng.integers(100, 256, size=(20, 20, 3), dtype=np.uint8)
    frame_b[0, 0] = shared_pixel

    quantized_a = quantize_palette(frame_a, colors=16)
    quantized_b = quantize_palette(frame_b, colors=16)

    assert tuple(quantized_a[0, 0]) != tuple(quantized_b[0, 0])


def test_grade_and_downsample_then_finish_matches_pixelate():
    size = 60
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - 30) ** 2 + (xx - 30) ** 2 <= 25**2
    rgba[disc] = [80, 120, 60, 255]

    settings = PixelArtSettings(size=16, colors=8, display_scale=2)
    direct = pixelate(rgba, settings)
    split = finish(grade_and_downsample(rgba, settings), settings)

    assert np.array_equal(direct, split)


# --------------------------------------------------------------- orchestrator


def test_pixelate_end_to_end_shape_and_alpha():
    size = 200
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - 100) ** 2 + (xx - 100) ** 2 <= 90**2
    rgba[disc] = [40, 90, 160, 255]
    rgba[disc & (xx > 140)] = [200, 180, 140, 255]

    settings = PixelArtSettings(size=32, stylize=0.7, colors=16, display_scale=4)
    out = pixelate(rgba, settings)

    assert out.shape == (128, 128, 4)
    assert out.dtype == np.uint8
    assert out[..., 3].max() > 0  # something is opaque
    assert out[0, 0, 3] == 0  # corners, off the disc, stay transparent


def test_pixelate_respects_palette_size_after_upscale():
    size = 100
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - 50) ** 2 + (xx - 50) ** 2 <= 45**2
    rng = np.random.default_rng(3)
    rgba[disc, :3] = rng.integers(0, 256, size=(disc.sum(), 3))
    rgba[disc, 3] = 255

    settings = PixelArtSettings(size=24, colors=12, display_scale=2)
    out = pixelate(rgba, settings)
    on = out[out[..., 3] > 0][:, :3]
    assert len(np.unique(on, axis=0)) <= 12
