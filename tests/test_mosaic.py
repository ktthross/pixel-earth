"""Per-viewpoint compositing: selection never mixes channels across frames."""

from datetime import date

import numpy as np
import pytest

from conftest import synthetic_viewpoint_frame
from pixel_earth.catalog import FrameGeometry
from pixel_earth.cloud_score import rgb_heuristic_score
from pixel_earth.epic import Frame
from pixel_earth.geometry import forward
from pixel_earth.mosaic import (
    Candidate,
    build_reference_grid,
    make_viewpoint,
    render_viewpoint,
    render_viewpoint_from_reference,
    sample_reference_grid,
)

QUADRANTS = {
    "NW": (20, 60, 140),  # ocean blue
    "NE": (160, 140, 90),  # land tan
    "SW": (30, 90, 40),  # forest green
    "SE": (200, 170, 120),  # desert
}


def _candidate(name: str, viewpoint, *, cloudy_quadrants=(), colors=QUADRANTS) -> Candidate:
    """A candidate at exactly the viewpoint's own geometry -- isolates
    selection/blending behaviour from the reprojection math, which
    tests/test_geometry.py already covers on its own."""
    frame = Frame(
        collection="natural", image=name, archive_day=date(2024, 1, 1), captured=None,
        lat=viewpoint.lat0, lon=viewpoint.lon0,
    )
    geometry = FrameGeometry(
        frame=frame, cx=viewpoint.cx, cy=viewpoint.cy, radius=viewpoint.radius, looks_like_disc=True
    )
    rgb = synthetic_viewpoint_frame(
        viewpoint.size, viewpoint.cx, viewpoint.cy, viewpoint.radius,
        quadrant_colors=colors, cloudy_quadrants=cloudy_quadrants,
    )
    return Candidate(geometry=geometry, rgb=rgb)


def _quadrant_pixel(viewpoint, name: str) -> tuple[int, int]:
    """A representative interior pixel of one quadrant, away from any edge."""
    offset = int(viewpoint.radius * 0.4)
    row = int(viewpoint.cy) + (-offset if name[0] == "N" else offset)
    col = int(viewpoint.cx) + (-offset if name[1] == "W" else offset)
    return row, col


@pytest.fixture
def viewpoint():
    return make_viewpoint(0.0, 0.0, radius=60)


def test_hard_selection_recombines_clear_quadrants_from_different_frames(viewpoint):
    # Each frame is cloudy over a different quadrant -- exactly the "same
    # view, different day" case a multi-day composite exists to fix.
    frame_a = _candidate("a", viewpoint, cloudy_quadrants=("NW", "NE"))
    frame_b = _candidate("b", viewpoint, cloudy_quadrants=("SW", "SE"))

    result = render_viewpoint(viewpoint, [frame_a, frame_b], scorer=rgb_heuristic_score)

    for name, expected in QUADRANTS.items():
        row, col = _quadrant_pixel(viewpoint, name)
        assert tuple(result.rgba[row, col, :3]) == expected


def test_hard_selection_never_synthesises_a_channel_wise_mix(viewpoint):
    cloudy = _candidate("cloudy", viewpoint, cloudy_quadrants=("NW", "NE", "SW", "SE"))
    clear = _candidate("clear", viewpoint)

    result = render_viewpoint(viewpoint, [cloudy, clear], scorer=rgb_heuristic_score)

    on_disc = result.rgba[..., 3] > 0
    colors = np.unique(result.rgba[on_disc][:, :3], axis=0)
    expected = np.array(sorted(QUADRANTS.values()))
    assert np.array_equal(np.array(sorted(map(tuple, colors))), expected)


def test_all_candidates_cloudy_still_picks_the_least_cloudy(viewpoint):
    # A pale-cloud frame and a solid-white frame: both "cloudy", but the pale
    # one is the better of two bad options and should win outright.
    pale = _candidate(
        "pale", viewpoint, cloudy_quadrants=(), colors={k: (235, 235, 235) for k in QUADRANTS}
    )
    solid_white = _candidate(
        "solid", viewpoint, cloudy_quadrants=(), colors={k: (255, 255, 255) for k in QUADRANTS}
    )

    result = render_viewpoint(viewpoint, [solid_white, pale], scorer=rgb_heuristic_score)

    row, col = _quadrant_pixel(viewpoint, "NW")
    assert tuple(result.rgba[row, col, :3]) == (235, 235, 235)
    assert not result.suspect[row, col]  # this interior pixel had two usable, if cloudy, options


def test_no_visible_candidate_anywhere_is_suspect_and_transparent():
    viewpoint = make_viewpoint(0.0, 0.0, radius=60)
    result = render_viewpoint(viewpoint, [], scorer=rgb_heuristic_score)

    on_disc = result.suspect | (result.rgba[..., 3] > 0)
    assert on_disc.any()
    assert result.suspect.all() == on_disc.all()  # every on-disc pixel is suspect
    assert (result.rgba[..., 3] == 0).all()
    assert result.suspect_fraction == pytest.approx(1.0)


def test_luminance_floor_rejects_a_merely_dark_pixel(viewpoint):
    # Uniformly near-black -- clear by the whiteness/brightness score alone,
    # but too dark to trust as "clear" rather than "underexposed".
    too_dark = _candidate(
        "dark", viewpoint, colors={k: (5, 5, 5) for k in QUADRANTS}
    )
    normal = _candidate("normal", viewpoint)

    result = render_viewpoint(
        viewpoint, [too_dark, normal], scorer=rgb_heuristic_score, luminance_floor=30
    )

    row, col = _quadrant_pixel(viewpoint, "NW")
    assert tuple(result.rgba[row, col, :3]) == QUADRANTS["NW"]


def test_blend_k_produces_a_legitimate_convex_combination(viewpoint):
    # Two candidates, both clear, differing only by a small uniform offset --
    # a near-tie that should blend rather than hard-switch.
    base = _candidate("base", viewpoint)
    shifted = _candidate(
        "shifted", viewpoint, colors={k: tuple(min(255, c + 10) for c in v) for k, v in QUADRANTS.items()}
    )

    result = render_viewpoint(
        viewpoint, [base, shifted], scorer=rgb_heuristic_score, blend_k=2, blend_margin=2.0
    )

    row, col = _quadrant_pixel(viewpoint, "NW")
    low = np.array(QUADRANTS["NW"])
    high = low + 10
    blended = result.rgba[row, col, :3].astype(np.int32)
    # A real convex combination of the two triplets: between them channel by
    # channel, not below the lower or above the upper on any channel (which
    # would indicate a cross-channel mix had crept back in).
    assert np.all(blended >= low) and np.all(blended <= high)


def test_blend_k_one_matches_hard_selection(viewpoint):
    cloudy = _candidate("cloudy", viewpoint, cloudy_quadrants=("NW",))
    clear = _candidate("clear", viewpoint)

    hard = render_viewpoint(viewpoint, [cloudy, clear], scorer=rgb_heuristic_score, blend_k=1)
    row, col = _quadrant_pixel(viewpoint, "NW")
    assert tuple(hard.rgba[row, col, :3]) == QUADRANTS["NW"]


def test_off_disc_pixels_are_transparent_and_not_suspect(viewpoint):
    clear = _candidate("clear", viewpoint)
    result = render_viewpoint(viewpoint, [clear], scorer=rgb_heuristic_score)

    corner = result.rgba[0, 0]
    assert corner[3] == 0
    assert not result.suspect[0, 0]


def test_contributor_indexes_the_winning_candidate(viewpoint):
    cloudy = _candidate("cloudy", viewpoint, cloudy_quadrants=("NW", "NE", "SW", "SE"))
    clear = _candidate("clear", viewpoint)

    result = render_viewpoint(viewpoint, [cloudy, clear], scorer=rgb_heuristic_score)
    row, col = _quadrant_pixel(viewpoint, "NW")
    assert result.contributor[row, col] == 1  # index of `clear` in the candidates list


# ------------------------------------------------------------ reference grid


def test_reference_grid_picks_the_least_cloudy_candidate():
    cloudy = _candidate("cloudy", make_viewpoint(0.0, 0.0, radius=60), cloudy_quadrants=("NW", "NE", "SW", "SE"))
    clear = _candidate("clear", make_viewpoint(0.0, 0.0, radius=60))

    grid = build_reference_grid(360, 180, [cloudy, clear], scorer=rgb_heuristic_score)

    # (lat 0, lon 0) is dead centre of both candidates' discs, in the NW/NE/SW/SE
    # boundary; nudge slightly into the NW quadrant instead.
    row = int((90.0 - 20.0) / 180.0 * 180)
    col = int((-20.0 + 180.0) / 360.0 * 360)
    assert tuple(grid.rgb[row, col]) == QUADRANTS["NW"]
    assert grid.has_data[row, col]


def test_reference_grid_has_no_data_where_nothing_is_visible():
    grid = build_reference_grid(360, 180, [], scorer=rgb_heuristic_score)
    assert not grid.has_data.any()


def test_sample_reference_grid_agrees_regardless_of_which_viewpoint_asks():
    # The whole point: the same physical location must resolve to the same
    # colour no matter which rotation frame's geometry is asking for it --
    # that's what removes cross-frame flicker.
    cloudy = _candidate("cloudy", make_viewpoint(0.0, 0.0, radius=80), cloudy_quadrants=("NW", "NE", "SW", "SE"))
    clear = _candidate("clear", make_viewpoint(0.0, 0.0, radius=80))
    grid = build_reference_grid(720, 360, [cloudy, clear], scorer=rgb_heuristic_score)

    lat, lon = 15.0, 15.0  # NE quadrant, well inside both candidates' discs

    for lon0 in (0.0, 40.0, 200.0, -150.0):
        rgb, has_data = sample_reference_grid(grid, np.array([lat]), np.array([lon]))
        if lon0 == 0.0:
            first_rgb = rgb
        assert has_data[0]
        assert np.allclose(rgb, first_rgb, atol=1.0)


def test_sample_reference_grid_wraps_the_antimeridian():
    # A cell just past +180 should read the same as the equivalent cell just
    # past -180 -- there is no seam, even though the grid array itself ends.
    # A flat-colour candidate (no internal quadrant edge) isolates the wrap
    # behaviour from bilinear blending across a nearby hard colour boundary.
    flat = _candidate("flat", make_viewpoint(0.0, 179.5, radius=80), colors={k: (90, 140, 60) for k in QUADRANTS})
    grid = build_reference_grid(720, 360, [flat], scorer=rgb_heuristic_score)

    rgb_pos, has_pos = sample_reference_grid(grid, np.array([15.0]), np.array([179.9]))
    rgb_neg, has_neg = sample_reference_grid(grid, np.array([15.0]), np.array([-179.9]))
    assert has_pos[0] and has_neg[0]
    assert np.allclose(rgb_pos, rgb_neg, atol=2.0)


def test_sample_reference_grid_reports_no_data_far_from_any_candidate():
    clear = _candidate("clear", make_viewpoint(0.0, 0.0, radius=80))
    grid = build_reference_grid(360, 180, [clear], scorer=rgb_heuristic_score)

    _, has_data = sample_reference_grid(grid, np.array([0.0]), np.array([175.0]))
    assert not has_data[0]


def test_render_viewpoint_from_reference_is_identical_across_rotation_frames():
    # The actual bug this exists to fix: render_viewpoint (independent per
    # frame) can flip its winning candidate for the same physical point
    # between nearby viewpoints when two candidates are close in score.
    # render_viewpoint_from_reference must not, since the decision was
    # already made once, in the grid.
    near_tie_a = _candidate("a", make_viewpoint(0.0, 0.0, radius=100), colors={k: (140, 145, 150) for k in QUADRANTS})
    near_tie_b = _candidate("b", make_viewpoint(0.0, 0.0, radius=100), colors={k: (141, 144, 149) for k in QUADRANTS})
    grid = build_reference_grid(720, 360, [near_tie_a, near_tie_b], scorer=rgb_heuristic_score)

    lat, lon = 10.0, 10.0
    samples = []
    for lon0 in (0.0, 15.0, 30.0, 45.0, 60.0):
        vp = make_viewpoint(0.0, lon0, radius=200)
        col, row, cos_c = forward(np.array([lat]), np.array([lon]), geometry=vp.geometry)
        assert cos_c[0] > 0  # sanity: the point is actually on this viewpoint's disc
        result = render_viewpoint_from_reference(vp, grid)
        samples.append(tuple(result.rgba[round(float(row[0])), round(float(col[0])), :3]))

    assert len(set(samples)) == 1  # every frame agrees on this point's colour
