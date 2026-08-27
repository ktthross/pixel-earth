"""Per-viewpoint compositing: selection never mixes channels across frames."""

from datetime import date

import numpy as np
import pytest

from conftest import synthetic_viewpoint_frame
from pixel_earth.catalog import FrameGeometry
from pixel_earth.cloud_score import rgb_heuristic_score
from pixel_earth.epic import Frame
from pixel_earth.mosaic import Candidate, make_viewpoint, render_viewpoint

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
