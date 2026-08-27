"""Orthographic projection: round trips and known-value sanity checks."""

import numpy as np
import pytest

from pixel_earth.geometry import Geometry, forward, great_circle_distance_deg, inverse


def _random_disc_pixels(geometry: Geometry, n: int, *, seed: int = 0, max_r: float = 0.999):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0, max_r, n) ** 0.5 * geometry.radius  # uniform over the disc's area
    col = geometry.cx + r * np.cos(theta)
    row = geometry.cy + r * np.sin(theta)
    return col, row


@pytest.mark.parametrize(
    "geometry",
    [
        Geometry(lat0=0.0, lon0=0.0, cx=1024.0, cy=1024.0, radius=800.0),
        Geometry(lat0=23.4, lon0=-120.0, cx=1024.0, cy=1024.0, radius=650.0),
        Geometry(lat0=-23.4, lon0=179.0, cx=500.0, cy=480.0, radius=300.0),
        Geometry(lat0=89.0, lon0=45.0, cx=1024.0, cy=1024.0, radius=800.0),
    ],
)
def test_round_trip_identity(geometry):
    col, row = _random_disc_pixels(geometry, 2000)
    lat, lon, visible = inverse(col, row, geometry=geometry)
    assert visible.all()

    col2, row2, cos_c = forward(lat, lon, geometry=geometry)
    assert np.allclose(col2, col, atol=1e-6)
    assert np.allclose(row2, row, atol=1e-6)
    assert (cos_c > 0).all()  # everything we started with was on the near hemisphere


def test_center_pixel_is_the_sub_satellite_point():
    geometry = Geometry(lat0=12.5, lon0=-64.0, cx=100.0, cy=80.0, radius=50.0)
    lat, lon, visible = inverse(np.array([100.0]), np.array([80.0]), geometry=geometry)
    assert visible[0]
    assert lat[0] == pytest.approx(12.5, abs=1e-9)
    assert lon[0] == pytest.approx(-64.0, abs=1e-9)


def test_off_disc_pixel_is_not_visible():
    geometry = Geometry(lat0=0.0, lon0=0.0, cx=100.0, cy=100.0, radius=50.0)
    # Twice the radius out -- nowhere near the disc.
    _, _, visible = inverse(np.array([300.0]), np.array([100.0]), geometry=geometry)
    assert not visible[0]


def test_far_hemisphere_point_has_negative_cos_c():
    geometry = Geometry(lat0=0.0, lon0=0.0, cx=100.0, cy=100.0, radius=50.0)
    # Straight round the back of the planet.
    _, _, cos_c = forward(np.array([0.0]), np.array([180.0]), geometry=geometry)
    assert cos_c[0] < 0


def test_equatorial_limb_point_sits_on_the_disc_edge():
    geometry = Geometry(lat0=0.0, lon0=0.0, cx=1000.0, cy=1000.0, radius=800.0)
    # 90 degrees of longitude off, on the equator: exactly the limb when lat0 is 0.
    col, row, cos_c = forward(np.array([0.0]), np.array([90.0]), geometry=geometry)
    assert cos_c[0] == pytest.approx(0.0, abs=1e-9)
    assert np.hypot(col[0] - geometry.cx, row[0] - geometry.cy) == pytest.approx(
        geometry.radius, abs=1e-6
    )


@pytest.mark.parametrize(
    "lat1, lon1, expected",
    [
        (0.0, 0.0, 0.0),
        (0.0, 90.0, 90.0),
        (0.0, 180.0, 180.0),
        (90.0, 0.0, 90.0),
        (-90.0, 0.0, 90.0),
    ],
)
def test_great_circle_distance_known_values(lat1, lon1, expected):
    d = great_circle_distance_deg(0.0, 0.0, np.array([lat1]), np.array([lon1]))
    assert d[0] == pytest.approx(expected, abs=1e-9)


def test_great_circle_distance_is_symmetric_and_zero_at_self():
    d_self = great_circle_distance_deg(12.0, -55.0, np.array([12.0]), np.array([-55.0]))
    assert d_self[0] == pytest.approx(0.0, abs=1e-9)

    a_to_b = great_circle_distance_deg(10.0, 20.0, np.array([30.0]), np.array([40.0]))
    b_to_a = great_circle_distance_deg(30.0, 40.0, np.array([10.0]), np.array([20.0]))
    assert a_to_b[0] == pytest.approx(b_to_a[0], abs=1e-9)
