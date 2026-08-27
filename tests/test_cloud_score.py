"""RGB cloud heuristic: bright+white scores high, saturated colour scores low."""

import numpy as np
import pytest

from pixel_earth.cloud_score import cloudfraction_score, rgb_heuristic_score


def test_bright_white_scores_high():
    white = np.array([[[250, 250, 250]]], dtype=np.uint8)
    score = rgb_heuristic_score(white)
    assert score[0, 0] > 0.9


def test_saturated_ocean_blue_scores_low():
    ocean = np.array([[[20, 60, 140]]], dtype=np.uint8)
    score = rgb_heuristic_score(ocean)
    assert score[0, 0] < 0.3


def test_black_scores_zero():
    black = np.array([[[0, 0, 0]]], dtype=np.uint8)
    score = rgb_heuristic_score(black)
    assert score[0, 0] == pytest.approx(0.0)


def test_cloud_scores_higher_than_ocean():
    rgb = np.array([[[240, 245, 250], [20, 60, 140]]], dtype=np.uint8)
    score = rgb_heuristic_score(rgb)
    assert score[0, 0] > score[0, 1]


def test_land_is_bright_but_not_as_white_as_cloud():
    rgb = np.array([[[240, 245, 250], [190, 160, 110]]], dtype=np.uint8)  # cloud, sandy land
    score = rgb_heuristic_score(rgb)
    assert score[0, 0] > score[0, 1]


def test_output_shape_matches_input_leading_dims():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    score = rgb_heuristic_score(rgb)
    assert score.shape == (4, 5)


def test_cloudfraction_score_is_a_documented_stub():
    with pytest.raises(NotImplementedError):
        cloudfraction_score(np.zeros((1, 1, 3), dtype=np.uint8))
