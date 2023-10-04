from drt_1d import (
    calculate_distance,
    calculate_macroscopic_cross_section,
    calculate_optical_distance,
    calculate_interaction_probability,
)
import math


def test_calculate_distance():
    assert calculate_distance(0, 5) == 5
    assert calculate_distance(10, 7) == 3
    assert calculate_distance(0, 0) == 0  # Edge case: both points are the same
    assert calculate_distance(-5, 5) == 10  # Edge case: negative coordinates


def test_calculate_macroscopic_cross_section():
    assert calculate_macroscopic_cross_section(2, 3) == 5
    assert calculate_macroscopic_cross_section(0, 0) == 0
    # Edge case: one cross-section is zero
    assert calculate_macroscopic_cross_section(0, 5) == 5
    # Edge case: negative cross-sections
    assert calculate_macroscopic_cross_section(-2, 3) == 1


def test_calculate_optical_distance():
    assert calculate_optical_distance(1, 10) == 10
    assert calculate_optical_distance(0, 5) == 0
    # Edge case: zero attenuation coefficient
    assert calculate_optical_distance(0, 0) == 0
    # Edge case: negative distance
    assert calculate_optical_distance(2, -10) == -20


def test_calculate_interaction_probability():
    # Edge case: zero optical distance
    assert calculate_interaction_probability(0) == 1
    assert calculate_interaction_probability(10) == math.exp(-10)
    # Edge case: negative optical distance
    assert calculate_interaction_probability(-5) == math.exp(5)


if __name__ == "__main__":
    import pytest
    pytest.main()
