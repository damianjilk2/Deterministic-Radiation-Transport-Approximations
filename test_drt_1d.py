from drt_1d import (
    construct_source_term,
    calculate_K_trans,
    construct_transition_matrix,
    calculate_neutron_flux,
)
import numpy as np
import pytest

def test_construct_source_term():
    """Test the construction of the source term vector."""
    
    # Normal case
    s = construct_source_term(6, 2)
    expected_s = np.zeros(6)
    expected_s[2] = 1
    np.testing.assert_array_equal(s, expected_s)
    
    # Edge case: Zero voxels
    with pytest.raises(ValueError):
        construct_source_term(0, 0)
    
    # Edge case: Negative index
    with pytest.raises(IndexError):
        construct_source_term(6, -1)
    
    # Edge case: Index out of bounds
    with pytest.raises(IndexError):
        construct_source_term(6, 10)
    
    # Edge case: Zero source strength
    s_zero_strength = construct_source_term(6, 2, 0)
    expected_s_zero_strength = np.zeros(6)
    np.testing.assert_array_equal(s_zero_strength, expected_s_zero_strength)
    
    # Edge case: Large number of voxels
    s_large = construct_source_term(10000, 5000)
    assert s_large[5000] == 1 and np.sum(s_large) == 1

def test_calculate_K_trans():
    """Test calculation of transition probability between voxel centers."""
    
    # Normal case
    sigma_s_matrix = np.diag([0.2, 0.1, 0.2, 0.1, 0.3, 0.2])
    K = calculate_K_trans(1, 4, 2.0, sigma_s_matrix)
    expected_K = np.exp(-((0.1+0.2+0.1)*2.0)) / (4 * np.pi * (3 * 2.0)**2)  # Expected transition probability
    np.testing.assert_almost_equal(K, expected_K, decimal=5)
    
    # Edge case: Zero voxel size
    with pytest.raises(ValueError):
        calculate_K_trans(1, 4, 0.0, sigma_s_matrix)
    
    # Edge case: Negative ri
    with pytest.raises(ValueError):
        calculate_K_trans(-1, 4, 2.0, sigma_s_matrix)
    
    # Edge case: Negative rj
    with pytest.raises(ValueError):
        calculate_K_trans(1, -1, 2.0, sigma_s_matrix)
    
    # Edge case: ri out of range
    with pytest.raises(ValueError):
        calculate_K_trans(10, 4, 2.0, sigma_s_matrix)
    
    # Edge case: rj out of range
    with pytest.raises(ValueError):
        calculate_K_trans(1, 10, 2.0, sigma_s_matrix)

def test_construct_transition_matrix():
    """Test construction of the transition matrix."""
    
    # Normal case
    sigma_s_matrix = np.diag([0.2, 0.1, 0.2, 0.1, 0.3, 0.2])
    K_trans = construct_transition_matrix(6, 2.0, sigma_s_matrix)
    assert K_trans.shape == (6, 6)
    
    # Edge case: Zero num_voxels
    with pytest.raises(ValueError):
        construct_transition_matrix(0, 2.0, sigma_s_matrix)
    
    # Edge case: Zero voxel size
    with pytest.raises(ValueError):
        construct_transition_matrix(6, 0.0, sigma_s_matrix)

def test_calculate_neutron_flux():
    """Test calculation of the neutron flux vector."""
    
    # Normal case
    sigma_s_matrix = np.diag([0.2, 0.1, 0.2, 0.1, 0.3, 0.2])
    K_trans = construct_transition_matrix(6, 2.0, sigma_s_matrix)
    s = construct_source_term(6, 2)
    phi = calculate_neutron_flux(K_trans, sigma_s_matrix, s)
    assert len(phi) == 6
    
    # Edge case: Zero source term
    s_zero = np.zeros(6)
    phi_zero = calculate_neutron_flux(K_trans, sigma_s_matrix, s_zero)
    assert np.all(phi_zero == 0)

if __name__ == "__main__":
    import pytest
    pytest.main()
