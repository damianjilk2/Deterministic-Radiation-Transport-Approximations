from drt_1d import (
    construct_source_term,
    calculate_K_trans,
    construct_transition_matrix,
    calculate_phi,
    read_input
)
import numpy as np
import scipy.special as sc
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
    
    # Normal case going right
    sigma_s = [0.2, 0.1, 0.2, 0.1, 0.3, 0.2]
    K = calculate_K_trans(1, 2, 2.0, 5.0, 2.0, sigma_s)
    # expected_K = np.exp(-((0.1 + (0.5*0.2))*2.0)) / (4 * np.pi * (5.0 - 2.0)**2) NOTE: previous streaming operator calculation
    expected_K = (1/2)*sc.exp1((0.1 + (0.5*0.2))*2.0)
    np.testing.assert_almost_equal(K, expected_K, decimal=5)
    
    # Normal case going left
    sigma_s = [0.2, 0.1, 0.2, 0.1, 0.3, 0.2]
    K = calculate_K_trans(2, 1, 5.0, 2.0, 2.0, sigma_s)
    # expected_K = np.exp(-((0.1 + (0.5*0.2))*2.0)) / (4 * np.pi * (5.0 - 2.0)**2) NOTE: previous streaming operator calculation
    expected_K = (1/2)*sc.exp1((0.1 + (0.5*0.2))*2.0)
    np.testing.assert_almost_equal(K, expected_K, decimal=5)
    
    # Edge case: Zero voxel size
    with pytest.raises(ValueError):
        calculate_K_trans(1, 4, 2.0, 6.0, 0.0, sigma_s)
    
    # Edge case: Negative ri
    with pytest.raises(ValueError):
        calculate_K_trans(1, 4, -2.0, 6.0, 2.0, sigma_s)
    
    # Edge case: Negative rj
    with pytest.raises(ValueError):
        calculate_K_trans(1, 4, 2.0, -6.0, 2.0, sigma_s)
    
    # Edge case: ri out of range
    with pytest.raises(ValueError):
        calculate_K_trans(10, 4, 20.0, 6.0, 2.0, sigma_s)
    
    # Edge case: rj out of range
    with pytest.raises(ValueError):
        calculate_K_trans(1, 10, 2.0, 60.0, 2.0, sigma_s)
        
    #TODO: add testing for invalid indicies (starting and ending voxels)

def test_construct_transition_matrix():
    """Test construction of the transition matrix."""
    
    # Normal case
    sigma_s = [0.2, 0.1, 0.2, 0.1, 0.3]
    positions = {"0": [1.0,1.005,1.9],"1": [2.0,2.2,2.3],"2": [4.5],"3": [7.0],"4": [10.0]}
    K_trans = construct_transition_matrix(5, 2.0, sigma_s, positions)
    assert K_trans.shape == (5, 5)
    
    # Edge case: Zero num_voxels
    with pytest.raises(ValueError):
        construct_transition_matrix(0, 2.0, sigma_s, positions)
    
    # Edge case: Zero voxel size
    with pytest.raises(ValueError):
        construct_transition_matrix(5, 0.0, sigma_s, positions)

def test_calculate_phi():
    """Test calculation of the neutron flux vector."""
    
    # Normal case
    sigma_s = [0.2, 0.1, 0.2, 0.1, 0.3]
    positions = {"0": [1.0,1.005,1.9],"1": [2.0,2.2,2.3],"2": [4.5],"3": [7.0],"4": [10.0]}
    K_trans = construct_transition_matrix(5, 2.0, sigma_s, positions)
    s = construct_source_term(5, 2)
    phi = calculate_phi(K_trans, sigma_s, s)
    assert len(phi) == 5
    
    # Edge case: Zero source term
    s_zero = np.zeros(5)
    phi_zero = calculate_phi(K_trans, sigma_s, s_zero)
    assert np.all(phi_zero == 0)

def test_read_input():
    """Test reading input file"""
    
    with pytest.raises(FileNotFoundError):
        read_input('nonexistent_file.txt')  # Test case for non-existent file

if __name__ == "__main__":
    import pytest
    pytest.main()
