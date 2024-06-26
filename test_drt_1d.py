from drt_1d import (
    construct_source_term,
    calculate_K_trans,
    construct_transition_matrix,
    calculate_neutron_flux,
    read_input,
    validate_input,
    Voxel
)
import numpy as np
import scipy.special as sc
import pytest
import copy

def test_construct_source_term():
    """
    Test the construction of the source term vector.
    """
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
    
def test_calculate_K_trans():
    """
    Test calculation of transition probability between voxel centers.
    """
    voxels = [
        Voxel(0, 0.0, 2.0, 0.2, [0.0,0.1,0.4]),
        Voxel(1, 2.0, 4.0, 0.1, [2.2,3.6]),
        Voxel(2, 4.0, 6.0, 0.2, [4.0]),
        Voxel(3, 6.0, 8.0, 0.1, [6.0]),
        Voxel(4, 8.0, 10.0, 0.3, [8.0])
    ]
    # Normal case going right
    K = calculate_K_trans(0.0, 2.2, voxels[0:2])
    expected_K = (1/2)*sc.exp1((2*0.2)+(0.2*0.1))
    np.testing.assert_almost_equal(K, expected_K, decimal=5)
    
    # Normal case going left
    K = calculate_K_trans(2.2, 0.0, voxels[0:2])
    np.testing.assert_almost_equal(K, expected_K, decimal=5)
    
    # Edge case: Zero voxel size
    voxels_empty = []
    with pytest.raises(ValueError):
        calculate_K_trans(0.0, 2.2, voxels_empty)
        
    #TODO: add testing for invalid indicies (starting and ending voxels)

def test_construct_transition_matrix():
    """
    Test construction of the transition matrix.
    """
    voxels = [
        Voxel(0, 0.0, 2.0, 0.2, [0.0,0.1,0.4]),
        Voxel(1, 2.0, 4.0, 0.1, [2.2,3.6]),
        Voxel(2, 4.0, 6.0, 0.2, [4.0]),
        Voxel(3, 6.0, 8.0, 0.1, [6.0]),
        Voxel(4, 8.0, 10.0, 0.3, [8.0])
    ]
    # Normal case
    K_trans = construct_transition_matrix(voxels)
    assert K_trans.shape == (5, 5)
    
    # Edge case: Empty voxels list
    with pytest.raises(ValueError):
        construct_transition_matrix([])

def test_calculate_neutron_flux():
    """
    Test calculation of the neutron flux vector.
    """
    voxels = [
        Voxel(0, 0.0, 2.0, 0.2, [0.0,0.1,0.4]),
        Voxel(1, 2.0, 4.0, 0.1, [2.2,3.6]),
        Voxel(2, 4.0, 6.0, 0.2, [4.0]),
        Voxel(3, 6.0, 8.0, 0.1, [6.0]),
        Voxel(4, 8.0, 10.0, 0.3, [8.0])
    ]
    # Normal case
    K_trans = construct_transition_matrix(voxels)
    s = construct_source_term(5, 2)
    sigma_s = [0.2, 0.1, 0.2, 0.1, 0.3]
    phi = calculate_neutron_flux(K_trans, sigma_s, s)
    assert len(phi) == 5
    
    # Edge case: Zero source term
    s_zero = np.zeros(5)
    phi_zero = calculate_neutron_flux(K_trans, sigma_s, s_zero)
    assert np.all(phi_zero == 0)

def test_read_input():
    """Test reading input file"""
    
    with pytest.raises(FileNotFoundError):
        read_input('nonexistent_file.txt')  # Test case for non-existent file

def test_validate_input():
    """
    Test the validate_input function.
    """
    # Normal case
    input_data = {
        'voxels': [
            {'index': 0, 'start_position': 0.0, 'end_position': 2.0, 'scattering_cross_section': 0.2, 'positions': [0.0, 0.1, 0.4]},
            {'index': 1, 'start_position': 2.0, 'end_position': 4.0, 'scattering_cross_section': 0.1, 'positions': [2.2, 3.6]},
            {'index': 2, 'start_position': 4.0, 'end_position': 6.0, 'scattering_cross_section': 0.2, 'positions': [4.0]},
            {'index': 3, 'start_position': 6.0, 'end_position': 8.0, 'scattering_cross_section': 0.1, 'positions': [6.0]},
            {'index': 4, 'start_position': 8.0, 'end_position': 10.0, 'scattering_cross_section': 0.3, 'positions': [8.0]}
        ],
        'source_voxel_index': 2
    }
    validate_input(input_data)  # Should not raise any errors
    
    # Edge case: Empty input data
    with pytest.raises(ValueError):
        validate_input({})
    
    # Edge case: Missing voxels key
    with pytest.raises(ValueError):
        validate_input({'source_voxel_index': 0})
    
    # Edge case: Non-list voxels
    with pytest.raises(ValueError):
        validate_input({'voxels': {}, 'source_voxel_index': 0})
    
    # Edge case: Voxel missing required keys
    with pytest.raises(ValueError):
        validate_input({'voxels': [{'index': 0, 'start_position': 0.0, 'end_position': 2.0}], 'source_voxel_index': 0})
    
    # Edge case: Negative voxel index
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['voxels'][0]['index'] = -1
        validate_input(input_data_error)
    
    # Edge case: Duplicate voxel index
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['voxels'][1]['index'] = 0  # Make index 0 appear twice
        validate_input(input_data_error)
    
    # Edge case: Invalid start and end positions
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['voxels'][0]['start_position'] = 2.0
        validate_input(input_data_error)
    
    # Edge case: Negative scattering cross-section
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['voxels'][0]['scattering_cross_section'] = -0.2
        validate_input(input_data_error)
    
    # Edge case: Non-list positions
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['voxels'][0]['positions'] = 0.0
        validate_input(input_data_error)
    
    # Edge case: Missing source voxel index
    with pytest.raises(ValueError):
        validate_input({'voxels': input_data['voxels']})
    
    # Edge case: Source voxel index out of bounds
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['source_voxel_index'] = 10
        validate_input(input_data_error)
        
    validate_input(input_data)

if __name__ == "__main__":
    import pytest
    pytest.main()
