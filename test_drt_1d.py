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
    """
    Test reading input file
    """
    # Edge case: non-existent file
    with pytest.raises(FileNotFoundError):
        read_input('nonexistent_file.txt')

def test_validate_input():
    """
    Test the validate_input function.
    """
    # Normal case
    input_data = {
        'start_x': 0.0,
        'regions': [
            {'name': 'Region1', 'width': 2.0, 'num_voxels': 1, 'positions_per_voxel': 2,
             'position_location': 'evenly-spaced', 'scattering_cross_section': 0.3},
            {'name': 'Region2', 'width': 3.0, 'num_voxels': 2, 'positions_per_voxel': 1,
             'position_location': 'random', 'scattering_cross_section': 0.2},
            {'name': 'Region3', 'width': 5.0, 'num_voxels': 3, 'positions_per_voxel': 1,
             'position_location': 'evenly-spaced', 'scattering_cross_section': 0.1}
        ],
        'source': {'voxel_index': 3, 'strength': 1.0}
    }
    validate_input(input_data)
    
    # Edge case: Non-list regions
    with pytest.raises(ValueError):
        validate_input({'start_x': 0.0, 'regions': {}, 'source': input_data['source']})
    
    # Edge case: Region missing required keys
    with pytest.raises(ValueError):
        validate_input({'start_x': 0.0, 'regions': [{'name': 'Region1', 'width': 2.0}], 'source': input_data['source']})
    
    # Edge case: Negative voxel index in source
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['source']['voxel_index'] = -1
        validate_input(input_data_error)
    
    # Edge case: Non-numeric start_x
    with pytest.raises(ValueError):
        validate_input({'start_x': 'invalid', 'regions': input_data['regions'], 'source': input_data['source']})
    
    # Edge case: Non-numeric width in region
    with pytest.raises(ValueError):
        input_data_error = copy.deepcopy(input_data)
        input_data_error['regions'][0]['width'] = 'invalid'
        validate_input(input_data_error)

if __name__ == "__main__":
    import pytest
    pytest.main()
