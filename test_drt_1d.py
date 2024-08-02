from drt_1d import Region, Domain, read_input, perform_calculation
import numpy as np
import scipy.special as sc
import pytest

def test_region_properties():
    """
    Test properties and methods of the Region class.
    """
    region = Region(width=10.0, num_voxels=5, scattering_cross_section=0.2, positions_per_voxel=3, position_location_str='evenly-spaced')
    assert region.voxel_width == 2.0
    positions = region.generate_positions(start_x=0.0)
    assert len(positions) == 15  # 5 voxels, 3 positions each

def test_domain_initialization():
    """
    Test initialization and properties of the Domain class.
    """
    region1 = Region(width=10.0, num_voxels=5, scattering_cross_section=0.2, positions_per_voxel=3, position_location_str='evenly-spaced')
    region2 = Region(width=20.0, num_voxels=10, scattering_cross_section=0.1, positions_per_voxel=2, position_location_str='random')
    domain = Domain(start_x=0.0, regions=[region1, region2])
    
    assert domain.num_voxels == 15
    assert len(domain.positions) == 35  # 5*3 + 10*2

def test_source_term():
    """
    Test construction of the source term vector.
    """
    region1 = Region(width=10.0, num_voxels=5, scattering_cross_section=0.2, positions_per_voxel=3, position_location_str='evenly-spaced')
    region2 = Region(width=20.0, num_voxels=10, scattering_cross_section=0.1, positions_per_voxel=2, position_location_str='random')
    domain = Domain(start_x=0.0, regions=[region1, region2])
    source_term = domain.construct_source_term(source_voxel_index=2, source_strength=1.0)
    assert source_term[2] == 1.0
    assert np.sum(source_term) == 1.0

def test_optical_thickness():
    """
    Test computation of optical thickness.
    """
    region1 = Region(width=10.0, num_voxels=5, scattering_cross_section=0.2, positions_per_voxel=3, position_location_str='evenly-spaced')
    region2 = Region(width=20.0, num_voxels=10, scattering_cross_section=0.1, positions_per_voxel=2, position_location_str='random')
    domain = Domain(start_x=0.0, regions=[region1, region2])
    
    tau = domain.compute_optical_thickness(x1=1.0, x2=15.0)
    expected_tau = (0.2 * 9.0) + (0.1 * 5.0)  # part of region1 and part of region2
    assert np.isclose(tau, expected_tau)

def test_streaming_operator():
    """
    Test computation of streaming operator.
    """
    region1 = Region(width=10.0, num_voxels=5, scattering_cross_section=0.2, positions_per_voxel=3, position_location_str='evenly-spaced')
    region2 = Region(width=20.0, num_voxels=10, scattering_cross_section=0.1, positions_per_voxel=2, position_location_str='random')
    domain = Domain(start_x=0.0, regions=[region1, region2])
    
    operator = domain.compute_streaming_operator(start_position=1.0, end_position=15.0)
    tau = (0.2 * 9.0) + (0.1 * 5.0)
    expected_operator = (1/2) * sc.exp1(tau)
    assert np.isclose(operator, expected_operator)

def test_transition_matrix():
    """
    Test construction of the transition matrix.
    """
    region1 = Region(width=10.0, num_voxels=5, scattering_cross_section=0.2, positions_per_voxel=3, position_location_str='evenly-spaced')
    region2 = Region(width=20.0, num_voxels=10, scattering_cross_section=0.1, positions_per_voxel=2, position_location_str='random')
    domain = Domain(start_x=0.0, regions=[region1, region2])
    
    K_trans = domain.construct_transition_matrix()
    assert K_trans.shape == (15, 15)

def test_neutron_flux():
    """
    Test calculation of the neutron flux vector.
    """
    region1 = Region(width=10.0, num_voxels=5, scattering_cross_section=0.2, positions_per_voxel=3, position_location_str='evenly-spaced')
    region2 = Region(width=20.0, num_voxels=10, scattering_cross_section=0.1, positions_per_voxel=2, position_location_str='random')
    domain = Domain(start_x=0.0, regions=[region1, region2])
    
    phi = domain.compute_neutron_flux(source_voxel_index=2, source_strength=1.0)
    assert len(phi) == 15

def test_read_input():
    """
    Test reading input data from a file.
    """
    with pytest.raises(FileNotFoundError):
        read_input('nonexistent_file.json')

def test_perform_calculation():
    """
    Test the perform_calculation function.
    """
    input_data = {
        'start_x': 0.0,
        'regions': [
            {'width': 10.0, 'num_voxels': 5, 'positions_per_voxel': 3, 'position_location': 'evenly-spaced', 'scattering_cross_section': 0.2},
            {'width': 20.0, 'num_voxels': 10, 'positions_per_voxel': 2, 'position_location': 'random', 'scattering_cross_section': 0.1}
        ],
        'source': {'voxel_index': 2, 'strength': 1.0}
    }
    
    phi = perform_calculation(input_data['start_x'], input_data['regions'], input_data['source'])
    assert len(phi) == 15

if __name__ == "__main__":
    import pytest
    pytest.main()
