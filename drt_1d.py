"""
Title: Deterministic Radiation Transport - 1-D Toy Problem with Voxelized Geometry
Author: Damian Jilk
"""

import numpy as np
import json
import csv
import logging
import argparse
import scipy.special as sc
import random

class Region:
    def __init__(self, width:float, num_voxels:int, scattering_cross_section:float, positions_per_voxel:int, position_location_str:str):
        """
        Initializes a region with given properties.

        Parameters:
        - width (float): Width of the region.
        - num_voxels (int): Number of voxels in the region.
        - scattering_cross_section (float): Scattering cross section of the region.
        - positions_per_voxel (int): Number of positions per voxel.
        - position_location_str (str): Strategy for positioning ('evenly-spaced' or 'random').
        """
        self.width = width
        self.num_voxels = num_voxels
        self.scattering_cross_section = scattering_cross_section
        self.positions_per_voxel = positions_per_voxel
        self.position_location_str = position_location_str
        self.start_x = 0 # Initialize region start value.
        self.first_voxel_id = 0 # Initializes first voxel id.
        
    @property
    def voxel_width(self) -> float:
        """
        Computes the width of a single voxel.

        Returns:
        - float: Width of one voxel.
        """
        return self.width / self.num_voxels
    
    def get_local_voxel_id(self, x:float) -> int:
        """
        Get the voxel index relative to the scope of the region. 

        Parameters:
        - x (float): x-coordinate within the voxel.

        Returns:
            int: Local voxel index.
        """
        return int(np.floor( (x - self.start_x) / self.voxel_width))
    
    def get_global_voxel_id(self, x:float) -> int:
        """
        Get the voxel index relative to the entire global scope.

        Parameters:
        - x (float): x-coordinate within the voxel.

        Returns:
            int: Global voxel index.
        """
        return self.get_local_voxel_id(x) + self.first_voxel_id
    
    def generate_positions(self, voxel_id: int) -> np.ndarray:
        """
        Generates positions within a single voxel based on the specified strategy.

        Parameters:
        - voxel_id (int): Voxel index in which you wish to generate positions.

        Returns:
        - np.ndarray: Array of positions within the voxel.
        """
        voxel_start_x = self.start_x + (voxel_id - self.first_voxel_id) * self.voxel_width
        voxel_end_x = voxel_start_x + self.voxel_width
        if self.position_location_str == 'evenly-spaced':
            positions = np.linspace(voxel_start_x, voxel_end_x, self.positions_per_voxel + 2)[1:-1]
        elif self.position_location_str == 'random':
            positions = sorted(random.uniform(voxel_start_x, voxel_end_x) for _ in range(self.positions_per_voxel))
        return np.array(positions)

class Domain:
    def __init__(self, start_x:float, regions:list[Region]):
        """
        Initializes the domain with given starting position and regions.

        Parameters:
        - start_x (float): Starting x-coordinate of the domain.
        - regions (list): List of Region objects that make up the domain.
        """
        self.start_x = start_x
        self.regions = regions
        self.widths = np.array([region.width for region in regions])
        self.sigmas = np.array([region.scattering_cross_section for region in regions])
        self.cumsum_voxels = np.cumsum([0] + [region.num_voxels for region in self.regions])
        self.boundaries = np.cumsum([self.start_x] + [region.width for region in self.regions])
        
        # Set start_x and first_voxel_id for each region.
        for i, region in enumerate(self.regions):
            region.start_x = self.boundaries[i]
            region.first_voxel_id = self.cumsum_voxels[i]
        
    @property
    def num_voxels(self):
        return self.cumsum_voxels[-1]

    def get_region_id_from_position(self, x:float) -> int:
        """
        Determines the region ID for a given x-coordinate.
        
        Note: np.searchsorted uses binary search. 
        Additionally, this computation includes the left edge in the search but not the right edge.

        Parameters:
        - x (float): x-coordinate.

        Returns:
        - int: Region ID containing the x-coordinate.
        """
        return np.searchsorted(self.boundaries, x, side='right') - 1
    
    def get_region_from_voxel_index(self, voxel_index: int) -> int:
        """
        Determines the region ID for a given voxel index.

        Parameters:
        - voxel_index (int): Index of the voxel.

        Returns:
        - int: Region ID containing the voxel.
        """
        return np.searchsorted(self.cumsum_voxels, voxel_index, side='right') - 1
            
    def get_voxel_width(self, x:float) -> float:
        """
        Determines the width of the voxel containing a given x-coordinate.

        Parameters:
        - x (float): x-coordinate.

        Returns:
        - float: Width of the voxel containing the x-coordinate.
        """
        region_idx = self.get_region_id_from_position(x)
        return self.regions[region_idx].voxel_width
    
    def construct_source_term(self, source_voxel_index:int, source_strength:float=1.) -> np.ndarray:
        """
        Constructs the source term vector for the given source voxel.

        Parameters:
        - source_voxel_index (int): Index of the source voxel.
        - source_strength (float, optional): Strength of the source, default is 1.

        Returns:
        - np.ndarray: Source term vector.
        """
        s = np.zeros(self.num_voxels)
        s[source_voxel_index] = source_strength
        return s
    
    def compute_optical_thickness(self, x1:float, x2:float, start_region_idx:int, end_region_idx:int) -> float:
        """
        Computes the optical thickness between two positions.

        Parameters:
        - x1 (float): Starting x-coordinate.
        - x2 (float): Ending x-coordinate.
        - start_region_idx (int): Region index of start position.
        - end_region_idx (int): Region index of end position.

        Returns:
        - float: Optical thickness between x1 and x2.
        """
        # Since transport is symmetric, it does not matter which direction is traveled. Let's set it to travel left to right.
        if x1 > x2:
            x1, x2 = x2, x1
        
        right_of_x1 = self.boundaries[start_region_idx + 1] - x1
        left_of_x2 = x2 - self.boundaries[end_region_idx]
        
        widths = self.widths[start_region_idx:end_region_idx + 1]
        widths[0] = right_of_x1
        widths[-1] = left_of_x2
        
        sigmas = self.sigmas[start_region_idx:end_region_idx + 1]
        tau = np.dot(widths, sigmas)
        return tau
    
    def compute_streaming_operator(self, start_position:float, end_position:float, start_region_idx:int, end_region_idx:int) -> float:
        """
        Computes the streaming operator between two positions.

        Parameters:
        - start_position (float): Starting position.
        - end_position (float): Ending position.
        - start_region_idx (int): Region index of start position.
        - end_region_idx (int): Region index of end position.

        Returns:
        - float: Streaming operator value.
        """
        if start_position == end_position:
            return 1.0 #TODO need to discuss what this value should be. Perhaps it should not even be computed and simply skipped.
        tau = self.compute_optical_thickness(start_position, end_position, start_region_idx, end_region_idx)
        streaming_operator = (1/2) * sc.exp1(tau) #TODO need to update to include voxel_size in future PR. Concerns about implementation with varying voxel_size.
        return streaming_operator
    
    def construct_transition_matrix(self) -> np.ndarray:
        """
        Constructs the transition matrix for the domain.

        Returns:
        - np.ndarray: Transition matrix.
        """
        K_trans = np.zeros((self.num_voxels, self.num_voxels))
    
        region_idx_i = 0
        region_i = self.regions[region_idx_i]
        
        for i in range(self.num_voxels):
            if i >= self.cumsum_voxels[region_idx_i + 1]:
                region_idx_i += 1
                region_i = self.regions[region_idx_i]
            
            positions_i = region_i.generate_positions(i)
            num_positions_i = len(positions_i)
            
            region_idx_j = region_idx_i
            region_j = self.regions[region_idx_j]
            
            for j in range(i, self.num_voxels):
                if j >= self.cumsum_voxels[region_idx_j + 1]:
                    region_idx_j += 1
                    region_j = self.regions[region_idx_j]
                
                positions_j = region_j.generate_positions(j)
                num_positions_j = len(positions_j)
                
                streaming_sum = 0.0
                for pos_i in positions_i:
                    for pos_j in positions_j:
                        streaming_sum += self.compute_streaming_operator(pos_i, pos_j, region_idx_i, region_idx_j)
                
                avg_streaming_operator = streaming_sum / (num_positions_i * num_positions_j)
                K_trans[i, j] = avg_streaming_operator
                K_trans[j, i] = K_trans[i, j]

        return K_trans

    def compute_neutron_flux(self, source_voxel_index:int, source_strength:float = 1.) -> np.ndarray:
        """
        Computes the neutron flux for the domain given a source.

        Parameters:
        - source_voxel_index (int): Index of the source voxel.
        - source_strength (float, optional): Strength of the source, default is 1.

        Returns:
        - np.ndarray: Neutron flux vector.
        """
        s = self.construct_source_term(source_voxel_index, source_strength)
        logging.info(f"Constructed source term: {s}")
        
        K_trans = self.construct_transition_matrix()
        logging.info(f"Constructed transition matrix: {K_trans}")
        
        sigma_s = np.zeros(self.num_voxels)
        for region_idx, region in enumerate(self.regions):
            first_voxel_idx = self.cumsum_voxels[region_idx]
            last_voxel_idx = self.cumsum_voxels[region_idx + 1] - 1
            sigma_s[first_voxel_idx:last_voxel_idx] = region.scattering_cross_section
        sigma_s_matrix = np.diag(sigma_s)
        logging.info(f"Constructed sigma_s matrix: {sigma_s_matrix}")
        
        try:
            inverse_matrix = np.linalg.inv(np.eye(len(K_trans)) - np.dot(K_trans, sigma_s_matrix))
            phi = np.dot(inverse_matrix, np.dot(K_trans, s))
            logging.info(f"Computed neutron flux: {phi}")
            return phi
        except np.linalg.LinAlgError as e:
            logging.error(f"Linear algebra error during neutron flux calculation: {e}")
            raise e

def read_input(file_path:str) -> dict:
    """
    Reads input data from a specified JSON file.

    Parameters:
    - file_path (str): The path to the JSON file containing input data.

    Returns:
    - dict: A dictionary containing the input parameters loaded from the JSON file.
    """
    logging.debug(f"Reading input from {file_path}")
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        logging.info(f"Input data read successfully: {data}")
    except FileNotFoundError as e:
        logging.error(f"File {file_path} not found: {e}")
        raise e
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        raise e
    return data

def perform_calculation(start_x:float, regions_data:list, source_data:dict) -> np.ndarray:
    """
    Performs the main calculation for the neutron flux based on input parameters.

    Parameters:
    - start_x (float): Starting x-value for the calculation.
    - regions_data (list): List of dictionaries containing region parameters consisting of name, width, num_voxels, positions_per_voxel, position_location, and scattering_cross_section.
    - source_data (dict): Dictionary containing source parameters consisting of voxel_index and strength. 

    Returns:
    - numpy.ndarray: The neutron flux vector calculated from the input data.
    """
    regions = []
    for region_data in regions_data:
        region = Region(
            width=region_data['width'],
            num_voxels=region_data['num_voxels'],
            scattering_cross_section=region_data['scattering_cross_section'],
            positions_per_voxel=region_data['positions_per_voxel'],
            position_location_str=region_data['position_location']
        )
        regions.append(region)
    
    domain = Domain(start_x, regions)
    source_voxel_index = source_data['voxel_index']
    source_strength = source_data['strength']
    
    phi = domain.compute_neutron_flux(source_voxel_index, source_strength)
    return phi

def write_output(data:np.ndarray, file_path:str="results.csv"):
    """
    Writes the calculation results to a CSV file.

    Parameters:
    - data (numpy.ndarray): The neutron flux vector to be written to the file.
    - file_path (str, optional): The path to the CSV file where results will be saved, default is "results.csv".

    Returns:
    - None
    """
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Voxel Index", "Neutron Flux"])
            for i, flux in enumerate(data):
                writer.writerow([i, flux])
        logging.info(f"Data written to CSV file {file_path}")
    except Exception as e:
        logging.error(f"Error writing to CSV file: {e}")
        raise e

def setup_logging(level:int = logging.INFO):
    """
    Configures logging settings.

    Parameters:
    - level (int, optional): The logging level to set (e.g., logging.DEBUG, logging.INFO).

    Returns:
    - None
    """
    logging.basicConfig(
        filename='drt_1d.log',
        filemode='w',
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
    - argparse.Namespace: A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run deterministic radiation transport calculations in a 1-D voxelized geometry.')
    parser.add_argument(
        '--input_file',
        type=str,
        default='input_data.json',
        help='Path to the JSON file containing input data. Defaults to "input_data.json".'
    )
    return parser.parse_args()

def main():
    setup_logging()
    
    args = parse_arguments()
    input_file_path = args.input_file
    
    input_data = read_input(input_file_path)
    
    phi = perform_calculation(input_data["start_x"], input_data["regions"], input_data["source"])
    
    write_output(phi)

if __name__ == "__main__":
    main()