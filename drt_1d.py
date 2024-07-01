"""
Title: Deterministic Radiation Transport - 1-D Toy Problem with Voxelized Geometry
Author: Damian Jilk
"""

import numpy as np
import json
import csv
import logging
import argparse
import itertools
import scipy.special as sc
import random

class Voxel:
    def __init__(self, index:int, start_position:float, end_position:float, scattering_cross_section:float, positions:list):
        """
        Represents a voxel in the 1-D space.

        Parameters:
        - index (int): Index of the voxel.
        - start_position (float): Starting position of the voxel.
        - end_position (float): Ending position of the voxel.
        - scattering_cross_section (float): Scattering cross-section of the voxel.
        """
        self.index = index
        self.start_position = start_position
        self.end_position = end_position
        self.scattering_cross_section = scattering_cross_section
        self.positions = positions
        logging.debug(
            f"Voxel created with index {self.index}, "
            f"start_position {self.start_position}, "
            f"end_position {self.end_position}, "
            f"scattering_cross_section {self.scattering_cross_section}, "
            f"positions {self.positions}"
        )
        
    @property
    def voxel_size(self) -> float:
        """Calculate and return the size of the voxel."""
        return self.end_position - self.start_position
    
    @property
    def number_of_positions(self) -> int:
        """Calculate and return the number of positions."""
        return len(self.positions)

def construct_source_term(num_voxels:int, source_voxel_index:int, source_strength:float=1.) -> np.ndarray:
    """
    Constructs the source term vector. 
    
    Assumes a point source. Minor modifications can be made to allow for a distribution.

    Parameters:
    - num_voxels (int): Number of spatial elements (voxels).
    - source_voxel_index (int): Index of the voxel where the source is located.
    - source_strength (float, optional): Strength of the source with default value of 1.

    Returns:
    - numpy.ndarray: Source term vector.
    """
    logging.debug(f"Constructing source term with {num_voxels} voxels, source at index {source_voxel_index}, strength {source_strength}")
    
    s = np.zeros(num_voxels)
    s[source_voxel_index] = source_strength
    logging.debug(f"Constructed source vector: {s}")
    return s

def calculate_K_trans(start_position:float, end_position:float, voxels:list[Voxel]) -> float:
    """
    Calculate the streaming operator. 
    
    Used by the construct_transition_matrix function to build the K_trans matrix.
    
    Based on principles from the book by Lewis and Miller (1984). Equations derived using Chapter 5. 

    Parameters:
    - start_position (float): Starting point within a voxelized space.
    - end_position (float): Ending point within a voxelized space.
    - voxels (list[Voxel]): List of Voxel objects between the start and end positions.

    Returns:
    - float: Streaming operator. TODO: better understand what this value represents.
    """
    if end_position - start_position == 0:
        return 1.0  #TODO revisit this assumption after better understanding the streaming operator.
    
    logging.debug(f"Calculating K_trans from {start_position} to {end_position}")
    
    # Since transport is symmetric, it does not matter which direction is traveled. Let's set it to travel left to right.
    if start_position > end_position:
        start_position, end_position = end_position, start_position
    
    start_voxel = voxels[0]
    end_voxel = voxels[-1]
    
    # TODO: assume equal voxel size for now. Future implementations will have variable voxel sizes.
    voxel_size = start_voxel.voxel_size
    sigma_s = [voxel.scattering_cross_section for voxel in voxels]
    
    tau = 0.0
    
    if start_voxel.index == end_voxel.index:
        distance = end_position - start_position
        assert len(sigma_s) == 1
        tau = start_voxel.scattering_cross_section * distance
        logging.debug(f"Same voxel: tau = {tau}")
    else:
        left_distance = (start_voxel.index + 1) * voxel_size - start_position
        right_distance = end_position - (end_voxel.index*voxel_size)
        
        path = [left_distance] + [voxel_size] * (end_voxel.index - start_voxel.index - 1) + [right_distance]
        tau = np.dot(sigma_s, path)
        logging.debug(f"Different voxels: tau = {tau}")
    
    streaming_operator = (1/2) * sc.exp1(tau)
    logging.debug(f"Calculated streaming operator: {streaming_operator}")
    return streaming_operator

def construct_transition_matrix(voxels:list[Voxel]) -> np.ndarray:
    """
    Construct the transition matrix.

    Parameters:
    - voxels (list[Voxel]): List of Voxel objects.

    Returns:
    - numpy.ndarray: Transition matrix.
    """
    logging.debug("Constructing transition matrix")
    
    num_voxels = len(voxels)
    K_trans = np.zeros((num_voxels, num_voxels))
    
    for i, voxel_i in enumerate(voxels):
        if voxel_i.number_of_positions == 1:
            K_trans[i,i] = 1                    # TODO need to revisit this assumption
            logging.debug(f"Voxel {i} has a single position; K_trans[{i},{i}] set to 1")
        else:
            position_pairs = list(itertools.combinations(voxel_i.positions, 2))
            num_position_pairs = 2 * len(position_pairs)
            
            streaming_sum = sum(2 * calculate_K_trans(pos_a, pos_b, [voxel_i]) for pos_a, pos_b in position_pairs)
            K_trans[i,i] = streaming_sum / num_position_pairs
            logging.debug(f"K_trans[{i},{i}] calculated as {K_trans[i, i]}")

        for j in range(i):
            voxel_j = voxels[j]
            num_position_pairs = voxel_i.number_of_positions * voxel_j.number_of_positions
            
            # NOTE: j is always smaller than i. Using indexing, sublist of voxels to consider is voxels[j:i+1].
            streaming_sum = sum(calculate_K_trans(pos_i, pos_j, voxels[j:i+1])
                                for pos_i in voxel_i.positions
                                for pos_j in voxel_j.positions)
            
            K_trans[i,j] = streaming_sum  / num_position_pairs
            K_trans[j,i] = K_trans[i,j]
            logging.debug(f"K_trans[{i},{j}] and K_trans[{j},{i}] calculated as {K_trans[i, j]}")

    logging.info(f"Constructed transition matrix:\n{K_trans}")
    return K_trans

def calculate_neutron_flux(K_trans:np.ndarray, sigma_s:list, s:np.ndarray) -> np.ndarray:
    """
    Calculate the neutron flux vector.

    Parameters:
    - K_trans (numpy.ndarray): Transition matrix.
    - sigma_s (list): List of scattering cross-sections.
    - s (numpy.ndarray): Source term vector.

    Returns:
    - numpy.ndarray: Neutron flux vector.
    """
    logging.debug("Calculating neutron flux")
    
    sigma_s_matrix = np.diag(sigma_s)
    
    try:
        inverse_matrix = np.linalg.inv(np.eye(len(K_trans)) - np.dot(K_trans, sigma_s_matrix))
        phi = np.dot(inverse_matrix, np.dot(K_trans, s))
        logging.info(f"Calculated neutron flux: {phi}")
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
    logging.debug(f"Performing calculation with input data: start_x={start_x}, regions_data={regions_data}, source_data={source_data}")
    
    voxels:list[Voxel] = []
    total_num_voxels = 0
    
    # Read through the regions_data and create Voxel objects accordingly. 
    for index, region_data in enumerate(regions_data):
        width = region_data['width']
        num_voxels = region_data['num_voxels']
        positions_per_voxel = region_data['positions_per_voxel']
        position_location = region_data['position_location']
        scattering_cross_section = region_data['scattering_cross_section']

        total_num_voxels += num_voxels
        
        voxel_width = width / num_voxels
        for voxel_index in range(num_voxels):
            start_position = start_x + index*width + voxel_index*voxel_width
            end_position = start_position + voxel_width
            positions = []

            if position_location == 'evenly-spaced':
                # Evenly spaced between the two end points, excluding the end points themselves.
                positions = np.linspace(start_position, end_position, positions_per_voxel + 2)[1:-1].tolist()
            elif position_location == 'random':
                positions = sorted([random.uniform(start_position, end_position) for _ in range(positions_per_voxel)])

            voxel = Voxel(
                index = len(voxels),
                start_position = start_position,
                end_position = end_position,
                scattering_cross_section = scattering_cross_section,
                positions = positions
            )
            voxels.append(voxel)
    
    source_voxel_index = source_data['voxel_index']
    source_strength = source_data['strength']
    
    s = construct_source_term(total_num_voxels, source_voxel_index, source_strength)
    K_trans = construct_transition_matrix(voxels)
    
    sigma_s_values = [voxel.scattering_cross_section for voxel in voxels]
    
    phi = calculate_neutron_flux(K_trans, sigma_s_values, s)
    
    return phi

def validate_input(input_data:dict) -> None:
    """
    Validates the overall input data structure and key parameters.
    
    Parameters:
    - input_data (dict): The input data to be validated.

    Returns:
    - None

    Raises:
    - ValueError: If any validation checks fail.
    """
    logging.debug("Validating input data")
    
    # Validate start_x
    if not isinstance(input_data['start_x'], (int, float)):
        raise ValueError("start_x must be a number.")
    
    # Validate regions
    regions = input_data['regions']
    if not regions or not isinstance(regions, list):
        raise ValueError("regions must be a non-empty list.")
    
    for region in regions:
        # Validate each region
        if not isinstance(region, dict):
            raise ValueError("Each region must be a dictionary.")
        
        required_keys = ['name', 'width', 'num_voxels', 'positions_per_voxel',
                         'position_location', 'scattering_cross_section']
        for key in required_keys:
            if key not in region:
                raise ValueError(f"Region is missing required key: {key}")
        
        if not isinstance(region['name'], str):
            raise ValueError("Region name must be a string.")
        
        if not isinstance(region['width'], (int, float)) or region['width'] <= 0:
            raise ValueError("Region width must be a positive number.")
        
        if not isinstance(region['num_voxels'], int) or region['num_voxels'] <= 0:
            raise ValueError("Number of voxels must be a positive integer.")
        
        if not isinstance(region['positions_per_voxel'], int) or region['positions_per_voxel'] <= 0:
            raise ValueError("Positions per voxel must be a positive integer.")
        
        if region['position_location'] not in ['evenly-spaced', 'random']:
            raise ValueError("Position location must be 'evenly-spaced' or 'random'.")
        
        if not isinstance(region['scattering_cross_section'], (int, float)) or region['scattering_cross_section'] <= 0:
            raise ValueError("Scattering cross-section must be a positive number.")
    
    # Validate source
    source = input_data['source']
    if not source or not isinstance(source, dict):
        raise ValueError("source must be a dictionary.")
    
    required_keys = ['voxel_index', 'strength']
    for key in required_keys:
        if key not in source:
            raise ValueError(f"Source is missing required key: {key}")
    
    if not isinstance(source['voxel_index'], int) or source['voxel_index'] < 0:
        raise ValueError("Source voxel index must be a non-negative integer.")
    
    if not isinstance(source['strength'], (int, float)) or source['strength'] <= 0:
        raise ValueError("Source strength must be a positive number.")
    
    logging.info("Input data validation passed")

def write_output(data:np.ndarray, file_path:str="results.csv"):
    """
    Writes the calculation results to a CSV file.

    Parameters:
    - data (numpy.ndarray): The neutron flux vector to be written to the file.
    - file_path (str, optional): The path to the CSV file where results will be saved, default is "results.csv".

    Returns:
    - None
    """
    logging.debug(f"Writing data to CSV file {file_path}")
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
    setup_logging(logging.DEBUG)
    logging.debug("Main function started")
    
    args = parse_arguments()
    input_file_path = args.input_file
    
    input_data = read_input(input_file_path)
    validate_input(input_data)
    
    phi = perform_calculation(input_data["start_x"], input_data["regions"], input_data["source"])
    
    write_output(phi)

if __name__ == "__main__":
    main()