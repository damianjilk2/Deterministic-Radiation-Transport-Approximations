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

class Voxel:
    def __init__(self, index:int, start_position:float, end_position:float, scattering_cross_section:float, positions:list):
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
        return self.end_position - self.start_position
    
    @property
    def number_of_positions(self) -> int:
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
    if not isinstance(num_voxels, int) or num_voxels <= 0:
        error_msg = f"Number of voxels must be a positive integer. Received: {num_voxels}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    if not isinstance(source_voxel_index, int) or source_voxel_index < 0 or source_voxel_index >= num_voxels:
        error_msg = f"Source voxel index must be within the range of voxels (0 to {num_voxels-1}). Received: {source_voxel_index}"
        logging.error(error_msg)
        raise IndexError(error_msg)
    if not isinstance(source_strength, (int, float)) or source_strength < 0:
        error_msg = f"Source strength must be a non-negative number. Received: {source_strength}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    logging.debug(f"Constructing source term with {num_voxels} voxels, source at index {source_voxel_index}, strength {source_strength}")
    
    s = np.zeros(num_voxels)
    s[source_voxel_index] = source_strength
    logging.debug(f"Constructed source vector: {s}")
    return s

def calculate_K_trans(start_position:float, end_position:float, voxels:list[Voxel]) -> float:
    """
    Calculate the streaming operator. 
    
    Used by the construct_transition_matrix function to build the K_trans matrix.
    
    The calculation currently assumes equal voxel spacing but will be refactored soon.
    
    Based on principles from the book by Lewis and Miller (1984). Equations derived using Chapter 5. 

    Parameters:
    - start_position (float): Starting point within a voxelized space.
    - end_position (float): Ending point within a voxelized space.
    - voxels (list[Voxel]): List of Voxel objects between the start and end positions.

    Returns:
    - float: Streaming operator. TODO: better understand what this value represents.
    """
    if not isinstance(start_position, (int, float)) or not isinstance(end_position, (int, float)):
        error_msg = f"Start and end positions must be numbers. Received: start={start_position}, end={end_position}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    if not voxels:
        error_msg = "Voxels list cannot be empty."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    if end_position - start_position == 0:
        return 1.0  #TODO revisit this assumption after better understanding the streaming operator.
    
    logging.debug(f"Calculating K_trans from {start_position} to {end_position}")
    
    # Since transport is symmetric, it does not matter which direction is traveled. Let's set it to travel left to right.
    if start_position > end_position:
        start_position, end_position = end_position, start_position
    
    start_voxel = voxels[0]
    end_voxel = voxels[-1]
    
    # NOTE: assume equal voxel size for now. Future implementations will have variable voxel sizes.
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
    # NOTE: Assume sigma_s is still voxel based (will change with continuous geometry)
    
    if not voxels:
        error_msg = "Voxels list cannot be empty."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    logging.debug("Constructing transition matrix")
    
    num_voxels = len(voxels)
    K_trans = np.zeros((num_voxels, num_voxels))
    
    for i, voxel_i in enumerate(voxels):
        if voxel_i.number_of_positions == 1:
            K_trans[i,i] = 1        # TODO need to revisit this assumption
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
            
            # NOTE: j is always smaller than i. Using indexing, sublist of voxels to consider is voxels[j:i+1]
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
    if K_trans.shape[0] != K_trans.shape[1]:
        error_msg = f"K_trans must be a square matrix. Received: {K_trans.shape}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    if len(sigma_s) != K_trans.shape[0]:
        error_msg = "Length of sigma_s must match the dimension of K_trans."
        logging.error(error_msg)
        raise ValueError(error_msg)
    if s.shape[0] != K_trans.shape[0]:
        error_msg = "Length of source vector s must match the dimension of K_trans."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
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

def perform_calculation(input_data:dict) -> np.ndarray:
    """
    Performs the main calculation for the neutron flux based on input parameters.

    Parameters:
    - input_data (dict): A dictionary containing input parameters including spatial boundaries, voxel count, source index, and scattering cross-sections.

    Returns:
    - numpy.ndarray: The neutron flux vector calculated from the input data.
    """
    logging.debug(f"Performing calculation with input data: {input_data}")
    
    voxels_data = input_data['voxels']
    if not isinstance(voxels_data, list) or len(voxels_data) == 0:
        error_msg = "Voxels data must be a non-empty list."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    voxels:list[Voxel] = []
    for voxel_data in voxels_data:
        voxel = Voxel(
            index=voxel_data["index"],
            start_position=voxel_data["start_position"],
            end_position=voxel_data["end_position"],
            scattering_cross_section=voxel_data["scattering_cross_section"],
            positions=voxel_data["positions"]
        )
        voxels.append(voxel)
    if not voxels:
        error_msg = "No voxels found in input data."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    source_voxel_index = input_data['source_voxel_index']
    if not(0 <= source_voxel_index < len(voxels)):
        error_msg = "Source voxel index out of bounds."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    sigma_s_values = [voxel.scattering_cross_section for voxel in voxels]
    num_voxels = len(voxels)
    
    s = construct_source_term(num_voxels, source_voxel_index)
    K_trans = construct_transition_matrix(voxels)
    phi = calculate_neutron_flux(K_trans, sigma_s_values, s)
    
    return phi

def validate_input(input_data:dict) -> None:
    """
    Validates the overall input data structure and key parameters.
    
    NOTE: possible implementation could look at start and end positions of the voxels and ensure the user did create gaps.

    Parameters:
    - input_data (dict): The input data to be validated.

    Returns:
    - None

    Raises:
    - ValueError: If any validation checks fail.
    """
    logging.debug("Validating input data")
    try:
        if 'voxels' not in input_data or not isinstance(input_data['voxels'], list):
            raise ValueError("Input must contain a list of voxels.")

        # Sort voxels by start_position if they are out of order.
        voxels = sorted(input_data['voxels'], key=lambda x: x['start_position'])
        
        # Used to track indices of given voxels list
        index_set = set()
        for voxel in voxels:
            if not all(k in voxel for k in ("index", "start_position", "end_position", "scattering_cross_section", "positions")):
                raise ValueError("Each voxel must contain index, start_position, end_position, scattering_cross_section, and positions.")
            
            if not isinstance(voxel['index'], int) or voxel['index'] < 0:
                raise ValueError(f"Voxel index must be a non-negative integer. Found: {voxel['index']}")
            
            if voxel['index'] in index_set:
                raise ValueError(f"Duplicate voxel index found: {voxel['index']}")
            
            index_set.add(voxel['index'])
            
            if not (isinstance(voxel['start_position'], (int, float)) and isinstance(voxel['end_position'], (int, float)) and voxel['start_position'] < voxel['end_position']):
                raise ValueError(f"Voxel start_position and end_position must be numbers with start_position < end_position. Found: start={voxel['start_position']}, end={voxel['end_position']}")
            
            if not isinstance(voxel['scattering_cross_section'], (int, float)) or voxel['scattering_cross_section'] < 0:
                raise ValueError(f"Scattering cross-section must be a non-negative number. Found: {voxel['scattering_cross_section']}")
            
            if not isinstance(voxel['positions'], list) or not all(isinstance(pos, (int, float)) for pos in voxel['positions']):
                raise ValueError("Voxel positions must be a list of numbers.")
            
        expected_indices = set(range(len(voxels)))
        if not index_set == expected_indices:
            raise ValueError(f"Voxel indices are not as expected. Expected: {list(expected_indices)}, Found: {list(index_set)}")
        
        
        if 'source_voxel_index' not in input_data or not isinstance(input_data['source_voxel_index'], int):
            raise ValueError("Input must contain a source_voxel_index of type int.")
        
        source_voxel_index = input_data['source_voxel_index']
        if not(0 <= source_voxel_index < len(voxels)):
            raise ValueError(f"Source voxel index {source_voxel_index} is out of bounds for the given number of voxels ({len(voxels)}).")
        
        logging.info("Input data validation passed")
    except ValueError as e:
        logging.error(f"Input validation failed: {e}")
        raise e

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
    setup_logging(logging.INFO)
    logging.debug("Main function started")
    args = parse_arguments()
    input_file_path = args.input_file
    input_data = read_input(input_file_path)
    validate_input(input_data)
    phi = perform_calculation(input_data)
    write_output(phi)

if __name__ == "__main__":
    main()