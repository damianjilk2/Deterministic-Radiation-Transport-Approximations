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

def construct_source_term(num_voxels:int, source_voxel_index:int, source_strength:float=1):
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
    
    if not isinstance(num_voxels, int) or num_voxels <= 0:
        raise ValueError("Number of voxels must be a positive integer.")
    if not isinstance(source_voxel_index, int) or source_voxel_index < 0 or source_voxel_index >= num_voxels:
        raise IndexError("Source voxel index must be within the range of voxels.")
    
    s = np.zeros(num_voxels)
    s[source_voxel_index] = source_strength
    logging.debug(f"Constructed source vector: {s}")
    return s

def calculate_K_trans(start_index:int, end_index:int, ri:float, rj:float, voxel_size:float, sigma_s:np.ndarray):
    """
    Calculate the streaming operator. 
    
    Used by the construct_transition_matrix function to build the K_trans matrix.
    
    The calculation assumes equal voxel spacing.
    
    Based on principles from the book by Lewis and Miller (1984). Equations derived using Chapter 5. 

    Parameters:
    - start_index (int): Starting position voxel index
    - end_index (int): Ending position voxel index
    - ri (float): Starting point within a voxelized space.
    - rj (float): Ending point within a voxelized space.
    - voxel_size (float): Size of each voxel. Assume constant.
    - sigma_s (numpy.ndarray): Vector of scattering cross-sections.

    Returns:
    - float: Streaming operator. This represents the probability that a particle born at ri streams to rj without interacting.
    
    Example Diagrams:
    
      -->  Streaming to the right -->
    ----------------------------------------------------
    | Start Voxel  | In-between voxels |  End Voxel  | 
    ----------------------------------------------------
    |---[]---------|-------------------|---------[]--|
    |   ri         |                   |         rj  |
        |__________|                   |__________|
         left distance                  right distance
         
                  
      <--  Streaming to the left <--
    ----------------------------------------------------
    | End Voxel    | In-between voxels | Start Voxel | 
    ----------------------------------------------------
    |---[]---------|-------------------|---------[]--|
    |   rj         |                   |         ri  |
        |__________|                   |__________|
         left distance                  right distance
    """
    logging.debug(f"Calculating K_trans from voxel {ri} to voxel {rj} with voxel size {voxel_size}")
    
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive and non-zero.")
    if ri < 0 or ri > len(sigma_s) * voxel_size or rj < 0 or rj > len(sigma_s) * voxel_size:
        raise ValueError("Positions must be within the spatial domain defined by the voxels.")
    
    distance = abs(ri - rj)
    if distance == 0:
        raise ValueError(f"The positions of (ri,start_index):({ri},{start_index}) and (rj,end_index):({rj},{end_index}) are the same value and will result in an infinite streaming operator.")
    tau = 0
    
    if start_index == end_index:
        # Same voxel calculation
        tau = sigma_s[start_index] * distance
        logging.debug("Intra-voxel tau: %f", tau)
    else:
        # Multiple voxel calculation
        if ri < rj:
            # Direction is to the right (start_index and ri are on the left)
            
            # Calculate left fraction
            if ri == start_index*voxel_size:
                # ri is on the left-most point of the voxel
                left_distance = voxel_size
            elif start_index*voxel_size < ri and ri < (start_index+1)*voxel_size:
                # ri is between the voxel edges
                left_distance = (start_index+1)*voxel_size - ri
            elif ri == (start_index+1)*voxel_size:
                # ri is on the right-most point of the voxel
                left_distance = 0
            else:
                # ri is not within or on the edges of this voxel
                raise ValueError(f"Invalid position (ri: {ri}) and voxel index (start_index: {start_index}) combination.")
            left_tau = sigma_s[start_index] * left_distance
            
            # Calculate right fraction
            if rj == end_index*voxel_size:
                # rj is on the left-most point of the voxel
                right_distance = 0
            elif end_index*voxel_size < rj and rj < (end_index+1)*voxel_size:
                # rj is beteween the voxel edges
                right_distance = rj - (end_index*voxel_size)
            elif rj == (end_index+1)*voxel_size:
                # rj is on the right-most point of the voxel
                right_distance = voxel_size
            else:
                # rj is not within or on the edges of this voxel
                raise ValueError(f"Invalid position (rj: {rj}) and voxel index (end_index: {end_index}) combination.")
            right_tau = sigma_s[end_index] * right_distance
            
            # Sum full voxels
            center_tau = sum(sigma_s[start_index + 1:end_index]) * voxel_size
            
            logging.debug("Direction is to the right. Left_tau: %f, Right_tau: %f, Center_tau: %f", left_tau, right_tau, center_tau)
            tau = left_tau + right_tau + center_tau
                
        elif ri > rj:
            # Direction is to the left (start_index and ri are on the right)
            
            # Calculate left fraction
            if rj == end_index*voxel_size:
                # rj is on the left-most point of the voxel
                left_distance = voxel_size
            elif end_index*voxel_size < rj and rj < (end_index+1)*voxel_size:
                # rj is between the voxel edges
                left_distance = (end_index+1)*voxel_size - rj
            elif rj == (end_index+1)*voxel_size:
                # rj is on the right-most point of the voxel
                left_distance = 0
            else:
                # rj is not within or on the edges of this voxel
                raise ValueError(f"Invalid position (rj: {rj}) and voxel index (end_index: {end_index}) combination.")
            left_tau = sigma_s[end_index] * left_distance
            
            # Calculate right fraction
            if ri == start_index*voxel_size:
                # ri is on the left-most point of the voxel
                right_distance = 0
            elif start_index*voxel_size < ri and ri < (start_index+1)*voxel_size:
                # ri is beteween the voxel edges
                right_distance = ri - (start_index*voxel_size)
            elif ri == (start_index+1)*voxel_size:
                # ri is on the right-most point of the voxel
                right_distance = voxel_size
            else:
                # ri is not within or on the edges of this voxel
                raise ValueError(f"Invalid position (ri: {ri}) and voxel index (start_index: {start_index}) combination.")
            right_tau = sigma_s[start_index] * right_distance
            
            # Sum full voxels
            center_tau = sum(sigma_s[end_index + 1:start_index]) * voxel_size
            
            logging.debug("Direction is to the left. Left_tau: %f, Right_tau: %f, Center_tau: %f", left_tau, right_tau, center_tau)
            tau = left_tau + right_tau + center_tau
    
    #streaming_operator = np.exp(-tau) / (4 * np.pi * distance**2) NOTE: this is the old streaming operator calculation
    streaming_operator = (1/2) * sc.exp1(tau)
    logging.debug(f"Calculated K_trans: {streaming_operator}")
    return streaming_operator

def construct_transition_matrix(num_voxels:int, voxel_size:float, sigma_s:np.ndarray, positions:dict):
    """
    Construct the transition matrix.

    Parameters:
    - num_voxels (int): Number of spatial elements (voxels).
    - voxel_size (float): Size of each voxel.
    - sigma_s (numpy.ndarray): Vector of scattering cross-sections.
    - positions (dict): Dictionary of positions corresponding to each voxel index.

    Returns:
    - numpy.ndarray: Transition matrix.
    """
    logging.debug(f"Constructing transition matrix with {num_voxels} voxels and voxel size {voxel_size}")
  
    if num_voxels <= 0:
        raise ValueError("Number of voxels must be positive and non-zero.")
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive and non-zero.")
        
    K_trans = np.zeros((num_voxels, num_voxels))
    for i in range(num_voxels):
        for j in range(num_voxels):
            if i == j:
                # Diagonal element (same voxel)
                # Retrieve the list of positions for the current voxel i (same as j in this case)
                positions_in_voxel = positions.get(str(i))
                
                # Check if there are multiple positions within this voxel
                if len(positions_in_voxel) > 1:
                    # Calculate the total number of position pairs (combinations of 2)
                    position_pairs = itertools.combinations(positions_in_voxel, 2)
                    
                    streaming_operator_sum = 0.0
                    num_position_pairs = 0
                    for position_a, position_b in position_pairs:
                        # Calculate the streaming operator for the pair (both directions)
                        streaming_operator_sum += calculate_K_trans(i, j, position_a, position_b, voxel_size, sigma_s)
                        streaming_operator_sum += calculate_K_trans(i, j, position_b, position_a, voxel_size, sigma_s)
                        num_position_pairs += 2
                    
                    # Compute average streaming operator for this voxel and assign it to the transition matrix
                    K_trans[i, j] = streaming_operator_sum / num_position_pairs
                    
                    logging.debug("Voxel %d has %d positions, average streaming operator: %f", i, len(positions_in_voxel), K_trans[i, j])
                else:
                    # If there is only one position, set diagonal element to 1
                    K_trans[i, j] = 1
                    logging.debug("Voxel %d has a single position, set K_trans[%d, %d] to 1", i, i, j)
            else:
                # Off-diagonal element (different voxels)
                positions_in_voxel_i = positions.get(str(i))
                positions_in_voxel_j = positions.get(str(j))
                
                streaming_operator_sum = 0.0
                num_position_pairs = 0
                
                # Iterate over all combinations of positions between voxel i and voxel j
                for position_i in positions_in_voxel_i:
                    for position_j in positions_in_voxel_j:
                        # Calculate the streaming operator for the pair
                        streaming_operator_sum += calculate_K_trans(i, j, position_i, position_j, voxel_size, sigma_s)
                        num_position_pairs += 1
                
                # Compute average streaming operator between voxels and assign to the transition matrix
                K_trans[i, j] = streaming_operator_sum / num_position_pairs
                
                logging.debug("Voxel pair (%d, %d) has %d position pairs, average streaming operator: %f",i, j, num_position_pairs, K_trans[i, j])
    logging.debug("Constructed transition matrix: %s", K_trans)
    return K_trans

def calculate_neutron_flux(K_trans:np.ndarray, sigma_s:np.ndarray, s:np.ndarray):
    """
    Calculate the neutron flux vector.

    Parameters:
    - K_trans (numpy.ndarray): Transition matrix.
    - sigma_s (numpy.ndarray): Vector of scattering cross-sections.
    - s (numpy.ndarray): Source term vector.

    Returns:
    - numpy.ndarray: Neutron flux vector.
    """
    logging.debug("Calculating neutron flux vector")
    sigma_s_matrix = np.diag(sigma_s) # Converts sigma_s vector to a diagonal matrix
    phi = np.dot(np.linalg.inv(np.eye(len(K_trans)) - np.dot(K_trans, sigma_s_matrix)), np.dot(K_trans, s))
    logging.debug(f"Calculated neutron flux vector: {phi}")
    return phi

def read_input(file_path:str):
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
        logging.debug(f"Input data: {data}")
    except FileNotFoundError as e:
        logging.error(f"File {file_path} not found: {e}")
        raise e
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        raise e
    return data

def perform_calculation(input_data:dict):
    """
    Performs the main calculation for the neutron flux based on input parameters.

    Parameters:
    - input_data (dict): A dictionary containing input parameters including spatial boundaries, voxel count, source index, and scattering cross-sections.

    Returns:
    - numpy.ndarray: The neutron flux vector calculated from the input data.
    """
    logging.debug(f"Performing calculation with input data: {input_data}")
    x_min = input_data.get("minimum_x")
    x_max = input_data.get("maximum_x")
    num_voxels = input_data.get("number_of_voxels")
    source_voxel_index = input_data.get("source_voxel_index")
    sigma_s_values = input_data.get("scattering_cross_section")
    positions = input_data.get("positions")
    
    voxel_size = (x_max - x_min) / num_voxels
    
    s = construct_source_term(num_voxels, source_voxel_index)
    K_trans = construct_transition_matrix(num_voxels, voxel_size, sigma_s_values, positions)
    phi = calculate_neutron_flux(K_trans, sigma_s_values, s)
    
    logging.info(f"Scattering Cross-section Vector (sigma_s):\n{sigma_s_values}")
    logging.info(f"Positions:\n{positions}")
    logging.info(f"Source Vector (s):\n{s}")
    logging.info(f"Transition Matrix (K_trans):\n{K_trans}")
    logging.info(f"Neutron Flux Vector (phi):\n{phi}")
    
    return phi

def write_output(data:np.ndarray, file_path:str="results.csv"):
    """
    Writes the calculation results to a CSV file.

    Parameters:
    - data (numpy.ndarray): The neutron flux vector to be written to the file.
    file_path (str, optional): The path to the CSV file where results will be saved, default is "results.csv".

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

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
    - argparse.Namespace: A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Deterministic Radiation Transport Calculation.')
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
    phi = perform_calculation(input_data)
    write_output(phi)

if __name__ == "__main__":
    main()