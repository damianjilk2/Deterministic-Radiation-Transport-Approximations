"""
Title: Deterministic Radiation Trasport - 1-D Toy Problem with Voxelized Geometry
Author: Damian Jilk
"""
# TODO: Add tests/validation to ensure tau is being calculated correctly
# TODO: Implement default position calculation for voxels set to center
# TODO: Handle scenarios where the user inputs multiple values for one voxel
#       - Average the tau values to find streaming operator and then flux
# TODO: Implement input validation and edge handling

import numpy as np
import json
import csv
import logging

def construct_source_term(num_voxels:int, source_voxel_index:int, source_strength:float=1):
    """
    Constructs the source term vector. Assumes a point source. Minor modifications can be made to allow for a distribution.

    Parameters:
    - num_voxels (int): Number of spatial elements (voxels).
    - source_voxel_index (int): Index of the voxel where the source is located.
    - source_strength (float, optional): Strength of the source with default value of 1.

    Returns:
    - numpy.ndarray: Source term vector.
    """
    logging.debug("Constructing source term with %d voxels, source at index %d, strength %f", num_voxels, source_voxel_index, source_strength)
    
    if num_voxels <= 0:
        raise ValueError("Number of voxels must be a positive integer.")
    if source_voxel_index < 0 or source_voxel_index >= num_voxels:
        raise IndexError("Source voxel index must be within the range of voxels.")
    
    s = np.zeros(num_voxels)
    s[source_voxel_index] = source_strength
    logging.debug("Constructed source vector: %s", s)
    return s

def calculate_K_trans(start_index:int, end_index:int, ri:float, rj:float, voxel_size:float, sigma_s:np.ndarray):
    """
    Calculate the streaming operator. Used by the construct_transition_matrix function to build the K_trans matrix.

    Parameters:
    - start_index (int): Starting positions list index and thus voxel index
    - end_index (int): Ending positions list index and thus voxel index
    - ri (float): Starting point within a voxelized space.
    - rj (float): Ending point within a voxelized space.
    - voxel_size (float): Size of each voxel. Assume constant.
    - sigma_s (numpy.ndarray): Vector of scattering cross-sections.

    Returns:
    - float: Streaming operator. This represents the probability that a particle born at ri streams to rj without interacting.
    """
    logging.debug("Calculating K_trans from index %d to index %d, position %.3f to position %.3f, with voxel size %f", start_index, end_index, ri, rj, voxel_size)
    
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive and non-zero.")
    if ri < 0 or ri > len(sigma_s) * voxel_size or rj < 0 or rj > len(sigma_s) * voxel_size:
        raise ValueError("Positions must be within the spatial domain defined by the voxels.")
    
    distance = abs(ri - rj)
    tau = 0
    
    if start_index == end_index:
        # Same voxel calculation
        # NOTE: currently not being used. Will be utilized in future commit
        tau = sigma_s[start_index] * distance
    else:
        # Multiple voxel calculation
        if ri < rj:
            # Direction is to the right (start_index is on the left)
            
            # Calculate left fraction
            left_distance = voxel_size*(start_index+1) - ri
            left_tau = sigma_s[start_index] * left_distance
            
            # Calculate right fraction
            right_distance = rj - voxel_size*(end_index)
            right_tau = sigma_s[end_index] * right_distance
            
            # Sum full voxels
            center_tau = sum(sigma_s[start_index + 1:end_index]) * voxel_size
            
            logging.debug("Direction is to the right. Left_tau: %f, Right_tau: %f, Center_tau: %f", left_tau, right_tau, center_tau)
            tau = left_tau + right_tau + center_tau
                
        else:
            # Direction is to the left (start_index is on the right)
            
            # Calculate right fraction
            right_distance = ri - voxel_size*(start_index)
            right_tau = sigma_s[start_index] * right_distance
            
            # Calculate left fraction
            left_distance = voxel_size*(end_index + 1) - rj
            left_tau = sigma_s[end_index] * left_distance
            
            # Sum full voxels
            center_tau = sum(sigma_s[end_index + 1:start_index]) * voxel_size
            
            logging.debug("Direction is to the left. Right_tau: %f, Left_tau: %f, Center_tau: %f", right_tau, left_tau, center_tau)
            tau = left_tau + right_tau + center_tau
    
    streaming_operator = np.exp(-tau) / (4 * np.pi * distance**2)
    logging.debug("Calculated K_trans: %f", streaming_operator)
    return streaming_operator

def construct_transition_matrix(num_voxels:int, voxel_size:float, sigma_s:np.ndarray, positions:list):
    """
    Construct the transition matrix.

    Parameters:
    - num_voxels (int): Number of spatial elements (voxels).
    - voxel_size (float): Size of each voxel.
    - sigma_s (numpy.ndarray): Vector of scattering cross-sections.
    - positions (list): List of positions corresponding to each voxel. Assume the list is formatted so that index zero goes with the first voxel and so forth.

    Returns:
    - numpy.ndarray: Transition matrix.
    """
    logging.debug("Constructing transition matrix with %d voxels and voxel size %f", num_voxels, voxel_size)
  
    if num_voxels <= 0:
        raise ValueError("Number of voxels must be positive and non-zero.")
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive and non-zero.")
        
    K_trans = np.zeros((num_voxels, num_voxels))
    for i in range(num_voxels):
        for j in range(num_voxels):
            if i == j:
                # Set diagonal values to one based on defintion of streaming operator
                K_trans[i, j] = 1
            else:
                ri = positions[i]
                rj = positions[j]
                K_trans[i, j] = calculate_K_trans(i, j, ri, rj, voxel_size, sigma_s)
    logging.debug("Constructed transition matrix: %s", K_trans)
    return K_trans

def calculate_phi(K_trans:np.ndarray, sigma_s:np.ndarray, s:np.ndarray):
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
    logging.debug("Calculated neutron flux vector: %s", phi)
    return phi

def read_input(file_path:str):
    """
    Reads input data from a specified JSON file.

    Parameters:
    - file_path (str): The path to the JSON file containing input data.

    Returns:
    - dict: A dictionary containing the input parameters loaded from the JSON file.
    """
    logging.debug("Reading input from %s", file_path)
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        logging.debug("Input data: %s", data)
    except FileNotFoundError:
        logging.error("File %s not found.", file_path)
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from file %s.", file_path)
        raise
    return data

def perform_calculation(input_data:dict):
    """
    Performs the main calculation for the neutron flux based on input parameters.

    Parameters:
    - input_data (dict): A dictionary containing input parameters including spatial boundaries, voxel count, source index, and scattering cross-sections.

    Returns:
    - numpy.ndarray: The neutron flux vector calculated from the input data.
    """
    logging.debug("Performing calculation with input data: %s", input_data)
    x_min = input_data.get("minimum_x")
    x_max = input_data.get("maximum_x")
    num_voxels = input_data.get("number_of_voxels")
    source_voxel_index = input_data.get("source_voxel_index")
    sigma_s_values = input_data.get("scattering_cross_section")
    positions = input_data.get("positions")

    voxel_size = (x_max - x_min) / num_voxels
    
    s = construct_source_term(num_voxels, source_voxel_index)
    K_trans = construct_transition_matrix(num_voxels, voxel_size, sigma_s_values, positions)
    phi = calculate_phi(K_trans, sigma_s_values, s)
    
    logging.info("Scattering Cross-section Vector (sigma_s):\n%s", sigma_s_values)
    logging.info("Source Vector (s):\n%s", s)
    logging.info("Transition Matrix (K_trans):\n%s", K_trans)
    logging.info("Neutron Flux Vector (phi):\n%s", phi)
    
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
    logging.debug("Writing data to CSV file %s", file_path)
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Voxel Index", "Neutron Flux"])
            for i, flux in enumerate(data):
                writer.writerow([i, flux])
        logging.info("Data written to CSV file %s", file_path)
    except Exception as e:
        logging.error("Error writing to CSV file: %s", e)
        raise

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

def main():
    setup_logging(logging.DEBUG)
    logging.debug("Main function started")
    input_data = read_input("input_data.json")
    phi = perform_calculation(input_data)
    write_output(phi)

if __name__ == "__main__":
    main()