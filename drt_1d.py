"""
Title: Deterministic Radiation Trasport - 1-D Toy Problem with Voxelized Geometry
Author: Damian Jilk
"""

import numpy as np
import json
import csv
import logging
import argparse

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

def calculate_K_trans(ri:int, rj:int, voxel_size:float, sigma_s:np.ndarray):
    """
    Calculate the streaming operator. 
    
    Used by the construct_transition_matrix function to build the K_trans matrix.

    Current method calculates tau by summing the cross-sections of the voxels from the start index to the end index, 
    always summing from the left edge of the starting voxel to the left edge of the ending voxel, regardless of the direction of transport.
    
    The calculation assumes equal voxel spacing.
    
    Based on principles from the book by Lewis and Miller (1984). Equations derived using Chapter 5. 

    Parameters:
    - ri (int): Index of the starting voxel.
    - rj (int): Index of the ending voxel.
    - voxel_size (float): Size of each voxel.
    - sigma_s (numpy.ndarray): Vector of scattering cross-sections.

    Returns:
    - float: Streaming operator. This represents the probability that a particle born at the left side of ri streams to the left side of rj without interacting.
    
    TODO: Update this function to accurately calculate tau for two arbitrary points within the voxelized space, considering the actual path of transport.
    """
    logging.debug(f"Calculating K_trans from voxel {ri} to voxel {rj} with voxel size {voxel_size}")
    
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive and non-zero.")
    if ri < 0 or ri >= len(sigma_s) or rj < 0 or rj >= len(sigma_s):
        raise ValueError("Voxel indices must be non-negative and within the range of the sigma_s_matrix.")
    
    min_index = min(ri, rj)
    max_index = max(ri, rj)
    distance = abs(ri - rj) * voxel_size
    tau = np.sum(sigma_s[min_index:max_index]) * voxel_size
    streaming_operator = np.exp(-tau) / (4 * np.pi * distance**2)
    logging.debug(f"Calculated K_trans: {streaming_operator}")
    return streaming_operator

def construct_transition_matrix(num_voxels:int, voxel_size:float, sigma_s:np.ndarray):
    """
    Construct the transition matrix.

    Parameters:
    - num_voxels (int): Number of spatial elements (voxels).
    - voxel_size (float): Size of each voxel.
    - sigma_s (numpy.ndarray): Vector of scattering cross-sections.

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
        # Set diagonal values to one 
        K_trans[i,i] = 1
        for j in range(i):
            K_trans[i, j] = calculate_K_trans(i, j, voxel_size, sigma_s)
            # The transition matrix is symmetric
            K_trans[j,i] = K_trans[i,j]
    logging.debug(f"Constructed transition matrix: {K_trans}")
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

    voxel_size = (x_max - x_min) / num_voxels
    
    s = construct_source_term(num_voxels, source_voxel_index)
    K_trans = construct_transition_matrix(num_voxels, voxel_size, sigma_s_values)
    phi = calculate_neutron_flux(K_trans, sigma_s_values, s)
    
    logging.info(f"Scattering Cross-section Vector (sigma_s):\n{sigma_s_values}")
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
    setup_logging()
    logging.debug("Main function started")
    args = parse_arguments()
    input_file_path = args.input_file
    input_data = read_input(input_file_path)
    phi = perform_calculation(input_data)
    write_output(phi)

if __name__ == "__main__":
    main()
