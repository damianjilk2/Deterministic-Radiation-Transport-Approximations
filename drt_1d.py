"""
Title: Deterministic Radiation Trasport - 1-D Toy Problem with Voxelized Geometry
Author: Damian Jilk
"""

import numpy as np
import json
import csv

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
    if num_voxels <= 0:
        raise ValueError("Number of voxels must be a positive integer.")
    if source_voxel_index < 0 or source_voxel_index >= num_voxels:
        raise IndexError("Source voxel index must be within the range of voxels.")

    s = np.zeros(num_voxels)
    s[source_voxel_index] = source_strength
    return s

def calculate_K_trans(ri:int, rj:int, voxel_size:float, sigma_s_matrix:np.ndarray):
    """
    Calculate the neutron transition probability. Used by the construct_transition_matrix function to build the K_trans matrix.
    The current method calculates tau by summing the cross-sections that are being traveled through. The standard in place is to include the starting point but not the ending point. 
    For example, with starting point 1 and ending 4, the cross-section values of voxels 1, 2, and 3 will be added to find the adjusted cross-section.
    
    Parameters:
    - ri (int): Index of the starting voxel.
    - rj (int): Index of the ending voxel.
    - voxel_size (float): Size of each voxel.
    - sigma_s_matrix (numpy.ndarray): Diagonal matrix of scattering cross-sections.

    Returns:
    - float: Neutron transition probability.
    """
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive and non-zero.")
    if ri < 0 or ri >= sigma_s_matrix.shape[0] or rj < 0 or rj >= sigma_s_matrix.shape[0]:
        raise ValueError("Voxel indices must be non-negative and within the range of the sigma_s_matrix.")
    
    min_index = min(ri, rj)
    max_index = max(ri, rj)
    distance = abs(ri - rj) * voxel_size
    tau = np.sum(np.diag(sigma_s_matrix[min_index:max_index, min_index:max_index])) * voxel_size
    return np.exp(-tau) / (4 * np.pi * distance**2)

def construct_transition_matrix(num_voxels:int, voxel_size:float, sigma_s_matrix:np.ndarray):
    """
    Construct the transition matrix.

    Parameters:
    - num_voxels (int): Number of spatial elements (voxels).
    - voxel_size (float): Size of each voxel.
    - sigma_s_matrix (numpy.ndarray): Diagonal matrix of scattering cross-sections.

    Returns:
    - numpy.ndarray: Transition matrix.
    """
    if num_voxels <= 0:
        raise ValueError("Number of voxels must be positive and non-zero.")
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive and non-zero.")
    
    K_trans = np.zeros((num_voxels, num_voxels))
    for i in range(num_voxels):
        for j in range(num_voxels):
            if i == j:
                # Set diagonal values to small positive value
                K_trans[i, j] = 1e-20
            else:
                K_trans[i, j] = calculate_K_trans(i, j, voxel_size, sigma_s_matrix)
    return K_trans

def calculate_phi(K_trans:np.ndarray, sigma_s_matrix:np.ndarray, s:np.ndarray):
    """
    Calculate the neutron flux vector.

    Parameters:
    - K_trans (numpy.ndarray): Transition matrix.
    - sigma_s_matrix (numpy.ndarray): Diagonal matrix of scattering cross-sections.
    - s (numpy.ndarray): Source term vector.

    Returns:
    - numpy.ndarray: Neutron flux vector.
    """
    phi = np.dot(np.linalg.inv(np.eye(len(K_trans)) - np.dot(K_trans, sigma_s_matrix)), np.dot(K_trans, s))
    return phi

def main():
    # Load global variables from the JSON file
    with open("input_data.json", "r") as json_file:
        input_data = json.load(json_file)

    x_min = input_data.get("minimum_x")
    x_max = input_data.get("maximum_x")
    num_voxels = input_data.get("number_of_voxels")
    source_voxel_index = input_data.get("source_voxel_index")
    sigma_s_values = input_data.get("scattering_cross_section")
    sigma_s_matrix = np.diag(sigma_s_values)
    print("Scattering Cross-section Matrix (sigma_s):\n", sigma_s_matrix)

    # Calculate voxel size
    voxel_size = (x_max - x_min) / num_voxels
    
    # Calculate source vector
    s = construct_source_term(num_voxels, source_voxel_index)
    print(f"Source Vector (s):\n{s}")
    
    # Calculate transition matrix
    K_trans = construct_transition_matrix(num_voxels, voxel_size, sigma_s_matrix)
    print(f"Transition Matrix (K_trans):\n{K_trans}")
    
    # Calculate neutron flux vector
    phi = calculate_phi(K_trans, sigma_s_matrix, s)
    print(f"Neutron Flux Vector (phi):\n{phi}")
    
    # Write results to CSV file
    with open("results.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Voxel Index", "Neutron Flux"])
        for i, flux in enumerate(phi):
            writer.writerow([i, flux])
    print("Results have been written to 'results.csv'.")

if __name__ == "__main__":
    main()
