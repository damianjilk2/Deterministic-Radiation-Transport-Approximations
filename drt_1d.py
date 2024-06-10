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
    s = np.zeros(num_voxels)
    s[source_voxel_index] = source_strength
    return s

def calculate_K_trans(ri:int, rj:int, voxel_size:float, sigma_s_matrix:np.ndarray):
    """
    Calculate the neutron transition probability. Used by the construct_transition_matrix function to build the K_trans matrix.

    Parameters:
    - ri (int): Index of the starting voxel.
    - rj (int): Index of the ending voxel.
    - voxel_size (float): Size of each voxel.
    - sigma_s_matrix (numpy.ndarray): Diagonal matrix of scattering cross-sections.

    Returns:
    - float: Neutron transition probability.
    """
    min_index = min(ri, rj)
    max_index = max(ri, rj)
    distance = abs(ri - rj) * voxel_size
    tau = np.sum(np.diag(sigma_s_matrix[min_index:max_index])) * voxel_size
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
    sigma_s_values = input_data.get("scattering_cross_section")
    sigma_s_matrix = np.diag(sigma_s_values)
    print("Scattering Cross-section Matrix (sigma_s):\n", sigma_s_matrix)

    # Calculate voxel size
    voxel_size = (x_max - x_min) / num_voxels
    
    # Calculate source vector
    s = construct_source_term(num_voxels, 2, 1) #currently using arbitrary values
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
