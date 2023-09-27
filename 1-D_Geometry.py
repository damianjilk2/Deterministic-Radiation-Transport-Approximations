"""
Title: 1-D Geometry

Description:
This script attempts to approximate determinisitc radiation transport for a 1-D geometry.

Contents:

TODO:
- should forward solution even be calculated? If yes, merge adjoint and foward solutions with a method
- code review and tidying up
- user input?
- how to add obstacles for the neutrons?

Author: Damian Jilk
Date: 09/27/2023
"""
import numpy as np

# Define extents of 1-D geometry
x_min = -10.0
x_max = 10.0
N = 6       # Discretization of geometry (Number of voxels)

# Define composition/properties of geometry
sigma_s = 0.4  # Scattering cross-section
sigma_a = 0.2  # Absorption cross-section


def calculate_distance(x_a: float, x_b: float):
    """
    Calculate the distance between two points along a 1-D geometry.

    Args:
        x_a (float): The x-coordinate of point A.
        x_b (float): The x-coordinate of point B.

    Returns:
        float: The absolute distance between point A and point B.
    """
    return abs(x_b - x_a)

# Task 2.2: Construct optical distance along ray

# Calculate macroscopic cross-section (Sigma_t_m)


def calculate_macroscopic_cross_section(sigma_s_m: float, sigma_a_m: float):
    """
    Calculate the total cross section.

    Args:
        sigma_s_m (float): The scattering cross section.
        sigma_a_m (float): The absorption cross section.

    Returns:
        float: The total cross section.
    """
    return sigma_s_m + sigma_a_m


def calculate_optical_distance(sigma_t_m: float, distance: float):
    """
    Calculate the optical distance along a ray given the attenuation coefficient and distance.

    Args:
        sigma_t_m (float): The attenuation coefficient.
        distance (float): The physical distance along the ray.

    Returns:
        float: The optical distance along the ray.
    """
    return sigma_t_m * distance

# Task 2.3: Calculate probability of interaction


def calculate_interaction_probability(optical_distance: float):
    """
    Calculate the probability of interaction given the optical distance.

    Args:
        optical_distance (float): The optical distance along the ray.

    Returns:
        float: The probability of interaction.
    """
    return np.exp(-optical_distance)


probability_matrix = np.zeros((N, N))  # initialize probability matrix

# iterate through all voxel pairs
for voxel_a in range(N):
    for voxel_b in range(N):
        distance = calculate_distance(voxel_a, voxel_b)
        sigma_t_m = calculate_macroscopic_cross_section(sigma_s, sigma_a)
        optical_distance = calculate_optical_distance(sigma_t_m, distance)
        probability = calculate_interaction_probability(optical_distance)
        probability_matrix[voxel_a, voxel_b] = probability

print(f"Probability matrix (H): \n{probability_matrix}")


iterations = 500  # Number of iterations for the sum
# forward solution

# Define source distribution
source_location = 0  # Point source [0, ..., N-1]
source_distribution = np.zeros(N)
source_distribution[source_location] = 1

matrix_iteration = np.linalg.matrix_power(probability_matrix, 0)
for i in range(iterations):
    if i == 0:
        continue
    matrix_iteration += np.linalg.matrix_power(probability_matrix, i)

F = np.dot(matrix_iteration, source_distribution)

print(f"Forward solution: \n{F}")


# adjoint solution

# Define adjoint source distribution
adj_source_location = 0  # Point source [0, ..., N-1]
adj_source_distribution = np.zeros(N)
adj_source_distribution[adj_source_location] = 1

adj_probability_matrix = np.transpose(probability_matrix)
adj_matrix_iteration = np.linalg.matrix_power(adj_probability_matrix, 0)
for i in range(iterations):
    if i == 0:
        continue
    adj_matrix_iteration += np.linalg.matrix_power(adj_probability_matrix, i)

A = np.dot(adj_matrix_iteration, adj_source_distribution)

print(f"Adjoint solution: \n{A}")
