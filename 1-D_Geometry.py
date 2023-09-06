import numpy as np

# Define extents of 1-D geometry
x_min = -10.0
x_max = 10.0
N = 6       # Discretization of geometry (Number of voxels)

# Define composition/properties of geometry
# Assume a uniform composition with attenuation coefficient mu
mu = 0.6  # Attenuation coefficient

# Define source distribution
source_distribution = np.ones(N)  # Assume uniform distrubution


# Ray-tracing between voxel centers
def calculate_distance(x_a, x_b):
    return abs(x_b - x_a)

# Task 2.2: Construct optical distance along ray


def calculate_optical_distance(sigma_t, distance):
    return (1-sigma_t) * distance

# Task 2.3: Calculate probability of interaction


def calculate_interaction_probability(optical_distance):
    return np.exp(-optical_distance)


probability_matrix = np.zeros((N, N))

for voxel_a in range(N):
    for voxel_b in range(N):
        distance = calculate_distance(voxel_a, voxel_b)
        optical_distance = calculate_optical_distance(mu, distance)
        probability = calculate_interaction_probability(optical_distance)
        probability_matrix[voxel_a, voxel_b] = probability

print(probability_matrix)
