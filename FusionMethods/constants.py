"""
Author: Kolja Thormann

Contains constants for consistent array indexing as well as parameters for the different methods
"""

import numpy as np

# constants for accessing x
X1 = 0
X2 = 1
M = [0, 1]  # center
AL = 2  # orientation
L = 3  # semi-axis length
W = 4  # semi-axis width
SR = [2, 3, 4]  # elements of square root matrix
V1 = 5
V2 = 6
V = [5, 6]
KIN = [0, 1, 5, 6]

N_PARTICLES_MMGW = 1000  # number of particles for mmgw estimation

# for mixture reduction
WEIGHT_THRESH = 1e-6  # weight threshold for discarding components
MAX_COMP = 32  # maximum number of components; prune if necessary
CLOSE_THRESH = 0.2

# motion model
SIGMA_V1 = 0.1
SIGMA_V2 = 0.1
SIGMA_SHAPE = np.array([0.1, 0.01, 0.01])

# measurement matrix
H = np.identity(7)
KIN_MEAS = [0, 1, 5, 6]
H_SHAPE = np.identity(4)

# process matrix
T = 1.0
F = np.array([
    [1, 0, 0, 0, 0, T, 0],
    [0, 1, 0, 0, 0, 0, T],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
])
