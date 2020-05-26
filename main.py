"""
Author: Kolja Thormann

Evaluation for
Thormann, Kolja; Baum, Marcus (2019): Bayesian Fusion of Orientation, Width and Length Estimates of Elliptic
Extended Targets based on the Gaussian Wasserstein Distance. TechRxiv. Preprint.
"""

import numpy as np

from FusionMethods.tests import test_convergence, test_mean


# setup ================================================================================================================
# Gaussian prior
prior = np.array([0, 0, 0.0 * np.pi, 8, 3, 10, 0])  # [m1, m2, alpha, length, width, v1, v2]
cov_prior = np.array([
    [0.5, 0.0, 0,         0,     0,   0,   0],
    [0.0, 0.5, 0,         0,     0,   0,   0],
    [0,   0,   0.5*np.pi, 0,     0,   0,   0],
    [0,   0,   0,         0.5,   0,   0,   0],
    [0,   0,   0,         0,   0.5,   0,   0],
    [0,   0,   0,         0,     0, 10.0,   0],
    [0,   0,   0,         0,     0,   0, 10.0],
])

# sensor A and B noise
cov_meas = np.array([
    [0.5, 0.0, 0,          0,     0,    0,    0],
    [0.0, 0.5, 0,          0,     0,    0,    0],
    [0,   0,   0.5*np.pi, 0,     0,    0,    0],
    [0,   0,   0,          0.5,   0,    0,    0],
    [0,   0,   0,          0,   0.1,    0,    0],
    [0,   0,   0,          0,     0, 0.05,    0],
    [0,   0,   0,          0,     0,    0, 0.05],
])

runs = 1000  # number of MC runs
steps = 20  # number of measurements (alternating sensors A and B)

save_path = './'

# tests ================================================================================================================
test_convergence(steps, runs, prior, cov_prior, cov_meas, True, save_path)
# test_mean(prior, cov_prior, 1000, save_path)
