from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist  # Calculate distance from 2 data
np.random.seed(11)

means = [[2, 2], [9, 2], [4, 9]] # Manually gennerate coordinate of means
conv = [[2, 0], [0, 2]]
n_sample = 500
n_cluster = 3 # K = 3 (We decide)
X0 = np.random.multivariate_normal(means[0], cov, n_sample)
X1 = np.random.multivariate_normal(means[1], cov, n_sample)
X2 = np.random.multivariate_normal(means[2], cov, n_sample)

X = np.concatenate((X0, X1, X2), axis = 0)


original_label = np.asarray([0]*N + [1]*N + [2]*N).T
