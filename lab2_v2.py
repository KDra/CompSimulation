# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:26:07 2015

@author: kostas
"""

import numpy as np
import itertools
from scipy import constants
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)

no_smples = 10000   # Number of samples
N = 100             # No of particles
dim = 2             # No of dimensions
a = 2               # Constant for density calculations
rho = a/10.         # Density
L = (N/rho)**(1/3)  # Box size
rc = L/2            # Cutoff radius
U = np.array([])    # Store all accepted energy levels
dist_mat = np.zeros([N, N])     # Store matrix distances

# Create a random set of N particles
pos = (np.random.rand(N, dim) * 2 - 1) * L
# Create a matrix to set up PBC
i = 0
b = np.empty([2**dim, dim]
for t in itertools.product([0, 1], repeat = dim):
    b[i, :] = np.array(t)
    i+=1
# Populate the distance matrix
for iter in np.arange(N):
    for iter2 in np.arange(iter + 1, N):
        for i in np.arange(2)
            a = nb.min(np.linalg.norm(pos[iter, :] - pos[iter2, :]),
                       np.linalg.norm(pos[iter, :] - [pos[iter2, 0]+L, pos[iter2, 1:]]))
            b = nb.min(np.linalg.norm(pos[iter, :] - [pos[iter2, 0],pos[iter2, 1]+L, pos[iter2, 2]]),
                       np.linalg.norm(pos[iter, :] - [pos[iter2, :1], pos[iter2, 2]+L]))
            c = nb.min(np.linalg.norm(pos[iter, :] - [pos[iter2, 0], pos[iter2, 1:]+L]),
                       np.linalg.norm(pos[iter, :] - [pos[iter2, :1]+L, pos[iter2, 2]]))

for i in np.arange(no_smples):
    choice = int(np.random.rand() * N)  # Choose a particle
    # Randomly move the particle
    for n in np.arange(dim):
        
    