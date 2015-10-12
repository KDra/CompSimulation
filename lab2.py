# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:25:30 2015

@author: kostas
"""

import numpy as np
from scipy import constants
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)

eps = 1.
T = 2.
dr = 0.5
num_samples = 10
num_part = 3
num_dim = 2
L = 10.

#def LJ_pot(A, eps):
#def PBs(part_pos, parts_pos_mat, cutoff):
    

#def mc_LJ(num_part, num_dim, eps, T, dr, L, num_samples):
U = np.array([])
U_old = 1e16    # Var to store energy from prev iteration
pos = np.random.rand(num_part, num_dim) * L #Create random positions
dist_mat = np.zeros((num_part, num_part))   #Def a matrix to store distances
# Populate the distance matrix
for iter in range(num_part):
    for iter2 in range(iter + 1, num_part):
        # Check if other particles are closer in the vertical direction
        a = np.min([np.linalg.norm(pos[iter, :] - pos[iter2, :]), \
            np.linalg.norm(pos[iter, :] - pos[iter2, :])])
        # Check if other particles are closer in the horizontal direction
        b = np.min([np.linalg.norm(pos[iter, :] - pos[iter2, :]), \
            np.linalg.norm(pos[iter, :] - pos[iter2, :])])
        dist_mat[iter, iter2] = np.min([a, b])
# Start the MC process
for iter in range(num_samples):
    U_new = 0
    # Choose a random particle
    choice = int(np.random.rand() * num_part)
    # Move randomly in all required dimensions
    temp_pos = np.zeros(num_dim)
    for local_dim in range(num_dim):
        temp_pos[local_dim] = pos[choice, local_dim] + np.random.rand() * 2 * dr - dr
    # Define a temporary matrix to store values for the current iteration
    # and change only the values affected by the particle movement
    temp_dist_mat = dist_mat
    for i in range(num_part):
        if i < choice:
            temp_dist_mat[i, choice] = np.linalg.norm(pos[i, :] - temp_pos)
        elif i > choice:
            temp_dist_mat[choice, i] = np.linalg.norm(pos[i, :] - temp_pos)
        else:
            continue
    # Compute the energy level after the move
    U_new += eps * 4 * sum(temp_dist_mat[np.nonzero(temp_dist_mat)]**(-12)\
        - temp_dist_mat[np.nonzero(temp_dist_mat)]**(-6))
    p_acc = np.exp((U_old-U_new)/T)
    if U_new < U_old or np.random.rand() < p_acc:
        dist_mat = temp_dist_mat
        pos[choice, :] = temp_pos
        U_old = U_new
        U = np.append(U, U_new)
    else:
        pass
    
#    return U_new

#mc_LJ(num_part, num_dim, eps, T, dr, L)