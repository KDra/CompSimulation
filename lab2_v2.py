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

def test_distances(f):
    assert np.linalg.norm(f(np.random.rand(10, 2), 2)) != 0.
    assert np.linalg.norm(f(np.random.rand(10, 2)*100, 2)) == 0.
    return "Success"

no_smples = 10000   # Number of samples
N = 100             # No of particles
dim = 2             # No of dimensions
a = 2               # Constant for density calculations
rho = a/10.         # Density
L = (N/rho)**(1/3.) # Box size
rc = L/2            # Cutoff radius


def distances(xArray, boxSize, cRatio=2):
    """
    Returns an N*N array of the distances between all particles in xArray. 
    The returned array is upper triangular (U-T) to avoid redundancies and the
    zeros in the U-T part show that the points are further apart than half the
    box size (assumed to be the cutoff distance)
    
        10/10/2015  K. Drakopoulos (kd1e15@soton.ac.uk)
    
    External variables:
        xArray      Contains array of point coordinates
        boxSize     The size of the domain containing the points
        cRatio      The ratio of the boxSize to be used for cutoff distances
        mDist       U-T matrix containing distances between all particles
    Local variables:
        rcut        Cutoff distance
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
        diff        Difference between two points
        dist        Distance between two points (norm)
    """
    N, dim = np.shape(xArray)   # Extraxt input matrix size
    rcut = boxSize / float(cRatio)  # Store the cutoff distance
    
    mDist = np.zeros([N, N])    # Initialise distance matrix
    mDist[:] = np.inf   # Set all values to infty to aid in energy calculations
    # Compute distance from each point to all others
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            diff = xArray[j, :] - xArray[i, :]
            for n in range(dim):
                if diff[n] > rcut:
                    diff[n] -= boxSize
                elif diff[n] < -rcut:
                    diff[n] += boxSize
            dist = np.linalg.norm(diff)
            if dist < rcut**dim:
                mDist[i, j] = dist
    return mDist


def rnd_move(xArray, boxSize, cRatio=2.):
    """
    Returns the input array (xArray) with one element that has been randomly 
    moved within a box wih periodic boundaries of the distance given by boxSize
    and centred at 0. The boxSize and cRatio are used to scale the movement.
    
        10/10/2015  K. Drakopoulos (kd1e15@soton.ac.uk)
    
    External variables:
        xArray      Contains array of point coordinates
        boxSize     The size of the domain containing the points
        cRatio      The ratio of the boxSize to be used for scaling of movement
    Internal variables:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
        rd          A random index to choose a point in xArray
    """
    
    N, dim = np.shape(xArray)       # Store input matrix dimensions
    rd = int(np.random.rand()*N)    # Choose rand point and scale to N
    # Randomly move point in all dimensions with scaling
    for i in np.arange(dim):
        xArray[rd, i] += (np.random.rand()*2 - 1) * boxSize/cRatio
        # Ensure that the moved point lies within the given box size
        if xArray[rd, i] > boxSize/2.:
            xArray[rd, i] -= boxSize
        if xArray[rd, i] < -boxSize/2.:
            xArray[rd, i] += boxSize
    return xArray


def mc_LJ(N, dim, rho, no_smples, boxSize, cRatio):
    """
    Return the potential energy of a set of particles using Lennard-Jones
    interaction.
    External variables:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
    Internal variables:
        
    """
    rc = boxSize * float(cRatio)
    # Create a random set of N particles
    pos = (np.random.rand(N, dim) * 2 - 1) * boxSize
    U = np.array([])    # Store all accepted energy levels
    utail = 8 * constants.pi * rho/3. * (1/3. * rc**(-9) - rc**(-3))
    for i in np.arange(no_smples):
        mDist = distances(pos, boxSize)
        u = np.sum(4 * (mDist**(-12) - mDist**(-6))) + utail
        U = np.append(U, u)
        pos = rnd_move(pos, boxSize)
        print "Iteration number %d, energy is %d" %(i, U[i])
    return U, np.mean(U)
    
    

def mc_pressure(N, dim, eps, sigma, no_smples, boxSize, cRatio):
    pass






















