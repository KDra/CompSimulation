# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:26:07 2015

@author: kostas
"""

import numpy as np
from numba import jit
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
T = 2

@jit
def pbcDiff(xArray, i, j, boxSize, rcut):
    diff = xArray[j, :] - xArray[i, :]
    for n in range(dim):
        if diff[n] > rcut:
            diff[n] -= boxSize
        elif diff[n] < -rcut:
            diff[n] += boxSize
    dist = np.linalg.norm(diff)
    if dist < rcut:
        return dist
    else:
        return np.inf



def distances(xArray, boxSize, mDist=None, rd=None, cRatio=0.5):
    """
    Returns an N*N array of the distances between all particles in xArray.
    The returned array is upper triangular (U-T) to avoid redundancies and the
    'inf's in the U-T part show that the points are further apart than half the
    box size (assumed to be the cutoff distance).
    
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
    rcut = boxSize * float(cRatio)  # Store the cutoff distance
    # Short internal function to compute distances and return if smaller than
    # the cutoff radius

    if not mDist is None:
        assert rd!=None, "Please also input the modified particle index"
        for i in np.arange(rd):
            mDist[i, rd] = pbcDiff(xArray, i, rd, boxSize, rcut)
        for i in np.arange(rd+1, N):
            mDist[rd, i] = pbcDiff(xArray, rd, i, boxSize, rcut)
    else:
        mDist = np.zeros([N, N])    # Initialise distance matrix
        mDist[:] = np.inf   # Set all values to infty to aid in energy calculations
        # Compute distance from each point to all others
        for i in np.arange(N):
            for j in np.arange(i+1, N):
                mDist[i, j] = pbcDiff(xArray, i, j, boxSize, rcut)
    return mDist

@jit
def rnd_move(xArray, boxSize, cRatio=0.01):
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
        xArray[rd, i] += (np.random.rand() - 0.5) * boxSize * cRatio
        # Ensure that the moved point lies within the given box size
        if xArray[rd, i] > boxSize/2.:
            xArray[rd, i] -= boxSize
        if xArray[rd, i] < -boxSize/2.:
            xArray[rd, i] += boxSize
    return xArray, rd


def mc_LJ(N, dim, rho, T, no_smples, cRatio):
    """
    Return the potential energy of a set of particles using Lennard-Jones
    interaction.
    External variables:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
    Internal variables:
        
    """
    boxSize = (N/rho)**(1/3.)
    acc = 0
    rc = boxSize * float(cRatio)
    # Create a random set of N particles
    pos = (np.random.rand(N, dim) - .5) * boxSize
    mDist = distances(xArray=pos, boxSize=boxSize)
    U = np.array([np.inf])    # Store all accepted energy levels
    utail = 8 * constants.pi * rho/3. * (1/3. * rc**(-9) - rc**(-3))
    ptail = 16 * constants.pi * rho/3. * (2/3. * rc**(-9) - rc**(-3))
    for i in np.arange(no_smples):
        u = np.sum(4 * (mDist**(-12) - mDist**(-6))) + utail
        temp_pos, rd = rnd_move(pos, boxSize)
        temp_mDist = distances(pos, boxSize, mDist, rd)
        if (u < U[-1]) or (np.exp((U[-1]-u)/T) > np.random.rand):
            U = np.append(U, u)
            mDist = temp_mDist
            pos = temp_pos
            acc += 1
            print "Iteration number %d, energy is %d" %(i, U[-1])
    p = pressure(pos, mDist, rho, boxSize, cRatio) + ptail
    return U, p, acc
    
    

def pressure(xArray, mDist, rho, boxSize, cRatio):
    

#a, b = mc_LJ(N, dim, rho, T, no_smples, 2.)




















