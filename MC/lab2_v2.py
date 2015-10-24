# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:26:07 2015

@author: kostas
"""

from __future__ import division
from multiprocessing import Process, Queue

import numpy as np
from numba import jit
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)


def test_distances(f):
    assert np.linalg.norm(f(np.random.rand(10, 2), 2)) != 0.
    assert np.linalg.norm(f(np.random.rand(10, 2)*100, 2)) == 0.
    return "Success"
"""
no_smples = 10000   # Number of samples
N = 100            # No of particles
dim = 3             # No of dimensions
a = 8               # Constant for density calculations
rc = 0.5            # Cutoff radius
T = 2.
"""

#@jit
def pbcDiff(xArray, i, j, boxSize, cRatio):
    """
    Return the difference (vectorial) between two particles while taking into
    account possible interactions across the periodic boundaries. If the distance
    is higher than the cutoff radius a 0 is returned.
            10/10/2015  K. Drakopoulos (kd1e15@soton.ac.uk)
    Inputs:
        xArray      Contains array of point coordinates
        boxSize     The size of the domain containing the points
        i           Index of the first particle
        j           Index of the seceond particle
        cRatio      The ratio of the boxSize to be used for cutoff distances
    Output:
        diff        The difference between two particles
    Internal variables:
        cases       Vector to check for interactions across the boundary
        index       Stores the index of the minimum distance between two particles
                    after a check is made for interactions over the boundary
    """
    N, dim = np.shape(xArray)
    diff = xArray[j, :] - xArray[i, :]
    # Contains three cases: closer as is, closer to the left, closer to the right
    cases = np.array([0, -boxSize, boxSize])
    for n in range(dim):
        # Add to the directions of interest to see if particles are closer over
        # the periodic boundaries
        index = np.argmin(np.abs(diff[n] + cases))
        diff[n] += cases[index]
    # Check if the cutoff condition is satisfied
    if np.linalg.norm(diff) < boxSize * float(cRatio):
        return diff
    else:
        return 0.0


#@jit
def distances(xArray, boxSize, mDist=None, rd=None, cRatio=0.5):
    """
    Returns an N*N array of the distances between all particles in xArray.
    The returned array is upper triangular (U-T) to avoid redundancies and the
    0s in the U-T part show that the points are further apart than cutoff distance.
            10/10/2015  K. Drakopoulos (kd1e15@soton.ac.uk)
    Inputs:
        xArray      Contains array of point coordinates
        boxSize     The size of the domain containing the points
        mDist       U-T matrix containing distances between all particles (opt)
        rd          Particle index where a change has occured (opt)
        cRatio      The ratio of the boxSize to be used for cutoff distances(opt)
    Outputs
        mDistOut    Compy of mDist with the changed distances
        my_bool     Returns a bool matrix to index ('True' values the changes 
                    made in the distance matrix and of all non-zero elements (opt)
    Local variables:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
    """
    N, dim = np.shape(xArray)   # Extraxt input matrix size
    # Check if the distance matrix has been computed before, i.e. if it is an input
    if mDist is not None:
        assert rd is not None,\
               "Please also input the modified particle index"
        my_bool = np.zeros(np.shape(mDist), dtype=bool)
        mDistOut = mDist.copy()
        for i in np.arange(rd):
            mDistOut[i, rd] = np.linalg.norm(
                pbcDiff(xArray, i, rd, boxSize, cRatio))
            if mDistOut[i, rd] > 0:
                my_bool[i, rd] = True
        for i in np.arange(rd, N):
            mDistOut[rd, i] = np.linalg.norm(
                pbcDiff(xArray, rd, i, boxSize, cRatio))
            if mDistOut[rd, i] > 0:
                my_bool[rd, i] = True
        return mDistOut, my_bool
    else:
        mDistOut = np.zeros([N, N])    # Initialise distance matrix
        # Compute distance from each particle to all others
        for i in np.arange(N):
            for j in np.arange(i+1, N):
                mDistOut[i, j] = np.linalg.norm(
                    pbcDiff(xArray, i, j, boxSize, cRatio))
        return mDistOut

@jit
def rnd_move(xArray, boxSize, cRatio=0.1, rnd=0):
    """
    Returns the input array with one particle that has been randomly
    moved within a box wih periodic boundaries of the distance given by boxSize
    and centred at 0. The boxSize and cRatio are used to scale the movement.
            10/10/2015  K. Drakopoulos (kd1e15@soton.ac.uk)
    Inputs:
        xArray      Contains array of point coordinates
        boxSize     The size of the domain containing the points
        cRatio      The ratio of the boxSize to be used for scaling of movement
    Outputs:
        yArray      A copy of the xArray to avoid overwriting memory
        rd          A random index to choose a point in xArray
    Internal variables:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
    """
    N, dim = np.shape(xArray)       # Store input matrix dimensions
    rd = int(np.random.rand()*N)    # Choose rand point and scale to N
    yArray = xArray.copy()
    # Randomly move point in all dimensions with scaling
    for i in np.arange(dim):
        if rnd==0:
            yArray[rd, i] += (np.random.rand() - 0.5) * cRatio * boxSize
        elif rnd==1:
            yArray[rd, i] += np.random.normal() * cRatio * boxSize
        else:
            raise ValueError("The only accepted values are 0 for uniform and 1 for gaussian/normal\
                             distributions. You have entered {}.".format(rnd))
        # Ensure that the moved point lies within the given box size
        if yArray[rd, i] > boxSize/2.:
            yArray[rd, i] -= boxSize
        if yArray[rd, i] < -boxSize/2.:
            yArray[rd, i] += boxSize
    return yArray, rd


@jit
def pressure(xArray, mDist, boxSize, cRatio):
    """
    Returns the pressure given the particle positions and the distances using
    the Lennard-Jones approximation.
            10/10/2015  K. Drakopoulos (kd1e15@soton.ac.uk)
    Inputs:
        xArray      Contains array of point coordinates
        mDist       U-T matrix containing distances between all particles
    Outputs:
        p           Pressure
    Internal variables:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
        diff        Difference between two points (vector)
        inner       Inner product    
    """
    N, dim = np.shape(xArray)
    p = 0.0
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            # Store distance in a variable to avoid accessing it repeatedly
            r = mDist[i, j]
            # Compute pressure only if distance is non-zero, i.e. the particles
            # are closer than the cutoff radius
            if r > 0:
                diff = pbcDiff(xArray, i, j, boxSize, cRatio)
                inner = np.dot(diff, diff)
                p += 8.0 * inner * (2 * r**(-14) - r**(-8))
    return p


@jit
def mc_LJ(N, dim, a, T, no_smples, cRatio, delta, dist=0, q=None):
    """
    Return the potential final positions, energy, pressure and the number of
    accepted trials (throught the length of the enery vector - 1) for a set of 
    particles using Lennard-Jones interaction. The distribution from which the
    particle movement is extracted can also be specified.
            10/10/2015  K. Drakopoulos (kd1e15@soton.ac.uk)
    Inputs:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
        a           Density magnitude
        T           Temperature
        no_smples   Number of random particle movements to be tried
        cRatio      A ratio of the box side length for radius cutoff
        dist        Specify the function that gives the random distribution
        q           A queue to be used to store the results when run in parallel(opt)
    Outputs:
        pos         Final position of particles for the relaxed system
        U           Vector of energies for all accepted cases
        p           Pressure of the final configuration (relaxed system)
    Internal variables:
        rho         Density
        boxSize     Length of one side of the periodic cube
        rc          Cutoff distance
        utail       Add an approximation for the contribution of particles further
                    than rc to the Energy
        ptail       Same as above but for Pressure
        temp_pos    Store positions for the trial move
        temp_mDist  Store distances for the trial move
        mBool       A bool array to store the indices where the current distance
                    matrix has been changed and where it is non-zero
        mBool_oCfg  A bool array to store the indeces of the previously accepted
                    distance matrix so that it can be easily subtracted from the
                    previously accepted energy level
        u           Variable to store the energy level due to the current move
        pv          Variable to store the virial part of the pressure / Volume
        
    """
    rho = float(a)/10.0           # Density
    boxSize = (N/rho)**(1/3.)       # Box side length
    rc = boxSize * float(cRatio)    # Compute the cutoff radius
    # Create a random set of N particles and compute the distances between them
    pos = (np.random.rand(N, dim) - .5) * boxSize
    mDist = distances(xArray=pos, boxSize=boxSize)
    # Compute tail values for energy (utail) and pressure (ptail)
    utail = 8 * constants.pi * rho/3.0 * (1/3. * rc**(-9) - rc**(-3))
    ptail = 16 * constants.pi * (rho**2)/3.0 * (2/3. * rc**(-9) - rc**(-3))
    # Initialise an array to store the energies with the value given by current
    # distances
    U = np.array([np.sum(4 * (mDist[mDist>0]**(-12) - mDist[mDist>0]**(-6))) + utail])
    E = np.zeros((no_smples,1))
    E[0] = U[0]
    for i in np.arange(no_smples):
        # Move particle and store new positions, then compute distances and a 
        # bool matrix to determine where replacements have occured
        temp_pos, rd = rnd_move(pos, boxSize, rnd=dist, cRatio=delta)
        temp_mDist, mBool = distances(temp_pos, boxSize, mDist, rd)
        # Create a bool matrix for the old non-zero indeces of the accepted mDist
        mBool_oCfg = np.zeros(np.shape(mBool), dtype = bool)
        mBool_oCfg[:rd, rd] = True
        mBool_oCfg[rd, rd:] = True
        mBool_oCfg = (mDist > 0) & mBool
        # Compute energy by subtracting from the last accepted value the
        # contribution from the prev distances and adding the new
        u = U[-1] - np.sum(4 * (mDist[mBool_oCfg]**(-12) - mDist[mBool_oCfg]**(-6))) +\
                np.sum(4 * (temp_mDist[mBool]**(-12) - temp_mDist[mBool]**(-6)))
        # Accept move either if the energy is lower or if the gaussian is larger
        # than a random value. Update all components
        if (u < U[-1]) or (np.exp((U[-1]-u)/float(T)) > np.random.rand()):
            U = np.append(U, u)
            mDist = temp_mDist
            pos = temp_pos
        E[i] = u
    U = U.reshape(len(U),1)
    pv = pressure(pos, mDist, boxSize, cRatio)/(float(N) / rho)
    p = rho*T + pv + ptail
    # Enables the use of a queue for parallel processing
    if q is not None:
        q.put((a, pos, U, E, p))
    return pos, U, E, p



"""
q = Queue()
d={}
for i in range(1, 10):
    p = Process(target=mc_LJ, args=(N, dim, i, T, no_smples, rc))
    p.start()

p = np.array([])
acc = np.array([])
U = np.empty((no_smples, 9))
for i in range(1, 10):
    temp = q.get()
    d["a{}".format(temp[0])] = temp[1:]

for i in range(1, 10):
    print "Pressure for a = %d is %f" %(i, d["a{}".format(i)][2])
    p = np.append(p, d["a{}".format(i)][2])
    acc = np.append(acc, d["a{}".format(i)][3])
    u = np.array(d["a{}".format(i)][1])
    sh = len(u)
    U[:sh, i-1] = u


p2 = Process(target=mc_LJ, args=(N, dim, 2, T, no_smples, rc))
p3 = Process(target=mc_LJ, args=(N, dim, 3, T, no_smples, rc))
p4 = Process(target=mc_LJ, args=(N, dim, 4, T, no_smples, rc))

#a, b, c, d, e = mc_LJ(N, dim, a, T, no_smples, rc)

p1.start()
p2.start()
p3.start()
p4.start()

res1 = q.get()
res2 = q.get()
res3 = q.get()
res4 = q.get()
"""
#debug
def perf(mArray):
    return np.log(np.abs(mArray - np.min(mArray) + 1))
    
    
A = {}
for i in range(3):
    A["run{}".format(i)] = mc_LJ(N=50, dim=3, a=5.0, T=2.0, no_smples=5000, cRatio=0.5, dist=1, delta=0.024)
enA1 = np.empty((5000,1))
for i in range(3):
    enA1 = np.append(enA1, A["run{}".format(i)][2], axis=1)
enA1 = np.delete(enA1, 0, 1) 
avgA = np.mean(enA1, axis=1)
"""
@jit
def search(x, mArray, indX, dst):
    len_ = len(mArray)
    if len_ < dst:
        dst = len_
    mid = int(dst/2)
    a = indX - mid
    b = indX + int(dst-mid)
    if a < 0:
        b = b - a
        a = 0
    if b > len_:
        a = a - (b - len_)
        b = len_
    indY = np.argmin(np.abs(mArray[a:b+1] - x))
    return indY


@jit
def adjust(xArray, yArray, dst):
    if len(xArray) > len(yArray):
        tempArray = yArray.copy()
        for i in np.arange(len(yArray)):
            ind = search(yArray[i], xArray, i, dst)
            if ind > i:
                
    elif len(xArray) < len(yArray):
        
    else:
        return xArray, yArray
"""








