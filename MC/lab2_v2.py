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

no_smples = 5000   # Number of samples
N = 50             # No of particles
dim = 3             # No of dimensions
a = 8               # Constant for density calculations
rc = 0.5            # Cutoff radius
T = 2.


#@jit
def pbcDiff(xArray, i, j, boxSize, cRatio):
    diff = xArray[j, :] - xArray[i, :]
    adder = np.array([0, -boxSize, boxSize])
    for n in range(dim):
        iter_ = np.argmin(np.abs([diff[n], diff[n] - boxSize, diff[n] + boxSize]))
        diff[n] += adder[iter_] 
        """
        if diff[n] > boxSize/2.:
            diff[n] -= boxSize
        elif diff[n] < -boxSize/2.:
            diff[n] += boxSize
        """
    if np.linalg.norm(diff) < boxSize * float(cRatio):
        return diff
    else:
        return np.inf


#@jit
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
    # Short internal function to compute distances and return if smaller than
    # the cutoff radius

    if mDist is not None:
        assert rd is not None,\
               "Please also input the modified particle index"
        mDistOut = mDist.copy()
        for i in np.arange(rd):
            mDistOut[i, rd] = np.linalg.norm(
                pbcDiff(xArray, i, rd, boxSize, cRatio))
        for i in np.arange(rd+1, N):
            mDistOut[rd, i] = np.linalg.norm(
                pbcDiff(xArray, rd, i, boxSize, cRatio))
    else:
        mDistOut = np.zeros([N, N])    # Initialise distance matrix
        mDistOut[:] = np.inf           # Set to inf to aid in energy calculations
        # Compute distance from each point to all others
        for i in np.arange(N):
            for j in np.arange(i+1, N):
                mDistOut[i, j] = np.linalg.norm(
                    pbcDiff(xArray, i, j, boxSize, cRatio))
    return mDistOut

#@jit
def rnd_move(xArray, boxSize, cRatio=0.4):
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
    yArray = xArray.copy()
    # Randomly move point in all dimensions with scaling
    yArray[rd, :] += (np.random.rand(dim) - 0.5) * cRatio * boxSize
    for i in np.arange(dim):
        # Ensure that the moved point lies within the given box size
        if yArray[rd, i] > boxSize/2.:
            yArray[rd, i] -= boxSize
        if yArray[rd, i] < -boxSize/2.:
            yArray[rd, i] += boxSize
    return yArray, rd

#@jit
def updateU(mDist, mDist_old = None, rd_acc=None, rd=None, U=None):
    if U is not None:
        if U == np.inf:
            U = 0
        assert a is not None, "Please give the index of the moved particle"
        assert mDist_old is not None, "Please give the old distance matrix"
        distV_old = np.append(mDist_old[:, rd], mDist_old[rd, :])
        distV = np.append(mDist[:, rd], mDist[rd, :])
        U_new = U + np.sum(4 * (distV**(-12) - distV**(-6))) -\
                        np.sum(4 * (distV_old**(-12) - distV_old**(-6)))
    else:
        U_new = np.sum(4 * (mDist**(-12) - mDist**(-6)))
    return U_new
        

@jit
def pressure(xArray, mDist, rho, boxSize, cRatio):
    N, dim = np.shape(xArray)
    rcut = boxSize * cRatio
    #f = open('my.dat', 'w')
    p = 0
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            diff = pbcDiff(xArray, i, j, boxSize, cRatio)
            inner = np.dot(diff, diff)
            if inner == np.inf:
                inner = 0.
            a_ = 8.0 * inner * (2 * mDist[i, j]**(-14) -
                                            mDist[i, j]**(-8))
            p += a_
            #print str(a_) + "\t" + str(inner)
            """f.write(str(p)+"\n")
    f.close()"""
    return p


def mc_LJ(N, dim, a, T, no_smples, cRatio):
    """
    Return the potential energy of a set of particles using Lennard-Jones
    interaction.
    External variables:
        N           Number of points, i.e. rows
        dim         Number of dimensions, i.e columns
    Internal variables:
    """
    f = open('my.dat', 'w')
    rho = a/10.         # Density
    boxSize = (N/rho)**(1/3.)
    acc = 0
    rc = boxSize * float(cRatio)
    # Create a random set of N particles
    pos = (np.random.rand(N, dim) - .5) * boxSize
    posO=pos
    mDist = distances(xArray=pos, boxSize=boxSize)
    U = np.array([np.inf])    # Store all accepted energy levels
    utail = 8 * constants.pi * rho/3. * (1/3. * rc**(-9) - rc**(-3))
    ptail = 16 * constants.pi * (rho**2)/3. * (2/3. * rc**(-9) - rc**(-3))
    for i in np.arange(no_smples):
        temp_pos, rd = rnd_move(pos, boxSize)
        temp_mDist = distances(temp_pos, boxSize, mDist, rd)
        u1 = np.sum(4 * (temp_mDist**(-12) - temp_mDist**(-6))) + utail
        print u1
        u = updateU(temp_mDist, mDist, rd=rd, U=U[-1])
        print u
        f.write(str(u1) + "\t+" + str(u) + "\n")
        if (u < U[-1]) or (np.exp((U[-1]-u)/float(T)) > np.random.rand()):
            U = np.append(U, u)
            mDist = temp_mDist
            pos = temp_pos
            rd_acc = rd
            acc += 1
    f.close()
    pv = pressure(pos, mDist, rho, boxSize, cRatio)/(float(N) / rho)
    p = rho*T + pv + ptail
    #print "p is {}, tail is {}, bulk is {}, virial is {}".format(p, ptail, rho*T, pv)
    """print mDist
    print p
    plt.plot(U)"""
    q.put((pos, U, p, acc))
    return pos, U, p, acc



q = Queue()
"""d={}

for i in range(1, 10):
    p = Process(target=mc_LJ, args=(N, dim, i, T, no_smples, rc))
    p.start()

p = np.array([])
acc = np.array([])
U = np.array([])
for i in range(1, 10):
    d["a{}".format(i)] = q.get()

for i in range(1, 10):
    print "Pressure for a = %d is %f" %(i, d["a{}".format(i)][2])
    p = np.append(p, d["a{}".format(i)][2])
    acc = np.append(acc, d["a{}".format(i)][3])
    U = np.append(U, d["a{}".format(i)][1])

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
a1, b, c, d = mc_LJ(10, dim, a, T, no_smples, rc)












