# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:43:41 2015

@author: kostas
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
from numba import jit
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)


#@jit
def dLJ(pos, i, j, rc, boxSize):
    diff = pos[i, :] - pos[j, :]
    # Contains three cases: closer as is, closer to the left, closer to the right
    cases = np.array([0, -0.5, 0.5])
    for n in range(pos.shape[1]):
        # Add to the directions of interest to see if particles are closer over
        # the periodic boundaries
        index = np.argmin(np.abs(diff[n] + cases))
        diff[n] += cases[index]
    r = np.linalg.norm(diff)*boxSize
    if r < rc:
        return 24 * (2 * r**(-14) - r**(-8))
    else:
        return 0.0


#@jit
def accel(pos, rc, boxSize):
    N = pos.shape[1]
    acc_new = np.zeros_like(pos)
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            dLJ_local = dLJ(pos, i, j, rc, boxSize)
            pos_diff = pos[i, :] - pos[j, :]
            acc_new[i, :] += pos_diff * dLJ_local
            acc_new[j, :] -= pos_diff * dLJ_local
    return acc_new


#@jit
def vv(pos, vel, acc, dt, rc, boxSize):
    N, dim = np.shape(pos)
    acc_new = np.zeros_like(acc)
    pos_new = pos + vel * dt + 0.5 * dt**2 * acc
    pos_new = part_reset(pos_new)
    vel_star = vel + 0.5 * dt * acc
    acc_new = accel(pos_new, rc, boxSize)
    vel_new = vel_star + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new

def part_reset(pos):
    pos[pos > 0.5] -= 1.0
    pos[pos < -0.5] += 1.0
    return pos

def temperature(vel, boxSize):
    N = vel.shape[0]
    energy = np.sum(np.einsum('ij,ji->i', vel, vel.T))
    return boxSize**2 * energy / float(3 * N)

def sim(filename, boxSize, rc, dt, steps):
    pos_orig = np.loadtxt(filename, skiprows=1)
    cpos = pos_orig / float(boxSize) - 0.5
    cpos -= np.mean(cpos, axis=0)
    cpos = part_reset(cpos)
    all_pos = np.zeros((steps+1, cpos.shape[0], cpos.shape[1]))
    all_pos[0, :, :] = cpos
    cvel = np.zeros_like(cpos)
    cacc = np.zeros_like(cpos)
    time = np.arange(0.0, steps*dt, dt)
    time = np.append(time, steps*dt)
    T = np.zeros_like(time)
    T[0] = temperature(cvel, boxSize)
    for i in np.arange(1, steps+1):
        cpos, cvel, cacc = vv(cpos, cvel, cacc, dt, rc, boxSize)
        all_pos[i, :, :] = cpos[:]
        T[i] = temperature(cvel, boxSize)
        #print "pos of first two are \n{}\n{}\n Temperature is {}\n".format(cpos[0,:], cpos[1, :], T[i])
    return all_pos, time, T


filename = "./input.dat"
boxSize = 6.1984
dt = 0.032
steps = 100
rc = 2.5
pos, t, T = sim(filename, boxSize, rc, dt, steps)


