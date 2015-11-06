# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:53:57 2015

@author: kostas
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)
from scipy.optimize import minimize
from numba import jit

ct = {'D0':101.9188,
      'alpha':2.567,
      'kt':328.645606,
      'krt':-211.4672,
      'krr':111.70765,
      'rOHeq':1.0,
      'rHHeq':1.633,
      'eH':0.41,
      'eO':-0.82,
      'sig':3.166,
      'eps':0.1554,
      'rc':14.0,
      'boxSize':35.0}



def accel_inter_molecular(pos, mass, ct):
    """
    Function that returns the acceleration due to intermolecular forces as given by eq 10 above.
    
                K.Drakopoulos@soton.ac.uk, 4/11/2015
    
    Inputs:
        pos   3D matrix of positions: with molecules along axis 0, atoms along axis 1 and x,y,z 
                coordinates on axis 2
        mass  Vector that contains the masses of the atoms
        ct    Dictionary that contains all necessary constants
    Internal:
        n     Stores the number of molecules in the system
        d     Stores the number of atoms in each molecule
        ox    Variable used to check if interaction occurs between two O atoms
        diff  Stores the vector difference between two atoms
        r     Norm of diff, i.e. the distance between the atoms.
        cases Vector used to check if atoms are closer across boundaries
    Outputs:
        a     3D matrix containing the accelerations along all axis (see 'pos')
        
    """
    a = np.zeros_like(pos)
    n = np.shape(pos)[0]
    d = np.shape(pos)[1]
    for i in np.arange(n*d):
        for j in np.arange((int(i/d) + 1) * d, n*d):
            
            ox = 0
            diff = pos[int(i/d), i%d, :] - pos[int(j/d), j%d, :]
            # Contains three cases: closer as is, closer to the left, closer to the right
            cases = np.array([0, -ct['boxSize'], ct['boxSize']])
            for k in np.arange(len(diff)):
                # Add to the directions of interest to see if particles are closer over
                # the periodic boundaries
                index = np.argmin(np.abs(diff[k] + cases))
                diff[k] += cases[index]
            r = np.linalg.norm(diff)
            if r < ct['rc']:
                local_LJ = 0
                if i%3 == 0:
                    ei = ct['eO']
                    ox += 1
                else:
                    ei = ct['eH']
                if j%3 == 0:
                    ej = ct['eO']
                    ox += 1
                else:
                    ej = ct['eH']
                if ox == 2:
                    local_LJ = 24.0 * ct['eps'] * (2.0 * (ct['sig']/r)**12 - (ct['sig']/r)**6) / r**2
                en = (local_LJ - ei * ej/(4.0 * np.pi * r**2)) * diff
                a[int(i/3), i%3, :] += en / mass[i%3]
                a[int(j/3), j%3, :] -= en / mass[j%3]
    return a


def accel_intra_molecular(x, mass, ct):
    """
    Compute the accelerations given the locations for the Lennard-Jones potential.
    
    Parameters
    ----------
    
    x : array of float
        Particle positions
    mass : array of float
        Particle masses
        
    Returns
    -------
    
    a : array of float
        Particle accelerations
    """
    
    D0 = ct['D0']
    alpha = ct['alpha']
    kt = ct['kt']
    krt = ct['krt']
    krr = ct['krr']
    r_OH_eq = ct['rOHeq']
    r_HH_eq = ct['rHHeq']
    
    r_O = x[0, :]
    r_H_1 = x[1, :]
    r_H_2 = x[2, :]
    diff_r_OH_1 = np.linalg.norm(r_O - r_H_1,2)
    diff_r_OH_2 = np.linalg.norm(r_O - r_H_2,2)
    diff_r_HH = np.linalg.norm(r_H_1 - r_H_2,2)
    dr_OH_1 = np.abs(diff_r_OH_1 - r_OH_eq)
    dr_OH_2 = np.abs(diff_r_OH_2 - r_OH_eq)
    dr_HH = np.abs(diff_r_HH - r_HH_eq)
    
    # Derivatives
    d_dr_OH_1_dO = 1.0/(dr_OH_1*diff_r_OH_1)*(diff_r_OH_1 - r_OH_eq)*(r_O - r_H_1)
    d_dr_OH_2_dO = 1.0/(dr_OH_2*diff_r_OH_2)*(diff_r_OH_2 - r_OH_eq)*(r_O - r_H_2)
    d_dr_OH_1_dH1 = -d_dr_OH_1_dO
    d_dr_OH_2_dH2 = -d_dr_OH_2_dO
    d_dr_HH_dH1 = 1.0/(dr_HH*diff_r_HH)*(diff_r_HH - r_HH_eq)*(r_H_1 - r_H_2)
    d_dr_HH_dH2 = -d_dr_HH_dH1
    
    # Forces
    a = np.zeros_like(x)
    # Oxygen
    a[0,:] = (2.0*D0*alpha*((1.0-np.exp(alpha*dr_OH_1))*np.exp(alpha*dr_OH_1)*d_dr_OH_1_dO\
                            +(1.0-np.exp(alpha*dr_OH_2))*np.exp(alpha*dr_OH_2)*d_dr_OH_2_dO)-\
              krt*dr_HH*(d_dr_OH_1_dO+d_dr_OH_2_dO)-\
              krr*(d_dr_OH_1_dO*dr_OH_2+dr_OH_1*d_dr_OH_2_dO))/mass[0]
    # Hydrogen(s)
    a[1,:] = (2.0*D0*alpha*(1.0-np.exp(alpha*dr_OH_1))*np.exp(alpha*dr_OH_1)*d_dr_OH_1_dH1-\
              kt*dr_HH*d_dr_HH_dH1-\
              krt*(d_dr_HH_dH1*(dr_OH_1+dr_OH_2)+dr_HH*d_dr_OH_1_dH1)-\
              krr*d_dr_OH_1_dH1*dr_OH_2)/mass[1]
    a[2,:] = (2.0*D0*alpha*(1.0-np.exp(alpha*dr_OH_2))*np.exp(alpha*dr_OH_2)*d_dr_OH_2_dH2-\
              kt*dr_HH*d_dr_HH_dH2-\
              krt*(d_dr_HH_dH2*(dr_OH_1+dr_OH_2)+dr_HH*d_dr_OH_2_dH2)-\
              krr*d_dr_OH_2_dH2*dr_OH_1)/mass[2]
                
    return a


def accel(pos, mass, ct):
    a_inner = np.zeros_like(pos)
    a_outer = np.zeros_like(pos)
    N = np.shape(pos)[0]
    for i in np.arange(N):
        a_inner[i, :, :] = accel_intra_molecular(pos[i, :, :], mass, ct)
    a_outer = accel_inter_molecular(pos, mass, ct)
    return a_outer + a_inner



def part_reset(pos):
    """
    Function to reset particle positions based on the position of the oxygen atom only
    because the interactions between these atoms affect the L-J potential.
    
                K.Drakopoulos@soton.ac.uk, 4/11/2015
    Input:
        pos   3D vector containing atom coordinates
    Output:
        pos_o 3D vector containing reset atom coordinates
    """
    pos_o = pos.copy()
    over = pos[:, 0, :] > 35.0
    under = pos[:, 0, :] < 0.0
    for i in range(3):
        pos_o[:, i, :][over] -= 35.0
        pos_o[:, i, :][under] += 35.0
    return pos_o


def temperature(vel, mass, boxSize):
    """
    Function to compute the temperatures given the velocities and system size
    
                K.Drakopoulos@soton.ac.uk, 4/11/2015
    
    Inputs:
        vel      3D matrix containing the velocity components for all atoms (see 'sim' function)
        boxSize  Scalar containing the size of the domain
    Output:
        Temperature value
    """
    N = np.shape(vel)[0]
    d = np.shape(vel)[1]
    energy = 0.0
    for i in np.arange(N):
        for j in np.arange(d):
            energy += np.dot(vel[i, j, :], vel[i, j, :]) * mass[j] * 0.5
    return 2 * energy / float(3 * N * d)


def vv(pos, vel, acc, mass, dt, ct):
    pos_new = pos + vel * dt + 0.5 * dt**2 * acc
    pos_new = part_reset(pos_new)
    vel_star = vel + 0.5 * dt * acc
    acc_new = accel(pos_new, mass, ct)
    vel_new = vel_star + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new


def sim(init, mass, dt, steps, ct):
    """
    Returns a 4D matrix of positions as they vary with time, a time vector and a 
    temperature vector, given a time step, the #steps, masses of atoms and a 
    dictionary of constants
        Note:
    The matrix containing the initial positions ('init') should be of shape 8, 3, 3,
    with the first number giving the no of molecules, the second: the number of atoms
    in a molcule and the third: the number of dimensions.
    
               K.Drakopoulos@soton.ac.uk, 4/11/2015
    
    Inputs:
        init        3D matrix of doubles containing the positions of all atoms with 
                    the molecules along axis 0, the atoms in each molecule along axis 1
                    and their coordinates along axis 2.
        mass        A vector of doubles to contain the atom masses
        dt          Scalar of time step size
        steps       Scalar contiaining the number of steps
        ct          A dictionary of constants
    Local:
        cpos        Array of doubles to store the position components of the 
                    'current' velocity-Verlet iteration
        cvel        As above but for velocities
        cacc        As above but for accelerations
    Outputs:
        all_pos     4D array containing the changes in position with each iteration
        time        A vector of time steps
        T           A vector containing the temperature values
    """
    # Trivial checks for reasonable inputs
    #assert np.linalg.norm(init[0, 0, :] - init[0, 1, :]) < np.linalg.norm(init[0, 0, :] - init[1, 0, :]),\
    #"Please check that atoms inside the molecules are ordered on axis 1 and the different molecules on axis 0"
    #assert np.shape(init) == (8, 3, 3),\
    #"Please check the position matrix to be of an acceptable shape (8, 3, 3)."
    assert np.shape(init)[1] == np.shape(mass)[0],\
    "Please check that you have provided a mass term for each atom inside a molecule"
    keys = ['rc', 'rHHeq', 'alpha', 'eO', 'eH', 'krr', 'eps',
            'krt', 'boxSize', 'rOHeq', 'sig', 'kt', 'D0']
    assert len(ct) == 13, "Dictionary is too short. Check if any terms have been omitted"
    assert all(key in keys for key in ct),\
    "Please check that all elements in the dictionary have appropriate key names"
    # Begin computations
    cpos = init.copy()
    cpos = part_reset(cpos)
    all_pos = np.zeros((steps+1, cpos.shape[0], cpos.shape[1], cpos.shape[2]))
    all_pos[0, :, :, :] = cpos
    cvel = np.zeros_like(cpos)
    cacc = np.zeros_like(cpos)
    time = np.arange(0.0, steps*dt, dt)
    time = np.append(time, steps*dt)
    T = np.zeros_like(time)
    T[0] = temperature(cvel, mass, ct['boxSize'])
    for i in np.arange(1, steps+1):
        cpos, cvel, cacc = vv(cpos, cvel, cacc, mass, dt, ct)
        all_pos[i, :, :] = cpos[:]
        T[i] = temperature(cvel, mass, ct['boxSize'])
        #print "pos of first two are \n{}\n{}\n Temperature is {}\n".format(cpos[0,:], cpos[1, :], T[i])
    return all_pos, time, T


mass = np.array([15.999, 1.008, 1.008])
rd = np.random.RandomState(102)
pos = np.zeros((8, 3, 3))
pos[:, 0, :] = (rd.rand(8,3))*35.0
pos[:, 1, :] = pos[:, 0, :] + np.array([0.8, 0.6, 0.0]) + (np.random.rand(8,3)-0.5)*1e-6
pos[:, 2, :] = pos[:, 0, :] + np.array([-0.8, 0.6, 0.0]) + (np.random.rand(8,3)-0.5)*1e-6

dt = 0.001
steps = 5000
#all_pos, time, T = sim(pos, mass, dt, steps, ct)
#plt.plot(time, T)
"""
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
for j in np.arange(3):
    for i in np.arange(0, 20, steps):
        ax1.scatter3D(all_pos[i, j, 0, 0], all_pos[i, j, 0, 1], all_pos[i, j, 0, 2], c='r', marker='o', s=200)
        ax1.scatter3D(all_pos[i, j, 1:, 0], all_pos[i, j, 1:, 1], all_pos[i, j, 1:, 2], c='b', marker='o', s=100)
plt.show()"""
#ao,ai = accel(pos, mass, ct)



#@jit
def wmorse(r, ct):
    return ct['D0'] * (1 - np.exp(-ct['alpha'] * (r - ct['rOHeq'])))**2

#@jit
def mwater(r, ct):
    """
    The vector r is considered to be built as following:
        r[0] = r_OH1
        r[1] = r_OH2
        r[2] = r_HH
    """
    assert len(r) == 3, "Please check input vector shape to be of length 3"
    
    droh1 = r[0] - ct['rOHeq']
    droh2 = r[1] - ct['rOHeq']
    drhh = r[2] - ct['rHHeq']
    Vpart = 0.5 * ct['kt'] * drhh**2 + ct['krt'] * drhh * (droh1 + droh2) + ct['krr'] * droh1 * droh2
    return np.sum(wmorse(r[:2], ct)) + Vpart

#@jit 
def inter(pos, ct):
    ene = 0
    n = np.shape(pos)[0]
    d = np.shape(pos)[1]
    local_LJ = 0.0
    for i in np.arange(n*d):
        for j in np.arange((int(i/d) + 1) * d, n*d):
            ox = 0
            diff = pos[int(i/d), i%d, :] - pos[int(j/d), j%d, :]
            # Contains three cases: closer as is, closer to the left, closer to the right
            cases = np.array([0, -ct['boxSize'], ct['boxSize']])
            for k in np.arange(len(diff)):
                # Add to the directions of interest to see if particles are closer over
                # the periodic boundaries
                index = np.argmin(np.abs(diff[k] + cases))
                diff[k] += cases[index]
            r = np.linalg.norm(diff)
            #print r
            #if r < ct['rc']:
            if i%3 == 0:
                ei = ct['eO']
                ox += 1
            else:
                ei = ct['eH']
            if j%3 == 0:
                ej = ct['eO']
                ox += 1
            else:
                ej = ct['eH']
            if ox == 2:
                local_LJ = 4 * ct['eps'] * ((ct['sig']/r)**(12) - (ct['sig']/r)**(6))
            ene += (-ei*ej/4/np.pi/r + local_LJ) / mass[i%3]
    return ene


#@jit
def energy(xpos, ct):
    lpos = np.zeros((8,3,3))
    lpos[:, 0, :] = xpos.reshape(8,3)
    lpos[:, 1, :] = lpos[:, 0, :] + np.array([0.8, 0.6, 0.0])
    lpos[:, 2, :] = lpos[:, 0, :] + np.array([-0.8, 0.6, 0.0])
    
    ene = 0
    n = np.shape(lpos)[0]
    d = np.shape(lpos)[1]
    local_LJ = 0.0
    for i in np.arange(n*d):
        for j in np.arange((int(i/d) + 1) * d, n*d):
            ox = 0.0
            diff = lpos[int(i/d), i%d, :] - lpos[int(j/d), j%d, :]
            # Contains three cases: closer as is, closer to the left, closer to the right
            cases = np.array([0, -ct['boxSize'], ct['boxSize']])
            for k in np.arange(len(diff)):
                # Add to the directions of interest to see if particles are closer over
                # the periodic boundaries
                index = np.argmin(np.abs(diff[k] + cases))
                diff[k] += cases[index]
            r = np.linalg.norm(diff)
            #print r
            if r < ct['rc']:
                if i%3 == 0:
                    ei = ct['eO']
                    ox += 1
                else:
                    ei = ct['eH']
                if j%3 == 0:
                    ej = ct['eO']
                    ox += 1
                else:
                    ej = ct['eH']
                if ox == 2:
                    local_LJ = 4 * ct['eps'] * ((ct['sig']/r)**(12) - (ct['sig']/r)**(6))
                ene += (-ei*ej/4.0/np.pi/r + local_LJ)
    return ene


res = minimize(energy, pos[:, 0, :], args=ct, tol=1e-6)
print res.success
optpos = np.zeros_like(pos)
optpos[:,0,:] = np.array(res.x).reshape(8,3)
optpos[:, 1, :] = optpos[:, 0, :] + np.array([0.8, 0.6, 0.0])
optpos[:, 2, :] = optpos[:, 0, :] + np.array([-0.8, 0.6, 0.0])