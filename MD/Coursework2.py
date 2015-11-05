# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:53:57 2015

@author: kostas
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)
from scipy.optimize import minimize

ct = {'D0':101.9188,
      'alpha':2.567,
      'kt':328.645606,
      'krt':-211.4672,
      'krr':11.70765,
      'rOHeq':1.0,
      'rHHeq':1.633,
      'eH':0.41,
      'eO':-0.82,
      'sig':3.166,
      'eps':0.1554,
      'rc':14.0,
      'boxSize':35.0}

rd = np.random.RandomState(100)
pos = np.zeros((8, 3, 3))
pos[:, 0, :] = (rd.rand(8,3)) * 45.0
pos[:, 1, :] = pos[:, 0, :] + np.array([0.8, 0.6, 0.0])
pos[:, 2, :] = pos[:, 0, :] + np.array([-0.8, 0.6, 0.0])

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
    
    #ndim, N = x.shape
    #assert(N==3)
    
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
    F = np.zeros_like(x)
    # Oxygen
    F[:,0] = (2.0*D0*alpha*((1.0-np.exp(alpha*dr_OH_1))*np.exp(alpha*dr_OH_1)*d_dr_OH_1_dO\
                            +(1.0-np.exp(alpha*dr_OH_2))*np.exp(alpha*dr_OH_2)*d_dr_OH_2_dO)-\
              krt*dr_HH*(d_dr_OH_1_dO+d_dr_OH_2_dO)-\
              krr*(d_dr_OH_1_dO*dr_OH_2+dr_OH_1*d_dr_OH_2_dO))/mass[0]
    # Hydrogen(s)
    F[:,1] = (2.0*D0*alpha*(1.0-np.exp(alpha*dr_OH_1))*np.exp(alpha*dr_OH_1)*d_dr_OH_1_dH1-\
              kt*dr_HH*d_dr_HH_dH1-\
              krt*(d_dr_HH_dH1*(dr_OH_1+dr_OH_2)+dr_HH*d_dr_OH_1_dH1)-\
              krr*d_dr_OH_1_dH1*dr_OH_2)/mass[1]
    F[:,2] = (2.0*D0*alpha*(1.0-np.exp(alpha*dr_OH_2))*np.exp(alpha*dr_OH_2)*d_dr_OH_2_dH2-\
              kt*dr_HH*d_dr_HH_dH2-\
              krt*(d_dr_HH_dH2*(dr_OH_1+dr_OH_2)+dr_HH*d_dr_OH_2_dH2)-\
              krr*d_dr_OH_2_dH2*dr_OH_1)/mass[2]
                
    return F

def accel(pos, mass, ct):
    a_inner = np.zeros_like(pos)
    a_outer = np.zeros_like(pos)
    N = np.shape(pos)[0]
    for i in np.arange(N):
        a_inner[i, :, :] = accel_intra_molecular(pos[i, :, :], mass, ct)
    a_outer = accel_inter_molecular(pos, mass, ct)
    return a_outer , a_inner