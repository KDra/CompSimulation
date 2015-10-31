# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:54:10 2015

@author: kostas
"""

import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)


def harmonic(r):
    return 0.5 * 2743.0 * (r - 1.1283)**2

def kretzer(r):
    return 258.9 * ((r - 1.1283)/float(r))**2

def morse(r):
    return 258.9 * (1 - np.exp(-2.302 * (r - 1.1283)))**2

