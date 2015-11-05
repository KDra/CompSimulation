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

rd = np.random.RandomState(100)
pos = np.zeros((8, 3, 3))
pos[:, 0, :] = (rd.rand(8,3) - 0.5) * 45.0
pos[:, 1, :] = pos[:, 0, :] + np.array([0.8, 0.6, 0.0])
pos[:, 2, :] = pos[:, 0, :] + np.array([-0.8, 0.6, 0.0])

over = pos[:, 0, :] > 17.5
under = pos[:, 0, :] < -17.5
post = pos.copy()

for i in range(3):
    pos[:, i, :][over] -= 35
    pos[:, i, :][under] += 35