# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:55:15 2015

@author: kostas
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot, animation
from JSAnimation import IPython_display

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)


def conway_iteration(grid):
    """
    Take one iteration of Conway's game of life.
    
    Parameters
    ----------
    
    grid : array
        (N+2) x (N+2) numpy array representing the grid (1: live, 0: dead)
    
    """
    n, d = np.shape(grid)
    ext_grid = grid.copy()
    grid_out = grid.copy()
    ext_grid = np.hstack((ext_grid[:,[-1]], ext_grid))
    ext_grid = np.hstack((ext_grid, ext_grid[:,[1]]))
    ext_grid = np.vstack((ext_grid[[-1],:], ext_grid))
    ext_grid = np.vstack((ext_grid, ext_grid[[1],:]))
    # Code to go here
    for i in np.arange(n):
        for j in np.arange(d):
            mgrid = ext_grid[i:i+3, j:j+3].copy()
            mgrid[1,1] = 0
            msum = np.sum(mgrid)
            if grid[i, j] == 1:
                if msum < 2 or msum >= 4:
                    grid_out[i,j] = 0
            elif grid[i, j] == 0 and msum == 3:
                grid_out[i,j] = 1
    grid=grid_out
    return grid

mvgrid=(np.round(np.random.rand(6, 6))).astype(int)
m_static = np.array([[0, 0, 1],[0, 0, 0], [1, 0, 0]])
# Try the loaf - this is static

grid_loaf = np.array([[0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,1,1,0,0,0],
                         [0,0,1,0,0,1,0,0],
                         [0,0,0,1,0,1,0,0],
                         [0,0,0,0,1,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0]])

fig = pyplot.figure()
im = pyplot.imshow(grid_loaf[1:-1,1:-1], cmap=pyplot.get_cmap('gray'))

def init():
    im.set_array(grid_loaf[1:-1,1:-1])
    return im,

def animate(i):
    conway_iteration(grid_loaf)
    im.set_array(grid_loaf[1:-1,1:-1])
    return im,

# This will only work if you have JSAnimation installed

animation.FuncAnimation(fig, animate, init_func=init, interval=50, frames=10, blit=True)