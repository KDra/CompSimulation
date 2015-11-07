# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:55:15 2015

@author: kostas
"""

def conway_iteration(grid):
    """
    Take one iteration of Conway's game of life.
    
    Parameters
    ----------
    
    grid : array
        (N+2) x (N+2) numpy array representing the grid (1: live, 0: dead)
    
    """
    n, d = np.shape(grid)
    print n, d
    # Code to go here
    for i in np.arange(n):
        for j in np.arange(d):
            #try:
            mgrid = grid[i-1:i+2, j-1:j+2]
            print i, j
            print mgrid
    return grid