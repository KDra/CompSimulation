# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:29:45 2015

@author: kostas
"""
from __future__ import division
from math import atan2, pi, cos, sin
from numba import jit
import numpy as np
from matplotlib import pyplot, animation
from JSAnimation import IPython_display

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)

import scipy
from scipy.optimize import minimize


class Agent(object):
    @jit
    def __init__(self, location, velocity, C=1, A=5, S=0.1):
        self.loc = np.array(location)
        self.v = np.array(velocity)
        self.C = C
        self.A = A
        self.S = S
    @jit
    def step(self, dt): 
        self.loc += self.v*dt
    @jit
    def steer(self, neighbours):
        N = len(neighbours)
        if N == 0:
            return
        min_sep = 100.0
        # Initialise vectors
        diff_avg_loc = np.zeros_like(self.loc)
        min_sep_direction = diff_avg_loc.copy()
        avg_v = diff_avg_loc.copy()
        for ag in neighbours:
            diff_avg_loc += ag.loc - self.loc
            avg_v += ag.v
            sep = np.linalg.norm(ag.loc - self.loc)
            if sep < min_sep:
                min_sep = sep
                min_sep_direction = ag.loc - self.loc
        avg_v /= N
        diff_avg_loc /= N
        v = np.linalg.norm(self.v)
        # Compute theta values
        th = atan2(self.v[1], self.v[0])
        th_z = atan2(diff_avg_loc[1], diff_avg_loc[0])
        th_V = atan2(avg_v[1], avg_v[0])
        th_min_z = atan2(min_sep_direction[1], min_sep_direction[0])
        minz = np.linalg.norm(min_sep_direction)
        
        C = self.C
        A = self.A
        S = self.S
        
        utility = lambda theta: -C * cos(theta - th_z)+\
                                A * cos(theta - th_V)-\
                                S * cos(theta - th_min_z) / minz**2
        res = minimize(utility, th, bounds=[[-pi, pi]])
        th_new = res.x[0]
        self.v = np.array([v * cos(th_new), v * sin(th_new)])

class Flock(object):
    @jit
    def __init__(self, locations, velocities, rl=1):
        self.rl = rl
        self.agents = []
        for loc, vel in zip(locations, velocities):
            self.agents.append(Agent(loc, vel))
    @jit
    def step(self, dt):
        for i, agent in enumerate(self.agents):
            index = np.logical_and(np.linalg.norm(self.locations(i) - np.array([agent.loc]).T,
                                   axis=0) < self.rl, np.logical_not(np.array(self.agents) == agent))
            agent.steer(np.array(self.agents)[index])
        for agent in self.agents:
            agent.step(dt)
    @jit
    def locations(self, i=None):
        x = np.zeros(len(self.agents))
        y = x.copy()
        for j, agent in enumerate(self.agents):
            if not i == j:
                x[j], y[j] = agent.loc
        return np.vstack((x, y))
    @jit
    def velocities(self):
        vx = np.zeros(len(self.agents))
        vy = vx.copy()
        for i, agent in enumerate(self.agents):
            vx[i], vy[i] = agent.v
        return np.vstack((vx, vy))
    @jit
    def average_location(self):
        return np.mean(self.locations(), axis=1)
    @jit
    def average_velocity(self):
        return np.mean(self.velocities(), axis=1)
    @jit
    def average_width(self):
        locs = self.locations()
        average_loc = self.average_location()
        distances = np.linalg.norm(locs-average_loc[:,np.newaxis], axis=0)
        return np.mean(distances)

def flock_animation(flock, dt, frames=10, xlim=(-0.1, 5), ylim=(-0.1, 5)):
    # First evolve
    
    locations = [flock.locations()]
    ave_width = [flock.average_width()]
    ave_loc = [flock.average_location()]
    ave_vel = [flock.average_velocity()]
    times = np.arange(0.0, frames*dt, dt)
    for i in range(frames):
        flock.step(dt)
        locations.append(flock.locations())
        ave_width.append(flock.average_width())
        ave_loc.append(flock.average_location())
        ave_vel.append(flock.average_velocity())
        
    max_width = max(ave_width)
    min_width = min(ave_width)
    d_width = max_width - min_width
    
    fig = pyplot.figure()
    ax1 = fig.add_subplot(131)
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    points, = ax1.plot([], [], 'ro')
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax2 = fig.add_subplot(132)
    width, = ax2.plot([], [], 'b-')
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("Average width of flock")
    ax2.set_xlim(0.0, dt*frames)
    ax2.set_ylim(min_width-0.1*d_width, max_width+0.1*d_width)
    ax3 = fig.add_subplot(133)
    loc, = ax3.plot([], [], 'bo', label='Location')
    vel, = ax3.plot([], [], 'r*', label='Heading')
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$y$")
    ax3.set_xlim(xlim[0], xlim[1])
    ax3.set_ylim(ylim[0], ylim[1])
    ax3.legend()
    fig.tight_layout()

    def init():
        points.set_data([], [])
        width.set_data([], [])
        loc.set_data([], [])
        vel.set_data([], [])
        return (points, width, loc, vel)

    def animate(i):
        points.set_data(locations[i][0,:], locations[i][1,:])
        width.set_data(times[:i+1], ave_width[:i+1])
        loc.set_data([ave_loc[i][0]], [ave_loc[i][1]])
        vel.set_data([ave_vel[i][0]], [ave_vel[i][1]])
        return (points, width, loc, vel)

    return animation.FuncAnimation(fig, animate, init_func=init, interval=100, frames=frames, blit=True)

if __name__ == '__main__':
    locations = np.array([[0.0, 0.0], [0.25, 0.25], [0.125, 0.125]])
    velocities = np.ones_like(locations)
    flock = Flock(locations, velocities)
    flock_animation(flock, 0.05, 200, xlim=(-20.0,20.0), ylim=(-20.0,20))