#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:16:04 2021

@author: Sonja Mathias
"""

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
from scipy.sparse.linalg import gmres
import scipy as scpi
import copy


import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')

def solve_ivp(fun, t_span, y0, t_eval=None, dt=None, n_newton=2, jacobian=None,
              out='', write_to_file=False):
    # do regular fixed time stepping
    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []

    while t < tf:

        y = copy.deepcopy(y)

        # do Newton iterations
        y_next = copy.deepcopy(y) # initialize with current y
        for j in np.arange(n_newton):
            if jacobian is not None:
                A = jacobian(y_next)
            else:
                print('Error: No jacobian provided!')
            J = np.eye(A.shape[0]) - dt*A
            F_curly = y_next - y - dt* fun(t, y_next)

            #solve linear system J*dy = F_curly for dy
            #dy, exitCode = gmres(J, -F_curly)
            dy = scpi.linalg.solve(J, -F_curly)
            #print(exitCode)
            y_next = y_next + dy

        y = copy.deepcopy(y_next)
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)


    ts = np.hstack(ts)
    ys = np.vstack(ys).T
    dts = np.hstack(dts)


    if write_to_file:
        with open('time_points'+out+'.txt', 'ab') as f:
            np.savetxt(f, ts)
        with open('step_sizes'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts)

    return OdeResult(t=ts, y=ys)

if __name__ == "__main__":

    # stability region for Euler forward for this problem is h<2/50=0.04
    #@np.vectorize
    def func(t, y):
        return -50*np.eye(len(y))@y
    def jac(y):
        return -50*np.eye(len(y))

#    t_span = (0,1)
#    y0 = np.array([1,1])
#
#    sol = solve_ivp(func, t_span, y0 )
#
#    plt.figure()
#    plt.plot(sol.t, sol.y)

    t_eval = np.linspace(0,3,10)
    y0 = np.array([1.0, 2.0])
    #y0 = np.array([0.5, 0.7, 1.0, 3.0])
    #y0 = np.array([0.0, 0.0, 0.0])

    try:
        os.remove('step_sizes.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('time_points.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, dt=0.1, n_newton = 2,
                     write_to_file=True, jacobian=jac)
    #plt.plot(sol2.t, sol2.y.T)
    plt.plot(sol2.t, sol2.y.T, '*')
    plt.xlabel('t')
    plt.ylabel('y')

    plt.figure()
    dt  = np.loadtxt('step_sizes.txt')
    plt.plot(np.cumsum(dt), dt)
    plt.xlabel('time')
    plt.ylabel('Global step size')
