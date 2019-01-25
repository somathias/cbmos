#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:35:37 2019

Relaxation experiment: two cells initially placed at overlap of 

@author: Sonja Mathias
"""
import numpy as np
import scipy.integrate as scpi
import matplotlib.pyplot as plt

import cbmos_serial as cbmos
import force_functions as ff

plt.style.use('seaborn')


cbm_solver = cbmos.CBMSolver(ff.linear_spring, scpi.solve_ivp)

T = np.linspace(0, 10, num=1000)

x1=np.array([0., 0., 0.])
x2=np.array([0.3, 0., 0.])

y0 = np.array([x1, x2]).reshape(-1)

sol = cbm_solver.simulate(T, y0, {'s': 1.0, 'mu': 1.0}, {'method': 'RK45'})

print(sol.y)

plt.figure()
plt.plot(T, np.abs(sol.y[0,:] - sol.y[3,:]))
