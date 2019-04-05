#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:13:35 2019

@author: Sonja Mathias
"""
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=0.01):

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]

    # start with 1 Euler forward step
    t = t + dt
    y = y0 + dt*fun(t,y)

    ts.append(t)
    ys.append(y)

    while t + dt <= tf:
        y = y + dt/2.0*(3*fun(t,y)-fun(ts[-1], ys[-1]))
        t = t + dt

        ts.append(t)
        ys.append(y)

    ts = np.hstack(ts)
    ys = np.hstack(ys)

    return OdeResult(t=ts, y=ys)

if __name__ == "__main__":

    # stability region for Euler forward for this problem is h<2/50=0.04
    @np.vectorize
    def func(t,y):
        return -50*y

    t_span = (0,1)
    y0 = 1

    sol = solve_ivp(func, t_span, y0 )

    plt.figure()
    plt.plot(sol.t, sol.y)



