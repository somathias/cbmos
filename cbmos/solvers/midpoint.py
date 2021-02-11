#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:13:35 2019

@author: Sonja Mathias
"""
import numpy as _np
from scipy.integrate._ivp.ivp import OdeResult
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=0.01, hpc_backend=_np):

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = hpc_backend.asarray(y0)

    ts = [t]
    ys = [y0]

    while t < tf:
        y = y + dt*fun(t+dt/2.0, y + dt/2.0*fun(t,y))
        t = t + dt

        ts.append(t)
        if hpc_backend.__name__ == "cupy":
            ys.append(hpc_backend.asnumpy(y))
        else:
            ys.append(hpc_backend.asarray(y))

    ts = _np.hstack(ts)
    ys = _np.vstack(ys).T

    return OdeResult(t=ts, y=ys)

if __name__ == "__main__":

    # stability region for Euler forward for this problem is h<2/50=0.04
    @_np.vectorize
    def func(t,y):
        return -50*y

    t_span = (0,1)
    y0 = 1

    sol = solve_ivp(func, t_span, y0 )

    plt.figure()
    plt.plot(sol.t, sol.y.T)



