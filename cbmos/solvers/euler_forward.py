#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=0.01):


    t0, tf = float(t_span[0]), float(t_span[-1])

    if t_eval is not None:
        assert t0 == t_eval[0]
        assert tf == t_eval[-1]

        # these variables are only needed if t_eval is not None
        i = 1
        tp = t0
        yp = y0

    t = t0
    y = y0

    ts = [t]
    ys = [y]

    while t < tf :
        y = y + dt*fun(t,y)
        t = t + dt

        if t_eval is not None:
            while i < len(t_eval) and t >= t_eval[i]:
                if t == t_eval[i]:
                    ts.append(t)
                    ys.append(y)
                    i += 1
                elif t > t_eval[i]:
                    yint = yp + (t_eval[i]-tp)*(y-yp)/(t-tp)
                    ts.append(t_eval[i])
                    ys.append(yint)
                    i += 1
            tp = t
            yp = y
        else:
            ts.append(t)
            ys.append(y)

    ts = np.hstack(ts)
    ys = np.vstack(ys).T

    return OdeResult(t=ts, y=ys)

if __name__ == "__main__":

    # stability region for Euler forward for this problem is h<2/50=0.04
    @np.vectorize
    def func(t,y):
        return -50*y

#    t_span = (0,1)
#    y0 = np.array([1,1])
#
#    sol = solve_ivp(func, t_span, y0 )
#
#    plt.figure()
#    plt.plot(sol.t, sol.y)

    t_eval = np.linspace(0,1,10)
    y0 = np.array([1])

    sol = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval)



