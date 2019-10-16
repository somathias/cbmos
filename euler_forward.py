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


def solve_ivp(fun, t_span, y0, t_eval=None, dt=None, eps=0.001, eta=0.01):


    t0, tf = float(t_span[0]), float(t_span[-1])

#    if t_eval is not None:
#        assert t0 == t_eval[0]
#        assert tf == t_eval[-1]
#
#        # these variables are only needed if t_eval is not None
#        i = 1
#        tp = t0
#        yp = y0

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []

    adaptive_dt = True if dt is None else False

    while t < tf :

        if adaptive_dt:
            # choose time step adaptively
            F = fun(t,y)
            norm_AF = np.linalg.norm(1/eta*(fun(t, y + eta * F) - F), np.inf)
            #print('AF'+ str(AF))
            dt = np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t
            #print('dt' + str(dt))

            y = y + dt*F
        else:
            y = y + dt*fun(t,y)

        t = t + dt

#        if t_eval is not None:
#            while i < len(t_eval) and t >= t_eval[i]:
#                if t == t_eval[i]:
#                    ts.append(t)
#                    ys.append(y)
#                    i += 1
#                elif t > t_eval[i]:
#                    yint = yp + (t_eval[i]-tp)*(y-yp)/(t-tp)
#                    ts.append(t_eval[i])
#                    ys.append(yint)
#                    i += 1
#            tp = t
#            yp = y
#        else:
        ts.append(t)
        ys.append(y)
        dts.append(dt)
#        print('len(ts) '+ str(len(ts)))
#        print('len(ys) '+ str(len(ys)))
#        print('len(dts) '+ str(len(dts)))

    ts = np.hstack(ts)
#    print('len(ts) '+ str(len(ts)))
    ys = np.vstack(ys).T
#    print('len(ys.T) '+ str(len(ys.T)))
    dts = np.hstack(dts)
#    print('len(dts) '+ str(len(dts)))

    with open('time_points.txt', 'ab') as f:
        np.savetxt(f, ts[1:])
    with open('step_sizes.txt', 'ab') as f:
        np.savetxt(f, dts)

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

    sol = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None, dt=0.01, eps=0.001 )
    plt.figure()
    plt.plot(sol.t, sol.y.T)
    plt.plot(sol.t, sol.y.T, '.', color='black')

    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None, eps=0.001 )
    plt.plot(sol2.t, sol2.y.T)
    plt.plot(sol2.t, sol2.y.T, '*', color='red')



