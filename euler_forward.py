#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:13:35 2019

@author: Sonja Mathias
"""
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
import matplotlib.pyplot as plt
import copy

plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=None, eps=0.001, eta=0.01, out='', local_adaptivity=False, p=0.1):


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
    dts_local = []

    adaptive_dt = True if dt is None else False

    while t < tf :

        y = copy.deepcopy(y)

        if len(y0) > 1 and local_adaptivity:
            # choose time step adaptively locally (if we have a system of eqs)
            F = fun(t,y)
            AF = 1/eta*(fun(t, y + eta * F) - F)
            # sort the indices such that AF(inds) is decreasing
            inds = np.argsort(-abs(AF))
            n_K0 = int(np.ceil(p*len(y0)))
            eta_0 = np.max(np.abs(AF[inds[:n_K0]]))
            eta_1 = np.max(np.abs(AF[inds[n_K0:]]))
            m = int(np.floor(eta_0/eta_1))

            dt_0 = np.sqrt(2*eps*eta_1)/eta_0
            dt = m*dt_0

            # advance equations in K\K0 with large timestep
            y[inds[n_K0:]] = y[inds[n_K0:]] + dt*F[inds[n_K0:]]
            # advance equations in K0 with several small timesteps
            for i in range(m):
                F = fun(t+i*dt_0, y)
                y[inds[:n_K0]] = y[inds[:n_K0]] + dt_0*F[inds[:n_K0]]

                dts_local.append(dt_0)


        elif adaptive_dt:
            # choose time step adaptively
            F = fun(t,y)
            norm_AF = np.linalg.norm(1/eta*(fun(t, y + eta * F) - F), np.inf)
            dt = np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t

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

    ts = np.hstack(ts)
    ys = np.vstack(ys).T
    dts = np.hstack(dts)
    dts_local = np.hstack(dts_local)


    with open('time_points'+out+'.txt', 'ab') as f:
        np.savetxt(f, ts[1:])
    with open('step_sizes'+out+'.txt', 'ab') as f:
        np.savetxt(f, dts)
    with open('step_sizes_local'+out+'.txt', 'ab') as f:
        np.savetxt(f, dts_local)

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

    t_eval = np.linspace(0,0.5,10)
    y0 = np.array([0.5, 0.7, 3.0])

#    sol = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None, dt=0.01, eps=0.001 )
#    plt.figure()
#    plt.plot(sol.t, sol.y.T)
#    plt.plot(sol.t, sol.y.T, '.', color='black')

    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None, eps=0.001, local_adaptivity=True)
    plt.plot(sol2.t, sol2.y.T)
    plt.plot(sol2.t, sol2.y.T, '*', color='red')





