#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:13:35 2019

@author: Sonja Mathias
"""
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
import copy


import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=None, eps=0.01, eta=0.001, out='', local_adaptivity=False, m0=2, m1=2, write_to_file=False):


    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]

    dts = []
    dts_local = []
    levels = []
    n_eq_per_level = []



    adaptive_dt = True if dt is None else False
    while t < tf:

        y = copy.deepcopy(y)

        if len(y0) > 1 and local_adaptivity:
            # choose time step adaptively locally (if we have a system of eqs)
            F = fun(t, y)
            af = 1/eta*(fun(t, y + eta * F) - F)

            if write_to_file:
                with open('AFs'+out+'.txt', 'ab') as f:
                    np.savetxt(f, np.abs(af).reshape((1, -1)))

            # sort the indices such that abs(AF(inds)) is decreasing
            inds = np.argsort(-abs(af))
            # find largest and smallest eta_k
            Xi_0 = abs(af[inds[0]])

            Xi_min = abs(af[inds[-1]])
            dt_max = np.sqrt(2*eps/Xi_min) if Xi_min > 0.0 else tf - t

            # calculate time steps for different levels
#            dt_0 = np.minimum(np.sqrt(2*eps / (m0*m1*Xi_0)), 0.04) if Xi_0 > 0.0 else tf - t
#            dt_1 = np.minimum(m0*dt_0, 0.04)
#            dt_2 = np.minimum(m1*dt_1, 0.04)
            dt_0 = np.sqrt(2*eps / (m0*m1*Xi_0)) if Xi_0 > 0.0 else tf - t
            dt_1 = m0*dt_0
            dt_2 = m1*dt_1

            # calculate corresponding maximum eta for each level
            #Xi_1 = 2*eps/(m1*dt_1**2)
            #Xi_2 = 2*eps/(dt_2**2)
            Xi_1 = Xi_0/m0
            Xi_2 = Xi_1/m1

            # find corresponding indices
            min_ind_1 = len(y0) - np.searchsorted(abs(af[inds])[::-1], Xi_1, side='right')
            min_ind_2 = len(y0) - np.searchsorted(abs(af[inds])[::-1], Xi_2, side='right')

            n_eqs = np.array([min_ind_1, min_ind_2 - min_ind_1, len(y0)- min_ind_2])
            n_eq_per_level.append(n_eqs)
            levels.append(np.sum(n_eqs > 0))

#            min_ind_2 = np.argmax(np.sqrt(2*eps/abs(af[inds])) >= dt_2)
#            min_ind_1 = np.argmax(np.sqrt(2*eps/abs(af[inds])/m1) >= dt_1)

            if (min_ind_1 < min_ind_2) and (min_ind_2 < len(y0)):
                # three levels
                for i in range(m1):

                    for j in range(m0):
                        y[inds[:min_ind_1]] = y[inds[:min_ind_1]] + dt_0*F[inds[:min_ind_1]]
                        F = fun(t, y)
                        dts_local.append(dt_0)

                    y[inds[min_ind_1:min_ind_2]] = y[inds[min_ind_1:min_ind_2]] + dt_1*F[inds[min_ind_1:min_ind_2]]
                    F = fun(t, y)

                y[inds[min_ind_2:]] = y[inds[min_ind_2:]] + dt_2*F[inds[min_ind_2:]]

                dt = dt_2
            elif (min_ind_1 < min_ind_2 and min_ind_2 == len(y0)) or (min_ind_1 == min_ind_2 and min_ind_2 < len(y0)):
                # two levels, always fall back on dt_1
                for j in range(m0):
                    y[inds[:min_ind_1]] = y[inds[:min_ind_1]] + dt_0*F[inds[:min_ind_1]]
                    F = fun(t, y)
                    dts_local.append(dt_0)

                y[inds[min_ind_1:]] = y[inds[min_ind_1:]] + dt_1*F[inds[min_ind_1:]]
                F = fun(t, y)

                dt = dt_1

            else:
                # single level
                y = y + dt_0*F
                dt = dt_0
                dts_local.append(dt_0)

        elif adaptive_dt:
            # choose time step adaptively
            F = fun(t,y)
            AF = 1/eta*(fun(t, y + eta * F) - F)

            #print(np.abs(AF))
            if write_to_file:
                with open('AFs'+out+'.txt', 'ab') as f:
                    np.savetxt(f, np.abs(AF).reshape((1, -1)))

            norm_AF = np.linalg.norm(AF, np.inf)
            #print('AF'+ str(AF))
            dt = np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t

            y = y + dt*F

        else:
            y = y + dt*fun(t, y)

        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

    ts = np.hstack(ts)
    ys = np.vstack(ys).T
    dts = np.hstack(dts)
    if local_adaptivity :
        dts_local = np.hstack(dts_local)
        n_eq_per_level = np.vstack(n_eq_per_level).T


    # Note that these files need to be appended in case there are cell events happening in between different solves.
    if write_to_file :
        with open('time_points'+out+'.txt', 'ab') as f:
            np.savetxt(f, ts[1:])
        with open('step_sizes'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts)
        if local_adaptivity :
            with open('step_sizes_local'+out+'.txt', 'ab') as f:
                np.savetxt(f, dts_local)
            with open('levels'+out+'.txt', 'ab') as f:
                np.savetxt(f, levels)
            with open('n_eq_per_level'+out+'.txt', 'ab') as f:
                np.savetxt(f, n_eq_per_level)

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

    t_eval = np.linspace(0,6,10)
    y0 = np.array([0.5, 2.7, 0.7, 1.3, 3.0, 5.0])
    #y0 = np.array([0.5, 0.7, 3.0])

    try:
        os.remove('step_sizes.txt')
        os.remove('step_sizes_local.txt')
        os.remove('levels.txt')
        os.remove('time_points.txt')
        os.remove('n_eq_per_level.txt')
        os.remove('AFs.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

#    sol = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None, dt=0.01, eps=0.001 )
#    plt.figure()
#    plt.plot(sol.t, sol.y.T)
#    plt.plot(sol.t, sol.y.T, '.', color='black')

    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None, eps=0.0001, eta = 0.00001, local_adaptivity=True, write_to_file=True)
    #plt.plot(sol2.t, sol2.y.T)
    plt.plot(sol2.t, sol2.y.T, '*')
    plt.xlabel('t')
    plt.ylabel('y')

    plt.figure()
    ts = np.loadtxt('time_points.txt')
    lev  = np.loadtxt('levels.txt')
    plt.plot(ts, lev)
    plt.xlabel('time')
    plt.ylabel('Number of levels')

    plt.figure()
    dt  = np.loadtxt('step_sizes.txt')
    plt.plot(np.cumsum(dt), dt)

    plt.plot(np.cumsum(dt), 0.04*np.ones(len(np.cumsum(dt))))
    plt.xlabel('time')
    plt.ylabel('Global step size')

    plt.figure()
    dt_locals  = np.loadtxt('step_sizes_local.txt')
    plt.plot(np.cumsum(dt_locals), dt_locals)
    plt.xlabel('time')
    plt.ylabel('Local step size')

    plt.figure()
    n_eq_per_level = np.loadtxt('n_eq_per_level.txt')
    plt.plot(ts, n_eq_per_level[0,:], label='level 0')
    plt.plot(ts, n_eq_per_level[1,:], label='level 1')
    plt.plot(ts, n_eq_per_level[2,:], label='level 2')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Number of equations per level')

    plt.figure()
    AFs = np.loadtxt('AFs.txt')
    sorted_AFs = -np.sort(-abs(AFs))
    plt.plot(sorted_AFs[0, :], label='$t=t_1$')
    plt.plot(sorted_AFs[1,:], label='$t=t_2$')
    plt.plot(sorted_AFs[2,:], label='$t=t_3$')

    plt.plot(sorted_AFs[-2,:], label='$t=t_f$')
    plt.plot(sorted_AFs[-1,:], label='$t=t_f$')
    plt.xlabel('k')
    plt.ylabel('$|\eta_k|$, sorted decreasingly')
    plt.legend()









