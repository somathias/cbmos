#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
import copy
import logging as _logging

import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=None, eps=0.01, eta=0.001,
              out='', write_to_file=False,
              local_adaptivity=False, m0=2, m1=2,
              jacobian=None, force_args={}, fix_eqs=0):
    """
    Note: t_eval can only be taken into account when dt is provided and thus
    fixed time stepping is done.
    """


    adaptive_dt = True if dt is None else False

    if len(y0) > 1 and local_adaptivity:
        if jacobian is None:
            # choose time step adaptively locally (if we have a system of eqs)
            return _do_local_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                                   out, write_to_file, m0, m1)
        else:
            return _do_local_adaptive_timestepping_with_stability(fun, t_span,
                                                                  y0, eps,
                                                                  eta, out,
                                                                  write_to_file,
                                                                  m0, m1,
                                                                  jacobian,
                                                                  force_args)
    elif adaptive_dt:
        # choose time step adaptively globally
        if jacobian is None:
            return _do_global_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                                    out, write_to_file)
        else:
            return _do_global_adaptive_timestepping_with_stability(fun, t_span,
                                                                   y0, eps,
                                                                   out,
                                                                   write_to_file,
                                                                   jacobian,
                                                                   force_args,
                                                                   fix_eqs)
    else:
        # do regular fixed time stepping
        return _do_fixed_timestepping(fun, t_span, y0, t_eval, dt)


def _do_fixed_timestepping(fun, t_span, y0, t_eval, dt):

    _logging.debug("Using EF, fixed time stepping with dt={}".format(
            dt))

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

    while t < tf:

        # take minimum of dt and tf-t in order to not overstep
        dt = np.minimum(dt, tf-t)

        y = copy.deepcopy(y)
        y = y + dt*fun(t, y)

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


def _do_global_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                     out, write_to_file):

    _logging.debug("Using EF, global adaptive time stepping with eps={}, eta={}".format(
            eps, eta))

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []


    while t < tf:
        y = copy.deepcopy(y)
        F = fun(t, y)
        AF = 1/eta*(fun(t, y + eta * F) - F)

        if write_to_file:
            with open('AFs'+out+'.txt', 'ab') as f:
                np.savetxt(f, np.abs(AF).reshape((1, -1)))

        norm_AF = np.linalg.norm(AF, np.inf)
        dt = np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t

        # take minimum of dt and tf-t in order to not overstep
        dt = np.minimum(dt, tf-t)

        y = y + dt*F
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

    ts = np.hstack(ts)
    ys = np.vstack(ys).T
    dts = np.hstack(dts)


    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts)

    return OdeResult(t=ts, y=ys)


def _do_global_adaptive_timestepping_with_stability(fun, t_span, y0, eps,
                                                    out, write_to_file,
                                                    jacobian,
                                                    force_args,
                                                    fix_eqs):

    _logging.debug("Using EF, global adaptive time stepping with Jacobian and eps={}".format(
            eps))

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []
    dt_as = []
    dt_ss = []


    while t < tf:
        _logging.debug("t={}".format(t))
        y = copy.deepcopy(y)

        # calculate stability bound
        A = jacobian(y, force_args)
        #w = np.linalg.eigvalsh(A)
        w, v = np.linalg.eigh(A)
        _logging.debug("Eigenvalues w={}".format(w))
        _logging.debug("Eigenvectors v={}".format(v))

        if write_to_file:
            with open('eigenvalues'+out+'.txt', 'ab') as f:
                np.savetxt(f, w.reshape((1, -1)))
            with open('eigenvectors'+out+'.txt', 'ab') as f:
                np.savetxt(f, v.reshape((1, -1), order='F'))


        # the eigenvalues are sorted in ascending order
        dt_s = 2.0/abs(w[0])

        F = fun(t, y)
        AF = A @ F

        if write_to_file:
            with open('AFs'+out+'.txt', 'ab') as f:
                np.savetxt(f, np.abs(AF).reshape((1, -1)))

        norm_AF = np.linalg.norm(AF, np.inf)
        dt_a = np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t

        # take minimum of stability and accuracy bound
        dt = np.minimum(dt_s, dt_a)

        # take minimum of dt and tf-t in order to not overstep
        dt = np.minimum(dt, tf-t)

        if fix_eqs > 0:
            F[:fix_eqs] = 0.0

        y = y + dt*F
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)
        dt_as.append(dt_a)
        dt_ss.append(dt_s)

    ts = np.hstack(ts)
    ys = np.vstack(ys).T
    #dts = np.hstack(dts)
    dts = np.vstack([dts, dt_as, dt_ss]).T


    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts)

    return OdeResult(t=ts, y=ys)


def _do_local_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                    out, write_to_file,
                                    m0, m1):
    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []
    dts_local = []
    levels = []
    n_eq_per_level = []

    while t < tf:

        y = copy.deepcopy(y)
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

        dt_0 = np.sqrt(2*eps / (m0*m1*Xi_0)) if Xi_0 > 0.0 else tf - t

        if dt_0 == tf - t:
            # This is the case if AF == 0
            # then all higher levels should use the same time step
            dt_1 = dt_0
            dt_2 = dt_1
        else:
            dt_1 = m0*dt_0
            dt_2 = m1*dt_1

        # calculate corresponding maximum eta for each level
        Xi_1 = Xi_0/m0
        Xi_2 = Xi_1/m1

        # find corresponding indices
        min_ind_1 = len(y0) - np.searchsorted(abs(af[inds])[::-1], Xi_1, side='right')
        min_ind_2 = len(y0) - np.searchsorted(abs(af[inds])[::-1], Xi_2, side='right')

        n_eqs = np.array([min_ind_1,
                          min_ind_2 - min_ind_1,
                          len(y0) - min_ind_2])
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

        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

    ts = np.hstack(ts)
    ys = np.vstack(ys).T
    dts = np.hstack(dts)
    dts_local = np.hstack(dts_local)
    n_eq_per_level = np.vstack(n_eq_per_level).T

    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts)
        with open('step_sizes_local'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts_local)
        with open('levels'+out+'.txt', 'ab') as f:
            np.savetxt(f, levels)
        with open('n_eq_per_level'+out+'.txt', 'ab') as f:
            np.savetxt(f, n_eq_per_level)

    return OdeResult(t=ts, y=ys)

def _do_local_adaptive_timestepping_with_stability(fun, t_span, y0, eps, eta,
                                                   out, write_to_file,
                                                   m0, m1, jacobian,
                                                   force_args):

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []
    dts_local = []
    levels = []
    n_eq_per_level = []

    while t < tf:

        y = copy.deepcopy(y)

        # calculate stability bound
        A = jacobian(y, force_args)
        w, v = np.linalg.eigh(A)

        # the eigenvalues are sorted in ascending order
        dt_s = 2.0/abs(w[0])

        # calculate the accuracy bound
        F = fun(t, y)
        af = 1/eta*(fun(t, y + eta * F) - F)

        if write_to_file:
            with open('AFs'+out+'.txt', 'ab') as f:
                np.savetxt(f, np.abs(af).reshape((1, -1)))

        # sort the indices such that abs(AF(inds)) is decreasing
        inds = np.argsort(-abs(af))
        # find largest and smallest eta_k
        Xi_0 = abs(af[inds[0]])

#        Xi_min = abs(af[inds[-1]])
#        dt_max = np.sqrt(2*eps/Xi_min) if Xi_min > 0.0 else tf - t

        dt_a = np.sqrt(2*eps / (m0*m1*Xi_0)) if Xi_0 > 0.0 else dt_s
        dt_0 = np.minimum(dt_a, dt_s)
        dt_1 = np.minimum(m0*dt_0, dt_s)
        dt_2 = np.minimum(m1*dt_1, dt_s)

        if dt_0 == dt_s:
            # single level sufficient with dt_s
            n_eqs = np.array([len(y0), 0, 0])
            n_eq_per_level.append(n_eqs)
            levels.append(np.sum(n_eqs > 0))

            y = y + dt_0*F
            dt = dt_0
            dts_local.append(dt_0)

        elif dt_1 == dt_s:
            # two levels with dt_0 and dt_s
            # calculate corresponding maximum eta for each level
            Xi_1 = Xi_0/m0
            # find corresponding indices
            min_ind_1 = len(y0) - np.searchsorted(abs(af[inds])[::-1], Xi_1, side='right')

            n_eqs = np.array([min_ind_1, len(y0) -min_ind_1, 0])
            n_eq_per_level.append(n_eqs)
            levels.append(np.sum(n_eqs > 0))

            for j in range(m0):
                y[inds[:min_ind_1]] = y[inds[:min_ind_1]] + dt_0*F[inds[:min_ind_1]]
                F = fun(t, y)
                dts_local.append(dt_0)

            y[inds[min_ind_1:]] = y[inds[min_ind_1:]] + dt_1*F[inds[min_ind_1:]]
            F = fun(t, y)
            dt = dt_1
        else:
            # not restricted by stability in lower levels
            # calculate corresponding maximum eta for each level
            Xi_1 = Xi_0/m0
            Xi_2 = Xi_1/m1

            # find corresponding indices
            min_ind_1 = len(y0) - np.searchsorted(abs(af[inds])[::-1], Xi_1, side='right')
            min_ind_2 = len(y0) - np.searchsorted(abs(af[inds])[::-1], Xi_2, side='right')

            n_eqs = np.array([min_ind_1,
                              min_ind_2 - min_ind_1,
                              len(y0) - min_ind_2])
            n_eq_per_level.append(n_eqs)
            levels.append(np.sum(n_eqs > 0))

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

        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

    ts = np.hstack(ts)
    ys = np.vstack(ys).T
    dts = np.hstack(dts)
    dts_local = np.hstack(dts_local)
    n_eq_per_level = np.vstack(n_eq_per_level).T

    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts)
        with open('step_sizes_local'+out+'.txt', 'ab') as f:
            np.savetxt(f, dts_local)
        with open('levels'+out+'.txt', 'ab') as f:
            np.savetxt(f, levels)
        with open('n_eq_per_level'+out+'.txt', 'ab') as f:
            np.savetxt(f, n_eq_per_level)

    return OdeResult(t=ts, y=ys)



if __name__ == "__main__":

    # stability region for Euler forward for this problem is h<2/50=0.04
    #@np.vectorize
    def func(t, y):
        return -50*np.eye(len(y))@y
    def jacobian(y, fa):
        return -50*np.eye(len(y))

#    t_span = (0,1)
#    y0 = np.array([1,1])
#
#    sol = solve_ivp(func, t_span, y0 )
#
#    plt.figure()
#    plt.plot(sol.t, sol.y)

    t_eval = np.linspace(0,3,10)
    y0 = np.array([0.5, 2.7, 0.7, 1.3, 3.0, 5.0])
    #y0 = np.array([0.5, 0.7, 1.0, 3.0])
    #y0 = np.array([0.0, 0.0, 0.0])

    try:
        os.remove('step_sizes.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('step_sizes_local.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('levels.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('n_eq_per_level.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('AFs.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

#
    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None,
                     eps=0.0001, eta = 0.00001, local_adaptivity=False,
                     write_to_file=True, jacobian=jacobian)
#    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None,
#                     eps=0.0001, eta = 0.00001, local_adaptivity=False,
#                     write_to_file=True)

    #plt.plot(sol2.t, sol2.y.T)
    plt.plot(sol2.t, sol2.y.T, '*')
    plt.xlabel('t')
    plt.ylabel('y')

#    plt.figure()
#    lev  = np.loadtxt('levels.txt')
#    plt.plot(sol2.t, lev)
#    plt.xlabel('time')
#    plt.ylabel('Number of levels')

    plt.figure()
    dt  = np.loadtxt('step_sizes.txt')
    plt.plot(sol2.t[:-1], dt[:,0])
    plt.plot(sol2.t, 0.04*np.ones(len(sol2.t)))
    plt.xlabel('time')
    plt.ylabel('Global step size')

#    plt.figure()
#    dt_locals  = np.loadtxt('step_sizes_local.txt')
#    plt.plot(np.cumsum(dt_locals), dt_locals)
#    plt.xlabel('time')
#    plt.ylabel('Local step size')
#
#    plt.figure()
#    n_eq_per_level = np.loadtxt('n_eq_per_level.txt')
#    plt.plot(ts[1:], n_eq_per_level[0,:], label='level 0')
#    plt.plot(ts[1:], n_eq_per_level[1,:], label='level 1')
#    plt.plot(ts[1:], n_eq_per_level[2,:], label='level 2')
#    plt.legend()
#    plt.xlabel('time')
#    plt.ylabel('Number of equations per level')

    plt.figure()
    AFs = np.loadtxt('AFs.txt')
    sorted_AFs = -np.sort(-abs(AFs))
    plt.plot(sorted_AFs[0, :], label='$t=t_1$')
    plt.plot(sorted_AFs[1,:], label='$t=t_2$')
    plt.plot(sorted_AFs[2,:], label='$t=t_3$')

    plt.plot(sorted_AFs[-2,:], label='$t=t_f$')
    plt.plot(sorted_AFs[-1,:], label='$t=t_f$')
    plt.xlabel('k')
    #plt.ylabel('$|\eta_k|$, sorted decreasingly')
    plt.legend()
