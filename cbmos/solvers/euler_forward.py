#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as _np
from scipy.integrate._ivp.ivp import OdeResult
import copy
import logging as _logging

import cbmos.solvers.geshgorin as gg
import cbmos.solvers.euler_backward as eb

import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=None, eps=0.01, eta=0.001,
              out='', write_to_file=False,
              local_adaptivity=False, m0=2, m1=2,
              jacobian=None, force_args={}, calculate_eigenvalues=False,
              fix_eqs=0, switch=False, K=5):
    """
    Note: t_eval can only be taken into account when dt is provided and thus
    fixed time stepping is done.
    """


    adaptive_dt = True if dt is None else False

    if len(y0) > 1 and local_adaptivity:
            # choose time step adaptively locally (if we have a system of eqs)
        return _do_local_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                               out, write_to_file, m0, m1,
                                               jacobian, force_args,
                                               calculate_eigenvalues, switch,
                                               K)

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
                                                                   calculate_eigenvalues,
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
        dt = _np.minimum(dt, tf-t)

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

    ts = _np.hstack(ts)
    ys = _np.vstack(ys).T

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
                _np.savetxt(f, _np.abs(AF).reshape((1, -1)))

        norm_AF = _np.linalg.norm(AF, _np.inf)
        dt = _np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t

        # take minimum of dt and tf-t in order to not overstep
        dt = _np.minimum(dt, tf-t)

        y = y + dt*F
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

    ts = _np.hstack(ts)
    ys = _np.vstack(ys).T
    dts = _np.hstack(dts)


    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts)

    return OdeResult(t=ts, y=ys)


def _do_global_adaptive_timestepping_with_stability(fun, t_span, y0, eps,
                                                    out, write_to_file,
                                                    jacobian,
                                                    force_args,
                                                    calculate_eigenvalues,
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

        if calculate_eigenvalues:
        #w = _np.linalg.eigvalsh(A)
            w, v = _np.linalg.eigh(A[fix_eqs:, fix_eqs:])  # adjust dimensions if equations are fixed
            _logging.debug("Eigenvalues w={}".format(w))
            _logging.debug("Eigenvectors v={}".format(v))

            if write_to_file:
                with open('eigenvalues'+out+'.txt', 'ab') as f:
                    _np.savetxt(f, w.reshape((1, -1)))
            w = w[0]
        else:

            #use gershgorin estimate
            xi = _np.diag(A)
            rho = _np.sum(_np.abs(A), axis=1) - _np.abs(xi)

            w = _np.amin(xi-rho)

            if write_to_file:
                with open('gershgorin'+out+'.txt', 'ab') as f:
                    _np.savetxt(f, [w])

        # the eigenvalues are sorted in ascending order
        dt_s = 2.0/abs(w)

        F = fun(t, y)
        F[:fix_eqs] = 0.0 # do not move fixed cells (by default fix_eqs=0)

        AF = A @ F
        AF = AF[fix_eqs:] # adjust dimensions if equations are fixed

        if write_to_file:
            with open('AFs'+out+'.txt', 'ab') as f:
                _np.savetxt(f, _np.abs(AF).reshape((1, -1)))

        norm_AF = _np.linalg.norm(AF, _np.inf)
        dt_a = _np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t

        # take minimum of stability and accuracy bound
        dt = _np.minimum(dt_s, dt_a)

        # take minimum of dt and tf-t in order to not overstep
        dt = _np.minimum(dt, tf-t)

        y = y + dt*F
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)
        dt_as.append(dt_a)
        dt_ss.append(dt_s)

    ts = _np.hstack(ts)
    ys = _np.vstack(ys).T
    dts = _np.hstack(dts)
    #dts = _np.vstack([dts, dt_as, dt_ss]).T


    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts)

    return OdeResult(t=ts, y=ys)


def _do_local_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                    out, write_to_file,
                                    m0, m1, jacobian, force_args,
                                    calculate_eigenvalues, switch, K):
    _logging.debug("Using EF, local adaptive time stepping with eps={}, eta={}, m0={} and m1={}".format(
            eps, eta, m0, m1))

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []
    dts_local = []
    dt_0s = []
    dt_1s = []
    dt_2s = []
    levels = []
    n_eq_per_level = []

    while t < tf:
        _logging.debug("t={}".format(t))

        y = copy.deepcopy(y)
        F = fun(t, y)
        #_logging.debug("F={}".format(F))

        # choose time step adaptively locally (if we have a system of eqs)
        (dt_0, dt_1, dt_2,
         dt_a,
         inds, af,
         Xi_1, Xi_2,
         EB_beneficial) = _choose_dts(fun, t, y, tf, F, eps, eta, out,
                                      write_to_file, m0, m1, jacobian,
                                      force_args, calculate_eigenvalues, K,
                                      switch)

        if EB_beneficial:
            n_eqs = _np.array([0, 0, len(y)])
            dt = _np.minimum(dt_a, tf-t)
            dts_local.append(dt)
            _logging.debug("Switching to EB with dt_a={}, K={}".format(dt, K))
            y = eb._do_newton_iterations(fun, t, y, dt, 4, jacobian,
                                         force_args, 0.001,
                                         min(1e-4, dt), min(1e-4, dt),
                                         min(1e-4, dt))
        else:

            # find corresponding indices
            min_ind_1 = len(y0) - _np.searchsorted(abs(af[inds])[::-1], Xi_1, side='right')
            min_ind_2 = len(y0) - _np.searchsorted(abs(af[inds])[::-1], Xi_2, side='right')


            if (min_ind_1 > 0) and (min_ind_1 < min_ind_2) and (min_ind_2 < len(y0)):
                _logging.debug("Three levels. i_min^1={}, i_min^2={}, dt_0={}, dt_1={}, dt_2={}".format(min_ind_1, min_ind_2, dt_0, dt_1, dt_2))
                # three levels
                n_eqs = _np.array([min_ind_1,
                                  min_ind_2 - min_ind_1,
                                  len(y) - min_ind_2])
                (y, dt) = _do_three_levels(fun, t, y, tf, F, dt_0, dt_1, dt_2,
                                           inds, min_ind_1, min_ind_2, m0, m1,
                                           dts_local)

            elif (min_ind_1 > 0 and min_ind_1 < min_ind_2 and min_ind_2 == len(y0)):
                _logging.debug("Two levels, K_2 empty. i_min^1={}, i_min^2={}, dt_0={}, dt_1={}".format(min_ind_1, min_ind_2, dt_0, dt_1))
                # two levels
                # K_2 empty, use m1=1 to ensure correct number of small time steps
                n_eqs = _np.array([min_ind_1, len(y) - min_ind_1, 0])
                (y, dt) = _do_two_levels(fun, t, y, tf, F, dt_0, dt_1, inds,
                                                min_ind_1, m0, 1, dts_local)

            elif (min_ind_1 > 0 and min_ind_1 == min_ind_2 and min_ind_2 < len(y0)):
                _logging.debug("Two levels, K_1 empty. i_min^1={}, i_min^2={}, dt_0={}, dt_2={}".format(min_ind_1, min_ind_2, dt_0, dt_2))
                # two levels
                # K_1 empty, however we shift the levels down, because else things don't seem to work
                n_eqs = _np.array([min_ind_2, 0, len(y) - min_ind_2])
                (y, dt) = _do_two_levels(fun, t, y, tf, F, dt_0, dt_2, inds,
                                                min_ind_1, m0, m1, dts_local)

            elif (min_ind_1 == 0 and min_ind_1 < min_ind_2 and min_ind_2 < len(y0)):
                _logging.debug("Two levels, K_0 empty. i_min^1={}, i_min^2={}, dt_1={}, dt_2={}".format(min_ind_1, min_ind_2, dt_1, dt_2))
                # two levels
                # K_0 empty, however we shift the levels down, because else things don't seem to work
                n_eqs = _np.array([0, min_ind_2, len(y) - min_ind_2])
                (y, dt) = _do_two_levels(fun, t, y, tf, F, dt_1, dt_2, inds,
                                                min_ind_2, 1, m1, dts_local)
            else:
                # single level
                _logging.debug("Single level, i_min^1={}, i_min^2={}".format(min_ind_1, min_ind_2))
                n_eqs = _np.array([0, 0, len(y)])
                _logging.debug("Using EF with with dt_a={}, dt_s={}, K={}".format(dt_a, dt_2, K))
                (y, dt) = _do_single_level(t, y, tf, F, dt_2, dts_local)

        _logging.debug("y={}".format(y))
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

        dt_0s.append(dt_0)
        dt_1s.append(dt_1)
        dt_2s.append(dt_2)

        n_eq_per_level.append(n_eqs)
        levels.append(_np.sum(n_eqs > 0))

    ts = _np.hstack(ts)
    ys = _np.vstack(ys).T
    dts = _np.hstack(dts)
    dts_local = _np.hstack(dts_local)
    dt_0s = _np.hstack(dt_0s)
    dt_1s = _np.hstack(dt_1s)
    dt_2s = _np.hstack(dt_2s)
    n_eq_per_level = _np.vstack(n_eq_per_level).T

    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts)
        with open('step_sizes_dt_0'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dt_0s)
        with open('step_sizes_dt_1'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dt_1s)
        with open('step_sizes_dt_2'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dt_2s)
        with open('step_sizes_local'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts_local)
        with open('levels'+out+'.txt', 'ab') as f:
            _np.savetxt(f, levels)
        with open('n_eq_per_level'+out+'.txt', 'ab') as f:
            _np.savetxt(f, n_eq_per_level)

    return OdeResult(t=ts, y=ys)


def _choose_dts(fun, t, y, tf, F, eps, eta, out, write_to_file, m0, m1,
                jacobian, force_args, calculate_eigenvalues, K, switch):

    EB_beneficial = False

    if jacobian is None:
        # accuracy limit based only
        af = 1/eta*(fun(t, y + eta * F) - F)
        # sort the indices such that abs(AF(inds)) is decreasing
        inds = _np.argsort(-abs(af))
        # find largest and smallest eta_k
        Xi_0 = abs(af[inds[0]])
        dt_a = _np.sqrt(2*eps / (m0*m1*Xi_0)) if Xi_0 > 0.0 else tf - t

        dt_0 = dt_a
        dt_1 = m0*dt_0
        dt_2 = m1*dt_1

        # calculate corresponding maximum eta for each level
        Xi_1 = Xi_0/m0
        Xi_2 = Xi_1/m1

    else:
        # calculate stability bound
        A = jacobian(y, force_args)

        if calculate_eigenvalues:
            #w = _np.linalg.eigvalsh(A)
            w, v = _np.linalg.eigh(A)
            _logging.debug("Eigenvalues w={}".format(w))
            _logging.debug("Eigenvectors v={}".format(v))

            if write_to_file:
                with open('eigenvalues'+out+'.txt', 'ab') as f:
                    _np.savetxt(f, w.reshape((1, -1)))
            w = w[0]
        else:

            # use gershgorin estimate
            xi = _np.diag(A)
            rho = _np.sum(_np.abs(A), axis=1) - _np.abs(xi)

            w = _np.amin(xi-rho)

            if write_to_file:
                with open('gershgorin'+out+'.txt', 'ab') as f:
                    _np.savetxt(f, [w])

        # the eigenvalues are sorted in ascending order
        dt_s = 2.0/abs(w)

        # calculate the accuracy bound
        af = A @ F
        # sort the indices such that abs(AF(inds)) is decreasing
        inds = _np.argsort(-abs(af))
        # find largest eta_k
        Xi_0 = abs(af[inds[0]])

        dt_a = _np.sqrt(2*eps*m0*m1/Xi_0) if Xi_0 > 0.0 else tf - t
        # do not overshoot the final time
#        dt_a = _np.minimum(dt_a, tf - t)

        if dt_a > K * dt_s and switch:
            EB_beneficial = True
            dt_0 = dt_a
            dt_1 = dt_a
            dt_2 = dt_a
            Xi_1 = None
            Xi_2 = None
        else:
            # stick with Euler forward
            if dt_a > dt_s:
                # stability bounded time step
                dt_2 = dt_s
                Xi_2 = 2.0*eps/(dt_s**2)
                Xi_1 = m1*Xi_2

            else:
                # accuracy bounded time step
                dt_2 = dt_a
                Xi_1 = Xi_0/m0
                Xi_2 = Xi_1/m1

            #Xi_1 = Xi_0/m0
            #Xi_2 = Xi_1/m1
            dt_1 = dt_2/m1
            dt_0 = dt_1/m0

    if write_to_file:
        with open('AFs'+out+'.txt', 'ab') as f:
            _np.savetxt(f, _np.abs(af).reshape((1, -1)))

    return (dt_0, dt_1, dt_2, dt_a, inds, af, Xi_1, Xi_2, EB_beneficial)


def _do_single_level(t, y, tf, F, dt_0, dts_local):
    # single level
    dt_0 = _np.minimum(dt_0, tf-t)
    y = y + dt_0*F
    dts_local.append(dt_0)
    return (y, dt_0)


def _do_two_levels(fun, t, y, tf, F, dt_0, dt_1, inds, min_ind_1, m0, m1,
                   dts_local):

    dt_0 = _np.minimum(dt_0, (tf-t)/m0)
    for j in range(m0*m1):
        y[inds[:min_ind_1]] = y[inds[:min_ind_1]] + dt_0*F[inds[:min_ind_1]]
        #F = fun(t, y)
        dts_local.append(dt_0)

    dt_1 = _np.minimum(dt_1, (tf-t))

    y[inds[min_ind_1:]] = y[inds[min_ind_1:]] + dt_1*F[inds[min_ind_1:]]
    return (y, dt_1)


def _do_three_levels(fun, t, y, tf, F, dt_0, dt_1, dt_2, inds, min_ind_1,
                     min_ind_2, m0, m1, dts_local):

    dt_1 = _np.minimum(dt_1, (tf-t)/m1)  # avoid overstepping at end of time interval
    for i in range(m1):
        dt_0 = _np.minimum(dt_0, (tf-t)/m0)  # avoid overstepping at end of time interval
        for j in range(m0):
            y[inds[:min_ind_1]] = y[inds[:min_ind_1]] + dt_0*F[inds[:min_ind_1]]
            #F = fun(t, y)
            dts_local.append(dt_0)

        y[inds[min_ind_1:min_ind_2]] = y[inds[min_ind_1:min_ind_2]] + dt_1*F[inds[min_ind_1:min_ind_2]]
        #F = fun(t, y)

    dt_2 = _np.minimum(dt_2, tf-t)
    y[inds[min_ind_2:]] = y[inds[min_ind_2:]] + dt_2*F[inds[min_ind_2:]]

    return (y, dt_2)


if __name__ == "__main__":

    logger = _logging.getLogger()
    logger.setLevel(_logging.DEBUG)

    # stability region for Euler forward for this problem is h<2/50=0.04
    #@_np.vectorize
    def func(t, y):
        return -50*_np.eye(len(y))@y
    def jacobian(y, fa):
        return -50*_np.eye(len(y))

#    t_span = (0,1)
#    y0 = _np.array([1,1])
#
#    sol = solve_ivp(func, t_span, y0 )
#
#    plt.figure()
#    plt.plot(sol.t, sol.y)

    t_eval = _np.linspace(0, 1, 10)
    #y0 = _np.array([0.7, 1.3, 3.0, 0.2])
    y0 = _np.array([3.0, 1.3, 0.7, 0.2])

    #y0 = _np.array([0.5, 0.7, 1.0, 3.0])
    #y0 = _np.array([0.0, 0.0, 0.0])

    try:
        os.remove('step_sizes.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('step_sizes_local.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('step_sizes_dt_0.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('step_sizes_dt_1.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('step_sizes_dt_2.txt')
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
                     eps=0.0001, eta = 0.00001, local_adaptivity=True,
                     write_to_file=True, jacobian=jacobian, switch=True, K=3)
    #plt.plot(sol2.t, sol2.y.T)
    plt.plot(sol2.t, sol2.y.T, '*')
    plt.xlabel('t')
    plt.ylabel('y')

#    plt.figure()
#    lev  = _np.loadtxt('levels.txt')
#    plt.plot(sol2.t[:-1], lev)
##    plt.xlabel('time')
#    plt.ylabel('Number of levels')

    plt.figure()
    dt  = _np.loadtxt('step_sizes.txt')
    plt.plot(sol2.t[:-1], dt)
    plt.plot(sol2.t, 0.04*_np.ones(len(sol2.t)))
    plt.xlabel('time')
    plt.ylabel('Global step size')

#    plt.figure()
#    dt_locals  = _np.loadtxt('step_sizes_local.txt')
#    plt.plot(_np.cumsum(dt_locals), dt_locals)
#    plt.xlabel('time')
#    plt.ylabel('Local step size')
#
#    plt.figure()
#    dt_0s = _np.loadtxt('step_sizes_dt_0.txt')
#    dt_1s = _np.loadtxt('step_sizes_dt_1.txt')
#    dt_2s = _np.loadtxt('step_sizes_dt_2.txt')
#    plt.plot(sol2.t[:-1], dt_0s, label='dt_0')
#    plt.plot(sol2.t[:-1], dt_1s, label='dt_1')
#    plt.plot(sol2.t[:-1], dt_2s, label='dt_2')
#    plt.xlabel('time')
#    plt.ylabel('dt_i')
#    plt.legend()
##
#    plt.figure()
#    n_eq_per_level = _np.loadtxt('n_eq_per_level.txt')
#    plt.plot(sol2.t[:-1], n_eq_per_level[0,:], label='level 0')
#    plt.plot(sol2.t[:-1], n_eq_per_level[1,:], label='level 1')
#    plt.plot(sol2.t[:-1], n_eq_per_level[2,:], label='level 2')
#    plt.legend()
#    plt.xlabel('time')
#    plt.ylabel('Number of equations per level')
#
#    print(sol2.t[-1])
#
#    plt.figure()
#    AFs = _np.loadtxt('AFs.txt')
#    sorted_AFs = -_np.sort(-abs(AFs))
#    plt.plot(sorted_AFs[0, :], label='$t=t_1$')
#    plt.plot(sorted_AFs[1,:], label='$t=t_2$')
#    plt.plot(sorted_AFs[2,:], label='$t=t_3$')
#
#    plt.plot(sorted_AFs[-2,:], label='$t=t_f$')
#    plt.plot(sorted_AFs[-1,:], label='$t=t_f$')
#    plt.xlabel('k')
#    #plt.ylabel('$|\eta_k|$, sorted decreasingly')
#    plt.legend()
