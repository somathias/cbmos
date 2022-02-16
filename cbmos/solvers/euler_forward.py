#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as _np
from scipy.integrate._ivp.ivp import OdeResult
from scipy.linalg import eigh
import copy
import logging as _logging
import time


import cbmos.solvers.euler_backward as eb

import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=None, eps=0.01, eta=0.001,
              out='', write_to_file=False,
              local_adaptivity=False, m0=2, m1=2,
              jacobian=None, force_args={}, calculate_eigenvalues=False,
              always_calculate_Jacobian=False,
              fix_eqs=0, switch=False, K=5, dim=None, rA=1.5,
              measure_wall_time=False):
    """
    Note: t_eval can only be taken into account when dt is provided and thus
    fixed time stepping is done.
    """

    n_av = 9
    av_tol = 0.001

    adaptive_dt = True if dt is None else False

    # make sure input is float
    y0 = y0.astype('float64')

    if len(y0) > 1 and local_adaptivity:
            # choose time step adaptively locally (if we have a system of eqs)
#        eps = eps*m0*m1 #scale local epsilon with level ratios
#        return _do_local_adaptive_timestepping(fun, t_span, y0, eps, eta,
#                                               out, write_to_file, m0, m1,
#                                               update_F,
#                                               jacobian, force_args,
#                                               calculate_eigenvalues,
#                                               always_calculate_Jacobian,
#                                               n_av, av_tol,
#                                               switch, K,
#                                               measure_wall_time)
        return _do_local_adaptive_timestepping2(fun, t_span, y0, eps, eta,
                                                out, write_to_file, m0,
                                                jacobian, force_args,
                                                calculate_eigenvalues,
                                                switch, K, dim, rA,
                                                measure_wall_time)

    elif adaptive_dt:
        # choose time step adaptively globally
        if jacobian is None:
            return _do_global_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                                    out, write_to_file,
                                                    measure_wall_time)
        else:
            return _do_global_adaptive_timestepping_with_stability(fun, t_span,
                                                                   y0, eps,
                                                                   out,
                                                                   write_to_file,
                                                                   jacobian,
                                                                   force_args,
                                                                   calculate_eigenvalues,
                                                                   always_calculate_Jacobian,
                                                                   n_av, av_tol,
                                                                   fix_eqs,
                                                                   measure_wall_time)
    else:
        # do regular fixed time stepping
        return _do_fixed_timestepping(fun, t_span, y0, t_eval, dt, out,
                                      measure_wall_time)


def _do_fixed_timestepping(fun, t_span, y0, t_eval, dt, out,
                           measure_wall_time):

    _logging.debug("Using EF, fixed time stepping with dt={}".format(
            dt))

    if measure_wall_time:
        exec_time_start = time.time()
        exec_times = []
        n_F_evaluations = 0
        F_evaluations = []


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

        if measure_wall_time:
            exec_time = time.time() - exec_time_start
            exec_times.append((t, exec_time))
            n_F_evaluations += 1
            F_evaluations.append((t, n_F_evaluations))


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

    if measure_wall_time:
        with open('exec_times'+out+'.txt', 'ab') as f:
            _np.savetxt(f, exec_times)
        with open('F_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, F_evaluations)

    return OdeResult(t=ts, y=ys)


def _do_global_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                     out, write_to_file,
                                     measure_wall_time):

    _logging.debug("Using EF, global adaptive time stepping with eps={}, eta={}".format(
            eps, eta))

    if measure_wall_time:
        exec_time_start = time.time()
        exec_times = []
        F_evaluations = []
        n_F_evaluations = 0


    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []


    while t < tf:

        if measure_wall_time:
            exec_time = time.time() - exec_time_start
            exec_times.append((t, exec_time))
            n_F_evaluations += 2
            F_evaluations.append((t, n_F_evaluations))

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

    if measure_wall_time:
        with open('exec_times'+out+'.txt', 'ab') as f:
            _np.savetxt(f, exec_times)
        with open('F_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, F_evaluations)

    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts)


    return OdeResult(t=ts, y=ys)


def _do_global_adaptive_timestepping_with_stability(fun, t_span, y0, eps,
                                                    out, write_to_file,
                                                    jacobian,
                                                    force_args,
                                                    calculate_eigenvalues,
                                                    always_calculate_Jacobian,
                                                    n_av, av_tol,
                                                    fix_eqs,
                                                    measure_wall_time):

    _logging.debug("Using EF, global adaptive time stepping with Jacobian and eps={}".format(
            eps))

    if measure_wall_time:
        exec_time_start = time.time()
        exec_times = []
        n_F_evaluations = 0
        F_evaluations = []
        n_A_evaluations = 0
        A_evaluations = []


    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []
    dt_as = []
    dt_ss = []

    do_not_update_dt_s = False

    while t < tf:
        if measure_wall_time:
            exec_time = time.time() - exec_time_start
            exec_times.append((t, exec_time))
            n_F_evaluations +=1
            F_evaluations.append((t, n_F_evaluations))


        _logging.debug("t={}".format(t))
        y = copy.deepcopy(y)

        # check if dt_s average has converged
        if not always_calculate_Jacobian and len(dt_ss) >= n_av and not do_not_update_dt_s:
            old_av_dts = _np.mean(dt_ss[-n_av:-1])
            new_av_dts = _np.mean(dt_ss[-n_av + 1:])

            if _np.abs(old_av_dts - new_av_dts) < av_tol*_np.abs(new_av_dts):
                do_not_update_dt_s = True
                dt_s = new_av_dts
                _logging.debug("Stopped updating Jacobian at t={}. Using dt_s={}".format(t, dt_s))

#        if do_not_update_dt_s:
##            dt_s = dt_ss[-1]
#            print('Do not update dt s')
        if not do_not_update_dt_s:
            # calculate stability bound
            A = jacobian(y, force_args)
            if measure_wall_time:
                n_A_evaluations += 1
                A_evaluations.append((t, n_A_evaluations))

            if calculate_eigenvalues:
            #w = _np.linalg.eigvalsh(A)
                w = eigh(A[fix_eqs:, fix_eqs:], eigvals_only=True)  # adjust dimensions if equations are fixed
                _logging.debug("Eigenvalues w={}".format(w))
                #_logging.debug("Eigenvectors v={}".format(v))

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
    dts = _np.vstack([dts, dt_as, dt_ss]).T

    if measure_wall_time:
        with open('exec_times'+out+'.txt', 'ab') as f:
            _np.savetxt(f, exec_times)
        with open('F_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, F_evaluations)
        with open('A_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, A_evaluations)

    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts)

    return OdeResult(t=ts, y=ys)


def _do_local_adaptive_timestepping(fun, t_span, y0, eps, eta,
                                    out, write_to_file,
                                    m0, m1, update_F, jacobian, force_args,
                                    calculate_eigenvalues,
                                    always_calculate_Jacobian,
                                    n_av, av_tol,
                                    switch, K,
                                    measure_wall_time):
    _logging.debug("Using EF, local adaptive time stepping with eps={}, eta={}, m0={} and m1={}".format(
            eps, eta, m0, m1))
    n_F_evaluations = 0
    n_A_evaluations = 0


    if measure_wall_time:
        exec_time_start = time.time()
        exec_times = []
        F_evaluations = []
        A_evaluations = []

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
    dt_ss = []
    levels = []
    n_eq_per_level = []

    do_not_update_dt_s = False

    while t < tf:
        if measure_wall_time:
            exec_time = time.time() - exec_time_start
            exec_times.append((t, exec_time))

        _logging.debug("t={}".format(t))

        y = copy.deepcopy(y)
        F = fun(t, y)
        n_F_evaluations += 1
        #_logging.debug("F={}".format(F))

        # choose time step adaptively locally (if we have a system of eqs)
        (dt_0, dt_1, dt_2,
         dt_a,
         dt_s,
         inds, af,
         Xi_1, Xi_2,
         do_not_update_dt_s,
         EB_beneficial,
         n_F_evaluations,
         n_A_evaluations) = _choose_dts(fun, t, y, tf, F, eps, eta, out,
                                      write_to_file, m0, m1, jacobian,
                                      force_args, calculate_eigenvalues,
                                      always_calculate_Jacobian,
                                      do_not_update_dt_s, dt_ss,
                                      n_av, av_tol,
                                      K, switch,
                                      n_F_evaluations, n_A_evaluations)

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
                                           dts_local, update_F)

            elif (min_ind_1 > 0 and min_ind_1 < min_ind_2 and min_ind_2 == len(y0)):
                _logging.debug("Two levels, K_2 empty. i_min^1={}, i_min^2={}, dt_0={}, dt_1={}".format(min_ind_1, min_ind_2, dt_0, dt_1))
                # two levels
                # K_2 empty, use m1=1 to ensure correct number of small time steps
                n_eqs = _np.array([min_ind_1, len(y) - min_ind_1, 0])
                (y, dt) = _do_two_levels(fun, t, y, tf, F, dt_0, dt_1, inds,
                                                min_ind_1, m0, 1, dts_local,
                                                update_F)

            elif (min_ind_1 > 0 and min_ind_1 == min_ind_2 and min_ind_2 < len(y0)):
                _logging.debug("Two levels, K_1 empty. i_min^1={}, i_min^2={}, dt_0={}, dt_2={}".format(min_ind_1, min_ind_2, dt_0, dt_2))
                # two levels
                # K_1 empty,
                n_eqs = _np.array([min_ind_2, 0, len(y) - min_ind_2])
                (y, dt) = _do_two_levels(fun, t, y, tf, F, dt_0, dt_2, inds,
                                                min_ind_1, m0, m1, dts_local,
                                                update_F)

            elif (min_ind_1 == 0 and min_ind_1 < min_ind_2 and min_ind_2 < len(y0)):
                _logging.debug("Two levels, K_0 empty. i_min^1={}, i_min^2={}, dt_1={}, dt_2={}".format(min_ind_1, min_ind_2, dt_1, dt_2))
                # two levels
                # K_0 empty, use m0=1 to ensure correct number of small time steps
                n_eqs = _np.array([0, min_ind_2, len(y) - min_ind_2])
                (y, dt) = _do_two_levels(fun, t, y, tf, F, dt_1, dt_2, inds,
                                                min_ind_2, 1, m1, dts_local,
                                                update_F)
            elif (0 == min_ind_1 == min_ind_2):
                # single level, K_0 and K_1 empty
                _logging.debug("Single level, K_0 and K_1 empty, i_min^1={}, i_min^2={}".format(min_ind_1, min_ind_2))
                n_eqs = _np.array([0, 0, len(y)])
                _logging.debug("Using EF with with dt_2={}, dt_a={}, dt_s={}, K={}".format(dt_2, dt_a, dt_s, K))
                (y, dt ) = _do_single_level(t, y, tf, F, dt_2, dts_local)
            elif (0 == min_ind_1 and min_ind_2 == len(y0)):
                # single level, K_0 and K_2 empty
                _logging.debug("Single level, K_0 and K_2 empty, i_min^1={}, i_min^2={}".format(min_ind_1, min_ind_2))
                n_eqs = _np.array([0, len(y), 0])
                _logging.debug("Using EF with with dt_1={}, dt_a={} dt_s={}, K={}".format(dt_1, dt_a, dt_s, K))
                (y, dt ) = _do_single_level(t, y, tf, F, dt_1, dts_local)
            else:
                # single level, K_1 and K_2 empty
                _logging.debug("Single level, K_1 and K_2 empty, i_min^1={}, i_min^2={}".format(min_ind_1, min_ind_2))
                n_eqs = _np.array([len(y), 0, 0])
                _logging.debug("Using EF with with dt_0={}, dt_a={} dt_s={}, K={}".format(dt_0, dt_a, dt_s, K))
                (y, dt ) = _do_single_level(t, y, tf, F, dt_0, dts_local)

        _logging.debug("y={}".format(y))
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

        dt_0s.append(dt_0)
        dt_1s.append(dt_1)
        dt_2s.append(dt_2)

        dt_ss.append(dt_s)

        n_eq_per_level.append(n_eqs)
        levels.append(_np.sum(n_eqs > 0))

        if measure_wall_time:
            F_evaluations.append((t, n_F_evaluations))
            A_evaluations.append((t, n_A_evaluations))

    ts = _np.hstack(ts)
    ys = _np.vstack(ys).T
    dts = _np.hstack(dts)
    #dts = _np.vstack([dts, dt_as, dt_ss]).T

    dts_local = _np.hstack(dts_local)
    dt_0s = _np.hstack(dt_0s)
    dt_1s = _np.hstack(dt_1s)
    dt_2s = _np.hstack(dt_2s)
    n_eq_per_level = _np.vstack(n_eq_per_level).T

    if measure_wall_time:
        with open('exec_times'+out+'.txt', 'ab') as f:
            _np.savetxt(f, exec_times)
        with open('F_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, F_evaluations)
        with open('A_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, A_evaluations)

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
                jacobian, force_args, calculate_eigenvalues,
                always_calculate_Jacobian, do_not_update_dt_s, dt_ss,
                n_av, av_tol,
                K, switch,
                n_F_evaluations, n_A_evaluations):

    EB_beneficial = False

    if jacobian is None:
        # accuracy limit based only
        af = 1/eta*(fun(t, y + eta * F) - F)
        n_F_evaluations += 1
        # sort the indices such that abs(AF(inds)) is decreasing
        inds = _np.argsort(-abs(af))
        # find largest and smallest eta_k
        Xi_0 = abs(af[inds[0]])
#        dt_a = _np.sqrt(2*eps / (m0*m1*Xi_0)) if Xi_0 > 0.0 else tf - t
#        dt_0 = dt_a
#        dt_1 = m0*dt_0
#        dt_2 = m1*dt_1
#
#        # calculate corresponding maximum eta for each level
#        Xi_1 = Xi_0/m0
#        Xi_2 = Xi_1/m1

        dt_a = _np.sqrt(2*eps*m0*m1/Xi_0) if Xi_0 > 0.0 else tf - t
        dt_2 = dt_a
        Xi_1 = Xi_0/m0
        Xi_2 = Xi_1/m1
        dt_1 = dt_2/m1
        dt_0 = dt_1/m0

        dt_s = None

    else:

        # check if dt_s average has converged
        if not always_calculate_Jacobian and len(dt_ss) >= n_av and not do_not_update_dt_s:
            old_av_dts = _np.mean(dt_ss[-n_av:-1])
            new_av_dts = _np.mean(dt_ss[-n_av + 1:])
#            _logging.debug("old_av_dts={}".format(old_av_dts))
#            _logging.debug("new_av_dts={}".format(new_av_dts))

            if _np.abs(old_av_dts - new_av_dts) < av_tol*_np.abs(new_av_dts):
                do_not_update_dt_s = True
                dt_ss.append(new_av_dts)
                _logging.debug("Stopped updating Jacobian at t={}. Using dt_s={}".format(t, dt_ss[-1]))

        if do_not_update_dt_s:
            dt_s = dt_ss[-1]
            af = 1/eta*(fun(t, y + eta * F) - F)
            n_F_evaluations += 1

        else:
            # calculate stability bound
            A = jacobian(y, force_args)
            n_A_evaluations += 1

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

    return (dt_0, dt_1, dt_2, dt_a, dt_s, inds, af, Xi_1, Xi_2,
            do_not_update_dt_s,
            EB_beneficial, n_F_evaluations, n_A_evaluations)


def _do_single_level(t, y, tf, F, dt_0, dts_local):
    # single level
    dt_0 = _np.minimum(dt_0, tf-t)
    y = y + dt_0*F
    dts_local.append(dt_0)
    return (y, dt_0)


def _do_two_levels(fun, t, y, tf, F, dt_0, dt_1, inds, min_ind_1, m0, m1,
                   dts_local, update_F):

    dt_0 = _np.minimum(dt_0, (tf-t)/m0)
    for j in range(m0*m1):
        #y[inds[:min_ind_1]] = y[inds[:min_ind_1]] + dt_0*F[inds[:min_ind_1]]
        y[inds[:min_ind_1]] += dt_0*F[inds[:min_ind_1]]
        if update_F:
            F = fun(t, y)
        dts_local.append(dt_0)

    dt_1 = _np.minimum(dt_1, (tf-t))

    #y[inds[min_ind_1:]] = y[inds[min_ind_1:]] + dt_1*F[inds[min_ind_1:]]
    y[inds[min_ind_1:]] += dt_1*F[inds[min_ind_1:]]

    return (y, dt_1)


def _do_three_levels(fun, t, y, tf, F, dt_0, dt_1, dt_2, inds, min_ind_1,
                     min_ind_2, m0, m1, dts_local, update_F):

    dt_1 = _np.minimum(dt_1, (tf-t)/m1)  # avoid overstepping at end of time interval
    for i in range(m1):
        dt_0 = _np.minimum(dt_0, (tf-t)/m0)  # avoid overstepping at end of time interval
        for j in range(m0):
            #y[inds[:min_ind_1]] = y[inds[:min_ind_1]] + dt_0*F[inds[:min_ind_1]]
            y[inds[:min_ind_1]] += dt_0*F[inds[:min_ind_1]]
            if update_F:
                F = fun(t, y)
            dts_local.append(dt_0)

        #y[inds[min_ind_1:min_ind_2]] = y[inds[min_ind_1:min_ind_2]] + dt_1*F[inds[min_ind_1:min_ind_2]]
        y[inds[min_ind_1:min_ind_2]] += dt_1*F[inds[min_ind_1:min_ind_2]]
        if update_F:
            F = fun(t, y)

    dt_2 = _np.minimum(dt_2, tf-t)
    #y[inds[min_ind_2:]] = y[inds[min_ind_2:]] + dt_2*F[inds[min_ind_2:]]
    y[inds[min_ind_2:]] += dt_2*F[inds[min_ind_2:]]

    return (y, dt_2)

def _do_local_adaptive_timestepping2(fun, t_span, y0, eps, eta,
                                     out, write_to_file,
                                     m0, jacobian, force_args,
                                     calculate_eigenvalues,
#                                     always_calculate_Jacobian,
#                                     n_av, av_tol,
                                     switch, K, dim, rA,
                                     measure_wall_time):
    _logging.debug("Using EF, local adaptive time stepping with eps={}, eta={}, m={}".format(
            eps, eta, m0))
    n_F_evaluations = 0.0
    n_A_evaluations = 0


    if measure_wall_time:
        exec_time_start = time.time()
        exec_times = []
        F_evaluations = []
        A_evaluations = []

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []
    dts_local = []
    dt_0s = []
    dt_1s = []
    dt_ss = []
    dt_as = []
    n_eq_per_level = []

#    do_not_update_dt_s = False

    while t < tf:
        if measure_wall_time:
            exec_time = time.time() - exec_time_start
            exec_times.append((t, exec_time))

        _logging.debug("t={}".format(t))

        y = copy.deepcopy(y)
        F = fun(t, y)
        n_F_evaluations += 1
        #_logging.debug("F={}".format(F))

        A = jacobian(y, force_args)
        n_A_evaluations += 1

        # choose time step adaptively locally (if we have a system of eqs)
        (dt_0, dt_1,
         dt_a,
         dt_s,
         inds, af,
         Xi_1,
         EB_beneficial) = _choose_dts2(fun, t, y, tf, F, A, eps, eta, out,
                                       write_to_file, m0,
                                       calculate_eigenvalues,
                                       K, switch)

        if EB_beneficial:
            n_eqs = _np.array([0, len(y)])
            dt = _np.minimum(dt_0, tf-t)
            dts_local.append(dt)
            _logging.debug("Switching to EB with dt_a={}, K={}".format(dt, K))
            y = eb._do_newton_iterations(fun, t, y, dt, 4, jacobian,
                                         force_args, 0.001,
                                         min(1e-4, dt), min(1e-4, dt),
                                         min(1e-4, dt))
        else:

            # find corresponding indices
            min_ind_1 = len(y0) - _np.searchsorted(abs(af[inds])[::-1], Xi_1, side='right')
            n_eqs = _np.array([min_ind_1, len(y) - min_ind_1])
            _logging.debug("i_min^1={}, dt_0={}, dt_1={}, dt_a={}, dt_s={}".format(min_ind_1, dt_0, dt_1, dt_a, dt_s))
            (dt, n_F_evaluations) = _do_levels2(fun, t, y, tf, F, A,
                                                   dt_0, dt_1, inds,
                                                   min_ind_1, m0, dts_local,
                                                   dim, rA,
                                                   n_F_evaluations)

        _logging.debug("y={}".format(y))
        t = t + dt

        ts.append(t)
        ys.append(y)
        dts.append(dt)

        dt_0s.append(dt_0)
        dt_1s.append(dt_1)

        dt_ss.append(dt_s)
        dt_as.append(dt_a)

        n_eq_per_level.append(n_eqs)

        if measure_wall_time:
            F_evaluations.append((t, n_F_evaluations))
            A_evaluations.append((t, n_A_evaluations))

    ts = _np.hstack(ts)
    ys = _np.vstack(ys).T
#    dts = _np.hstack(dts)
    dts = _np.vstack([dts, dt_as, dt_ss]).T

    dts_local = _np.hstack(dts_local)
    dt_0s = _np.hstack(dt_0s)
    dt_1s = _np.hstack(dt_1s)
    n_eq_per_level = _np.vstack(n_eq_per_level).T

    dt_0s[n_eq_per_level[0, :] == 0] = 0
    dt_1s[n_eq_per_level[1, :] == 0] = 0

    if measure_wall_time:
        with open('exec_times'+out+'.txt', 'ab') as f:
            _np.savetxt(f, exec_times)
        with open('F_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, F_evaluations)
        with open('A_evaluations'+out+'.txt', 'ab') as f:
            _np.savetxt(f, A_evaluations)

    if write_to_file:
        with open('step_sizes'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts)
        with open('step_sizes_dt_0'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dt_0s)
        with open('step_sizes_dt_1'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dt_1s)
        with open('step_sizes_local'+out+'.txt', 'ab') as f:
            _np.savetxt(f, dts_local)
        with open('n_eq_per_level'+out+'.txt', 'ab') as f:
            _np.savetxt(f, n_eq_per_level)

    return OdeResult(t=ts, y=ys)


def _choose_dts2(fun, t, y, tf, F, A, eps, eta, out, write_to_file, m0,
                 calculate_eigenvalues,
#                 always_calculate_Jacobian, do_not_update_dt_s, dt_ss,
#                 n_av, av_tol,
                 K, switch):

    EB_beneficial = False

#    # check if dt_s average has converged
#    if not always_calculate_Jacobian and len(dt_ss) >= n_av and not do_not_update_dt_s:
#        old_av_dts = _np.mean(dt_ss[-n_av:-1])
#        new_av_dts = _np.mean(dt_ss[-n_av + 1:])
##            _logging.debug("old_av_dts={}".format(old_av_dts))
##            _logging.debug("new_av_dts={}".format(new_av_dts))
#
#        if _np.abs(old_av_dts - new_av_dts) < av_tol*_np.abs(new_av_dts):
#            do_not_update_dt_s = True
##                dt_ss.append(new_av_dts)
#            _logging.debug("Stopped updating Jacobian at t={}. Using dt_s={}".format(t, dt_ss[-1]))
#
#    if do_not_update_dt_s:
#        dt_s = dt_ss[-1]
#        af = 1/eta*(fun(t, y + eta * F) - F)
#        A = None
#        n_F_evaluations += 1
#
#    else:

    # calculate stability bound
    if calculate_eigenvalues:
        #w = _np.linalg.eigvalsh(A)
        w = eigh(A, eigvals_only=True)
        _logging.debug("Eigenvalues w={}".format(w))
        #_logging.debug("Eigenvectors v={}".format(v))

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

    dt_a = _np.sqrt(2*eps*m0/Xi_0) if Xi_0 > 0.0 else tf - t

    # do not overshoot the final time
#        dt_a = _np.minimum(dt_a, tf - t)

    if dt_a > K * dt_s and switch:
        EB_beneficial = True
        dt_0 = dt_a
        dt_1 = dt_a
        Xi_1 = None
    else:
        # stick with EF
        dt_1 = _np.minimum(dt_a, dt_s)
        Xi_1 = 2.0*eps/(dt_1**2)
        dt_0 = dt_1/m0

    if write_to_file:
        with open('AFs'+out+'.txt', 'ab') as f:
            _np.savetxt(f, _np.abs(af).reshape((1, -1)))

    return (dt_0, dt_1, dt_a, dt_s, inds, af, Xi_1,
            EB_beneficial)


def _do_levels2(fun, t, y, tf, F, A, dt_0, dt_1, inds, min_ind_1, m0,
                dts_local, dim, rA, n_F_evals):

    dt_0 = _np.minimum(dt_0, (tf-t)/m0)

    # use the sparsity pattern of A to calculate perturbed indices
    pinds = _np.unique(_np.argwhere(A[inds[:min_ind_1], :] != 0)[:, 1])
    # make sure that updated equations are included
    pinds = _np.union1d(inds[:min_ind_1], pinds)

    for j in range(m0):
        y_old = copy.deepcopy(y)
        y[inds[:min_ind_1]] += dt_0*F[inds[:min_ind_1]]
        if dim is not None:
            # calculate perturbed indices
            pinds = _calculate_perturbed_indices(y, dim, rA, inds, min_ind_1)
            # subtract old force interactions between perturbed cells and add new ones
        F[pinds] += fun(t+(j+1)*dt_0, y[pinds]) - fun(t+j*dt_0, y_old[pinds])
        n_F_evals += 2*len(pinds)/float(len(y))
#        else:
#            F = fun(t+(j+1)*dt_0, y)
        dts_local.append(dt_0)

    dt_1 = _np.minimum(dt_1, (tf-t))
    y[inds[min_ind_1:]] += dt_1*F[inds[min_ind_1:]]

    return (dt_1, n_F_evals)


def _calculate_perturbed_indices(y, dim, rA, inds, min_ind_1):

    # calculate distance matrix
    y_r = _np.expand_dims(
        _np.asarray(y).reshape((-1, dim)),
        axis=-1,
        )  # shape (n, d, 1)
    cross_diff = y_r.transpose([2, 1, 0]) - y_r  # shape (n, d, n)
    norm = _np.sqrt((cross_diff**2).sum(axis=1))  # shape (n, n)

    # extract cell indices from equation indices
    updated_cell_inds = _np.unique(inds[:min_ind_1] // dim)
    # find other cells influenced by updated cells
    tmp = _np.argwhere(norm[updated_cell_inds, :] < rA)[:, 1]

    # convert back to equation indices
    # for cell j equations j*dim, j*dim + 1, ... , j*dim + dim -1 are perturbed
    tmp2 = _np.array([j*dim + d for d in range(dim) for j in tmp], dtype=int) # make sure that this stays int even if empty

    # make sure that updated equations are included (is this necessary?)
    return _np.union1d(inds[:min_ind_1], tmp2)


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

    t_eval = _np.linspace(0, 0.1, 10)
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
        os.remove('time_points.txt')
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

    rA = 1.5
    dim = 3
    y0 = _np.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 2.0, 0.0, 0.0])

    sol_full_update = solve_ivp(func, [0.0, 1.0], y0, jacobian=jacobian,
                                   local_adaptivity=True,
                                   always_calculate_Jacobian=True)
    sol_partial_update = solve_ivp(func, [0.0, 1.0], y0, jacobian=jacobian,
                                      local_adaptivity=True,
                                      always_calculate_Jacobian=True,
                                      dim=dim, rA=rA)


    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None,
                     eps=0.0001, eta = 0.00001, local_adaptivity=True,
                     write_to_file=True, jacobian=jacobian,
#                     always_calculate_Jacobian=True,
                     switch=False, K=3, dim=1)
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
    plt.plot(sol2.t[:-1], dt[:, 0])
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
