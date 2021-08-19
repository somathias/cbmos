#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import cg
import scipy as scpi
import copy
from scipy.sparse.linalg import LinearOperator
import logging as _logging


import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=0.1, eps=0.01, eta=0.001,
              n_newton=20, eps_newton=None, eps_max =1e-3, xi=0.001,
              jacobian=None, force_args={}, tol=None, atol=None,
              out='', write_to_file=False, disp=False):

    class gmres_counter(object):
        def __init__(self, disp=disp):
            self._disp = disp
            self.niter = 0
        def __call__(self, rk=None):
            self.niter += 1
            if self._disp:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))

    counter = gmres_counter()


    adaptive_dt = True if dt is None else False

    if adaptive_dt:
        # choose time step adaptively globally
       return _do_global_adaptive_timestepping(fun, t_span, y0, t_eval,
                                                    dt, eps, eta, n_newton,
                                                    eps_newton, xi, jacobian,
                                                    force_args, counter,
                                                    eps_max, tol,
                                                    atol, out, write_to_file)

    else:
        # do regular fixed time stepping
        if eps_newton is None:
            eps_newton = min(eps_max, dt)
        if tol is None:
            tol = min(eps_max, dt)
        if atol is None:
            atol = min(eps_max, dt)
        return _do_fixed_timestepping(fun, t_span, y0, t_eval, dt,
                                      n_newton, eps_newton, xi, jacobian,
                                      force_args, counter, tol, atol, out,
                                      write_to_file)

def _do_fixed_timestepping(fun, t_span, y0, t_eval, dt, n_newton,
                           eps_newton, xi, jacobian, force_args, counter,
                           tol, atol, out, write_to_file):
    _logging.debug("Using EB, fixed time stepping with dt={}".format(
            dt))

    # do regular fixed time stepping
    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []

    while t < tf:

        # take minimum of dt and tf-t in order to not overstep
        dt = np.minimum(dt, tf-t)

        _logging.debug("t={}".format(t))

        y = copy.deepcopy(y)

        # do Newton iterations
        y_next = copy.deepcopy(y)  # initialize with current y
        for j in np.arange(n_newton):

            F_curly = y_next - y - dt*fun(t, y_next)

            if jacobian is not None:
                A = jacobian(y_next, force_args)
                J = np.eye(A.shape[0]) - dt*A
            else:
                # approximate matrix vector product Jv where J = I-dt*A
                def Jv(v):
                    return 1/xi*(y_next + xi*v
                                  - y - dt*fun(t, y_next + xi*v)
                                  - F_curly)
                J = LinearOperator((len(y_next), len(y_next)), matvec=Jv)

            # solve linear system J*dy = F_curly for dy
            dy, exitCode = gmres(J, -F_curly, callback=counter, tol=tol,
                                 atol=atol, restart=5, maxiter=1,
                                 callback_type='x') # maxiter= number of outer iterations/restarts, restart= number of inner iterations (between restarts)
            _logging.debug("Number of GMRes iterations = {}".format(counter.niter))
            #dy, exitCode = lgmres(J, -F_curly, callback=counter)
            #dy, exitCode = cg(J, -F_curly)
            # dy = scpi.linalg.solve(J, -F_curly)
            #print('ExitCode='+str(exitCode))
            y_next = y_next + dy

            if np.linalg.norm(dy)/np.linalg.norm(y_next) < eps_newton:
                _logging.debug("Relative error tolerance of {} achieved.".format(eps_newton))
                break

        y = copy.deepcopy(y_next)
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

def _do_global_adaptive_timestepping(fun, t_span, y0, t_eval, dt, eps, eta,
                                     n_newton, eps_newton, xi, jacobian,
                                     force_args, counter, eps_max, tol, atol,
                                     out, write_to_file):
    _logging.debug("Using EB, adaptive time stepping.")

    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []

    while t < tf:

        _logging.debug("t={}".format(t))

        y = copy.deepcopy(y)

        F = fun(t, y)
        if jacobian is not None:
            A = jacobian(y, force_args)
            AF = A@F
        else:
            AF = 1/eta*(fun(t, y + eta * F) - F)

        if write_to_file:
            with open('AFs'+out+'.txt', 'ab') as f:
                np.savetxt(f, np.abs(AF).reshape((1, -1)))

        norm_AF = np.linalg.norm(AF, np.inf)
        dt = np.sqrt(2*eps/norm_AF) if norm_AF > 0.0 else tf - t

        # take minimum of dt and tf-t in order to not overstep
        dt = np.minimum(dt, tf-t)

        if eps_newton is None:
            eps_newton = min(eps_max, dt)
        if tol is None:
            tol = min(eps_max, dt)
        if atol is None:
            atol = min(eps_max, dt)

        # do Newton iterations
        y_next = copy.deepcopy(y)  # initialize with current y
        for j in np.arange(n_newton):

            F_curly = y_next - y - dt*fun(t, y_next)

            if jacobian is not None:
                A = jacobian(y_next, force_args)
                J = np.eye(A.shape[0]) - dt*A
            else:
                # approximate matrix vector product Jv where J = I-dt*A
                def Jv(v):
                    return 1/xi*(y_next + xi*v
                                  - y - dt*fun(t, y_next + xi*v)
                                  - F_curly)
                J = LinearOperator((len(y_next), len(y_next)), matvec=Jv)

            # solve linear system J*dy = F_curly for dy
            dy, exitCode = gmres(J, -F_curly, callback=counter, tol=tol,
                                 atol=atol, restart=5, maxiter=1,
                                 callback_type='x') # maxiter= number of outer iterations/restarts, restart= number of inner iterations (between restarts)
            _logging.debug("Number of GMRes iterations = {}".format(counter.niter))
            #dy, exitCode = lgmres(J, -F_curly, callback=counter)
            #dy, exitCode = cg(J, -F_curly)
            # dy = scpi.linalg.solve(J, -F_curly)
            #print('ExitCode='+str(exitCode))
            y_next = y_next + dy

            if np.linalg.norm(dy)/np.linalg.norm(y_next) < eps:
                _logging.debug("Relative error tolerance of {} achieved.".format(eps))
                break

        y = copy.deepcopy(y_next)
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

if __name__ == "__main__":

    # stability region for Euler forward for this problem is h<2/50=0.04
    #@np.vectorize
    def func(t, y):
        return -50*np.eye(len(y))@y
    def jac(y, r=1):
        return -50*np.eye(len(y))

#    t_span = (0,1)
#    y0 = np.array([1,1])
#
#    sol = solve_ivp(func, t_span, y0 )
#
#    plt.figure()
#    plt.plot(sol.t, sol.y)

    t_eval = np.linspace(0,1,10)
    y0 = np.array([1.0, 2.0])
    #y0 = np.array([0.5, 0.7, 1.0, 3.0])
    #y0 = np.array([0.0, 0.0, 0.0])

    try:
        os.remove('step_sizes.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    #sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, dt=0.1, n_newton = 2,
    #                 write_to_file=True, jacobian=jac)
    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, dt=None, n_newton = 2,
                     write_to_file=True)
    #plt.plot(sol2.t, sol2.y.T)
    plt.plot(sol2.t, sol2.y.T, '*')
    plt.xlabel('t')
    plt.ylabel('y')

    plt.figure()
    dt  = np.loadtxt('step_sizes.txt')
    plt.plot(np.cumsum(dt), dt)
    plt.xlabel('time')
    plt.ylabel('Global step size')
