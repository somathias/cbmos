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


import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')


def solve_ivp(fun, t_span, y0, t_eval=None, dt=0.1, n_newton=20,
              eps=None, eps_max =1e-3, eta=0.001, jacobian=None, force_args={},
              tol=None, atol=None,
              out='', write_to_file=False, disp=False):

    if eps is None:
        eps = min(eps_max, dt)
    if tol is None:
        tol = min(eps_max, dt)
    if atol is None:
        atol = min(eps_max, dt)


    class gmres_counter(object):
        def __init__(self, disp=disp):
            self._disp = disp
            self.niter = 0
        def __call__(self, rk=None):
            self.niter += 1
            if self._disp:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))

    # do regular fixed time stepping
    t0, tf = float(t_span[0]), float(t_span[-1])

    t = t0
    y = y0

    ts = [t]
    ys = [y]
    dts = []

    while t < tf:
        if disp:
            print('--------')
            print('t = '+str(t))
            print('--------')

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
                    return 1/eta*(y_next + eta*v
                                  - y - dt*fun(t, y_next + eta*v)
                                  - F_curly)
                J = LinearOperator((len(y_next), len(y_next)), matvec=Jv)

            # solve linear system J*dy = F_curly for dy
            counter = gmres_counter()
            dy, exitCode = gmres(J, -F_curly, callback=counter, tol=tol,
                                 atol=atol, restart=5, maxiter=1,
                                 callback_type='x') # maxiter= number of outer iterations/restarts, restart= number of inner iterations (between restarts)
            if disp:
                print('Number of GMRes iterations = '+str(counter.niter))
            #dy, exitCode = lgmres(J, -F_curly, callback=counter)
            #dy, exitCode = cg(J, -F_curly)
            # dy = scpi.linalg.solve(J, -F_curly)
            #print('ExitCode='+str(exitCode))
            y_next = y_next + dy

            if np.linalg.norm(dy)/np.linalg.norm(y_next) < eps:
                if disp:
                    print('Relative error tolerance of '+str(eps)+' achieved.')
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
        with open('time_points'+out+'.txt', 'ab') as f:
            np.savetxt(f, ts)
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
    try:
        os.remove('time_points.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    #sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, dt=0.1, n_newton = 2,
    #                 write_to_file=True, jacobian=jac)
    sol2 = solve_ivp(func, [t_eval[0], t_eval[-1]], y0, dt=0.1, n_newton = 2,
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
