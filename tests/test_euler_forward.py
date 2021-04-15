#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import cbmos.solvers.euler_forward as ef


@np.vectorize
def func(t, y):
    return -50*y


def test_y_shape():

    t_span = (0, 1)
    y0 = np.array([1, 1])

    sol = ef.solve_ivp(func, t_span, y0, dt=0.01)

    assert len(sol.t) == 101
    assert sol.y.shape == (2, 101)

def test_t_eval():
    t_eval = np.linspace(0,1,10)
    y0 = np.array([1, -1])

    sol = ef.solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, dt=0.01)
    assert len(sol.t) == len(t_eval)

    t_eval = np.linspace(0,1,1000)
    y0 = np.array([1, -1])

    sol = ef.solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, dt=0.01)
    assert len(sol.t) == len(t_eval)

def test_jacobian_arguments():
    # test that local adaptivity code in EF can handle the correct signature
    # of the Jacobian with the force function arguments
    def func(t, y):
        return -50*np.eye(len(y))@y
    def jacobian(y, fa):
        return -50*np.eye(len(y))

    t_eval = np.linspace(0,3,10)
    y0 = np.array([0.5, 1.0])

    sol = ef.solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None,
                       local_adaptivity=True, jacobian=jacobian)