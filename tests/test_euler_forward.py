#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import os

import cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.cell as cl


@np.vectorize
def func(t, y):
    return -50*y

def jacobian(y, fa):
    return -50*np.eye(len(y))


def test_y_shape():

    t_span = (0, 1)
    y0 = np.array([1, 1])

    sol = ef.solve_ivp(func, t_span, y0, dt=0.01)

    assert len(sol.t) == 101
    assert sol.y.shape == (2, 101)


def test_t_eval():
    t_eval = np.linspace(0, 1, 10)
    y0 = np.array([1, -1])

    sol = ef.solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval,
                       dt=0.01)
    assert len(sol.t) == len(t_eval)

    t_eval = np.linspace(0, 1, 1000)
    y0 = np.array([1, -1])

    sol = ef.solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval,
                       dt=0.01)
    assert len(sol.t) == len(t_eval)


def test_no_overstep():
    t_span = (0, 1)
    y0 = np.array([1, 1])

    # fixed time step
    sol = ef.solve_ivp(func, t_span, y0, dt=0.03)
    assert sol.t[-1] == t_span[1]

    # global adaptivity
    sol = ef.solve_ivp(func, t_span, y0)
    assert sol.t[-1] == t_span[1]

    # global adaptivity with stability bound
    sol = ef.solve_ivp(func, t_span, y0, jacobian=jacobian)
    assert sol.t[-1] == t_span[1]

    # local adaptivity
    sol = ef.solve_ivp(func, t_span, y0, local_adaptivity=True)
    assert sol.t[-1] == t_span[1]

    # local adaptivity with stability bound
    sol = ef.solve_ivp(func, t_span, y0, jacobian=jacobian,
                       local_adaptivity=True)
    assert sol.t[-1] == t_span[1]


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


def test_ordering_ts_when_using_global_adaptivity():
    # Simulation parameters
    tf = 10.0  # final time
    dim = 2
    seed=67

    dt = 0.05
    t_data = np.arange(0, tf, dt)

    # Solvers
    model = cbmos.CBModel(ff.Gls(), ef.solve_ivp, dim)
    mu_gls = 1.95
    params = {'mu': mu_gls, 'a': -2*np.log(0.002/mu_gls)}

    npr.seed(seed)

    cell_list = [
            cl.Cell(
                0, [0., 0.],
                proliferating=True,
                division_time_generator=lambda t: npr.exponential(4.0) + t)
            ]

    ts, history = model.simulate(cell_list, t_data, params,
                                 {"eps": 0.05, "eta": 0.0001,
                                  'write_to_file': True}, seed=seed)
    assert(np.all(np.diff(ts) >= 0))


def test_fix_eqs():
    def func(t, y):
        return -50*np.eye(len(y))@y
    def jacobian(y, fa):
        return -50*np.eye(len(y))

    t_eval = np.linspace(0, 3, 10)
    y0 = np.array([0.5, 1.0, 3.5, 6.0])

    sol = ef.solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=None,
                       jacobian=jacobian, fix_eqs=2)

    assert(np.all(sol.y[:2, :].T == y0[:2]))


def test_measure_wall_time_fixed_timestepping():

    try:
        os.remove('exec_times.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    t_span = [0, 1]
    y0 = np.array([1, 1])

    _ = ef.solve_ivp(func, t_span, y0, dt=0.01, measure_wall_time=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)

    assert len(exec_times) == 100


def test_measure_wall_time_global_adaptivity():

    try:
        os.remove('exec_times.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    t_span = [0, 1]
    y0 = np.array([1, 1])

    _ = ef.solve_ivp(func, t_span, y0, measure_wall_time=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)

    # crude check that list exists and is written to the file
    assert exec_times[0][0] == 0.0


def test_measure_wall_time_global_adaptivity_with_stab():

    try:
        os.remove('exec_times.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    def func(t, y):
        return -50*np.eye(len(y))@y

    def jacobian(y, fa):
        return -50*np.eye(len(y))

    t_span = [0, 1]
    y0 = np.array([1, 1])

    _ = ef.solve_ivp(func, t_span, y0, jacobian=jacobian, measure_wall_time=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)

    # crude check that list exists and is written to the file
    assert exec_times[0][0] == 0.0


def test_measure_wall_time_local_adaptivity():

    try:
        os.remove('exec_times.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    def func(t, y):
        return -50*np.eye(len(y))@y

    def jacobian(y, fa):
        return -50*np.eye(len(y))

    t_span = [0, 1]
    y0 = np.array([1, 1])

    _ = ef.solve_ivp(func, t_span, y0, jacobian=jacobian,
                     local_adaptivity=True,
                     measure_wall_time=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)

    # crude check that list exists and is written to the file
    assert exec_times[0][0] == 0.0
