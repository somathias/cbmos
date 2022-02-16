#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import os
import copy

import cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.cell as cl
import cbmos.events as ev

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
    y0 = np.array([1.0, -1.0])

    sol = ef.solve_ivp(func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval,
                       dt=0.01)
    assert len(sol.t) == len(t_eval)

    t_eval = np.linspace(0, 1, 1000)
    y0 = np.array([1.0, -1.0])

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

    # local adaptivity with stability bound
    sol = ef.solve_ivp(np.vectorize(func, otypes=['float']), t_span, y0, jacobian=jacobian,
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
            cl.ProliferatingCell(
                0, [0., 0.],
                proliferating=True,
                division_time_generator=lambda t: npr.exponential(4.0) + t)
            ]
    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    ts, history = model.simulate(cell_list, t_data, params,
                                 {"eps": 0.05, "eta": 0.0001,
                                  'write_to_file': True}, seed=seed,
                                  event_list=event_list)
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
    try:
        os.remove('F_evaluations.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    t_span = [0, 1]
    y0 = np.array([1, 1])

    _ = ef.solve_ivp(func, t_span, y0, dt=0.01, measure_wall_time=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)
    with open('F_evaluations.txt', 'r') as f:
        F_evaluations = np.loadtxt(f)

    assert len(exec_times) == 100
    assert F_evaluations[-1][1] == 100


def test_measure_wall_time_global_adaptivity():

    try:
        os.remove('exec_times.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    t_span = [0, 1]
    y0 = np.array([1, 1])

    sol = ef.solve_ivp(func, t_span, y0, measure_wall_time=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)
    with open('F_evaluations.txt', 'r') as f:
        F_evaluations = np.loadtxt(f)

    # crude check that list exists and is written to the file
    assert exec_times[0][0] == 0.0
    assert F_evaluations[-1][1] == 2*(len(sol.t)-1)


def test_measure_wall_time_global_adaptivity_with_stab():

    try:
        os.remove('exec_times.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('A_evaluations.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    def func(t, y):
        return -50*np.eye(len(y))@y

    def jacobian(y, fa):
        return -50*np.eye(len(y))

    t_span = [0, 1]
    y0 = np.array([1, 1])

    sol = ef.solve_ivp(func, t_span, y0, jacobian=jacobian, measure_wall_time=True, always_calculate_Jacobian=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)
    with open('F_evaluations.txt', 'r') as f:
        F_evaluations = np.loadtxt(f)
    with open('A_evaluations.txt', 'r') as f:
        A_evaluations = np.loadtxt(f)


    # crude check that list exists and is written to the file
    assert exec_times[0][0] == 0.0
    assert F_evaluations[-1][1] == (len(sol.t)-1)
    assert A_evaluations[-1][1] == (len(sol.t)-1)



def test_measure_wall_time_local_adaptivity():

    try:
        os.remove('exec_times.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('A_evaluations.txt')
    except FileNotFoundError:
        print('Nothing to delete.')

    def func(t, y):
        return -50*np.eye(len(y))@y

    def jacobian(y, fa):
        return -50*np.eye(len(y))

    t_span = [0, 1]
    y0 = np.array([1, 1])

    sol = ef.solve_ivp(func, t_span, y0, jacobian=jacobian,
                     local_adaptivity=True,
                     always_calculate_Jacobian=True,
                     measure_wall_time=True)

    with open('exec_times.txt', 'r') as f:
        exec_times = np.loadtxt(f)
    with open('F_evaluations.txt', 'r') as f:
        F_evaluations = np.loadtxt(f)
    with open('A_evaluations.txt', 'r') as f:
        A_evaluations = np.loadtxt(f)

    # crude check that list exists and is written to the file
    assert exec_times[0][0] == 0.0
    assert F_evaluations[-1][1] >= (len(sol.t)-1)
    assert A_evaluations[-1][1] == (len(sol.t)-1)


def test_calculate_perturbed_indices1D():

    rA = 1.5
    dim = 1
    y = np.array([0.0, 0.75, 2.0])

    inds = np.arange(len(y))
    min_ind_1 = 1

    pinds = ef._calculate_perturbed_indices(y, dim, rA, inds, min_ind_1)

    assert np.all(pinds == [0, 1])


def test_calculate_perturbed_indices2D():

    rA = 1.5
    dim = 2
    y = np.array([0.0, 0.0, 0.75, 0.0, 2.0, 0.0])

    inds = np.arange(len(y))
    min_ind_1 = 1

    pinds = ef._calculate_perturbed_indices(y, dim, rA, inds, min_ind_1)

    assert np.all(pinds == [0, 1, 2, 3])


def test_calculate_perturbed_indices3D():

    rA = 1.5
    dim = 3
    y = np.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 2.0, 0.0, 0.0])

    inds = np.arange(len(y))
    min_ind_1 = 1

    pinds = ef._calculate_perturbed_indices(y, dim, rA, inds, min_ind_1)

    assert np.all(pinds == [0, 1, 2, 3, 4, 5])


def test_calculate_perturbed_indices_min_ind_1_is_zero():

    rA = 1.5
    dim = 3
    y = np.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 2.0, 0.0, 0.0])

    inds = np.arange(len(y))
    min_ind_1 = 0

    pinds = ef._calculate_perturbed_indices(y, dim, rA, inds, min_ind_1)

    assert pinds.size == 0


def test_calculate_perturbed_indices_min_ind_1_is_length_of_y():

    rA = 1.5
    dim = 3
    y = np.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 2.0, 0.0, 0.0])

    inds = np.arange(len(y))
    min_ind_1 = len(y)

    pinds = ef._calculate_perturbed_indices(y, dim, rA, inds, min_ind_1)

    assert np.all(pinds == inds)


def test_calculate_perturbed_indices_using_A():

    dim = 2
    s = 1.0
    rA = 1.5
    model = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim )
    params_cubic = {"mu": 5.70, "s": s, "rA": rA}
    m0 = 2



    y0 = np.array([0.        , 0.        , 0.5       , 0.8660254 , 1.        ,
           0.        , 1.64381812, 0.90864406, 1.35618188, 0.82340675])

    inds = np.array([8, 6, 9, 7, 2, 5, 4, 3, 1, 0])
    min_ind_1 = 2

    F = model._ode_system(params_cubic)(0, y0)
    A = model.jacobian(y0, params_cubic)
    dt_0 = 0.007
    dt_1 = 0.014

    nF = 0
    y1 = copy.deepcopy(y0)
    F1 = copy.deepcopy(F)
    (dt, nF1) = ef._do_levels2(model._ode_system(params_cubic), 0, y1, 1.0 , F1, A, dt_0, dt_1, inds,
                                 min_ind_1, m0, [], None, rA, nF)

    y2 = copy.deepcopy(y0)
    F2 = copy.deepcopy(F)
    (dt2, nF2) = ef._do_levels2(model._ode_system(params_cubic), 0, y2, 1.0 , F2, A, dt_0, dt_1, inds,
                                 min_ind_1, m0, [], dim, rA, nF)

    # assert that both solutions the same
    assert(np.all(y1 == y2))
    assert(np.all(F1 == F2))

    # take another step
    y0 = copy.deepcopy(y1)
    F = model._ode_system(params_cubic)(0, y0)
    A = model.jacobian(y0, params_cubic)
    dt_0 = 0.0101531966512
    dt_1 = 0.0203063933902

    inds = np.array([8, 6, 9, 7, 2, 5, 3, 4, 1, 0])
    min_ind_1 = 2

    nF = 0
    y1 = copy.deepcopy(y0)
    F1 = copy.deepcopy(F)
    (dt, nF1) = ef._do_levels2(model._ode_system(params_cubic), 0, y1, 1.0 , F1, A, dt_0, dt_1, inds,
                                 min_ind_1, m0, [], None, rA, nF)

    y2 = copy.deepcopy(y0)
    F2 = copy.deepcopy(F)
    (dt2, nF2) = ef._do_levels2(model._ode_system(params_cubic), 0, y2, 1.0 , F2, A, dt_0, dt_1, inds,
                                 min_ind_1, m0, [], dim, rA, nF)

    assert(np.allclose(y1, y2, 1e-5, 1e-5))
    assert(np.allclose(F1, F2, 1e-3, 1e-3))


#def test_perturbed_eqs():
#
#    dim = 3
#    s = 1.0
#    rA = 1.5
#    model = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim )
#    params_cubic = {"mu": 5.70, "s": s, "rA": rA}
#
#
#    y0 = np.array([0.        , 0.        , 0.        , 0.5       , 0.28867513,
#       0.81649658, 0.        , 0.        , 1.63299316, 0.5       ,
#       0.8660254 , 0.        , 0.        , 1.15470054, 0.81649658,
#       0.5       , 0.8660254 , 1.63299316, 0.        , 1.73205081,
#       0.        , 0.5       , 2.02072594, 0.81649658, 0.        ,
#       1.73205081, 1.63299316, 1.        , 0.        , 0.        ,
#       1.5       , 0.28867513, 0.81649658, 1.        , 0.        ,
#       1.63299316, 1.5       , 0.8660254 , 0.        , 1.10014613,
#       1.18437756, 0.7088396 , 1.5       , 0.8660254 , 1.63299316,
#       1.        , 1.73205081, 0.        , 1.5       , 2.02072594,
#       0.81649658, 1.        , 1.73205081, 1.63299316, 2.        ,
#       0.        , 0.        , 2.5       , 0.28867513, 0.81649658,
#       2.        , 0.        , 1.63299316, 2.5       , 0.8660254 ,
#       0.        , 2.        , 1.15470054, 0.81649658, 2.5       ,
#       0.8660254 , 1.63299316, 2.        , 1.73205081, 0.        ,
#       2.5       , 2.02072594, 0.81649658, 2.        , 1.73205081,
#       1.63299316, 0.89985387, 1.12502352, 0.92415356])
#
#
#    sol_full_update = ef.solve_ivp(model._ode_system(params_cubic),
#                                   [0.0, 1.0],
#                                   y0, eps=0.001, jacobian=model.jacobian,
#                                   force_args=params_cubic,
#                                   local_adaptivity=True
#                                   )
#    sol_partial_update = ef.solve_ivp(model._ode_system(params_cubic),
#                                      [0.0, 1.0], y0,
#                                      eps=0.001, jacobian=model.jacobian,
#                                      force_args=params_cubic,
#                                      local_adaptivity=True,
#                                      dim=dim, rA=rA)
#
#    assert(np.all(sol_full_update.t == sol_partial_update.t))
#    assert np.allclose(sol_full_update.y, sol_partial_update.y, rtol=1e-3, atol=1e-3)
