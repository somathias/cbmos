#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 2019

Unit tests of the CBM solver

@author: Adrien Coulier
"""
import pytest
import numpy as np
import cbmos_serial as cbmos
import scipy.integrate as scpi
import logging
import io

import force_functions as ff
import euler_forward as ef
import heapq as hq
import cell as cl

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def two_cells():
    return np.array([[0., 0., 0.], [0.5, 0., 0.]]).reshape(-1)


def test_ode_force(two_cells):
    # Check that the force is made use of
    solver = cbmos.CBMSolver(lambda r: 0., scpi.solve_ivp)
    f = solver._ode_force({})
    assert not f(0., two_cells).any()
    solver = cbmos.CBMSolver(lambda r: 1., scpi.solve_ivp)
    f = solver._ode_force({})
    assert f(0., two_cells).any()

    # Check parameters are made use of
    solver = cbmos.CBMSolver(lambda r, p=1.: p, scpi.solve_ivp)
    f = solver._ode_force({})
    assert f(0., two_cells).any()
    f = solver._ode_force({'p': 0.})
    assert not f(0., two_cells).any()

    # Check force computation is correct
    solver = cbmos.CBMSolver(lambda r: 1., scpi.solve_ivp)
    f = solver._ode_force({})
    total_force = f(0., two_cells).reshape(2, 3)
    assert np.array_equal(total_force[0], np.array([1., 0., 0.]))
    assert np.array_equal(total_force[0], -total_force[1])


def test_calculate_positions(two_cells):
    for dim in [1, 2, 3]:
        cbm_solver = cbmos.CBMSolver(ff.linear, ef.solve_ivp, dim)

        T = np.linspace(0, 10, num=10)

        # Check cells end up at the resting length
        for s in [1., 2., 3.]:
            sol = cbm_solver._calculate_positions(
                    T,
                    two_cells.reshape(-1, 3)[:, :dim].reshape(-1),
                    {'s': s, 'mu': 1.0},
                    {'dt': 0.1}
                    ).y.reshape(-1, 2*dim)
            assert np.abs(sol[-1][:dim] - sol[-1][dim:]).sum() - s < 0.01


def test_build_event_queue():
    """
    Note
    ----
    Assumes the event list is using heapq. This test will break if we change
    data structure
    """
    dim = 3
    cbm_solver = cbmos.CBMSolver(ff.linear, ef.solve_ivp, dim)

    cells = [cl.Cell(i, [0, 0, i]) for i in range(5)]
    for i, cell in enumerate(cells):
        cell.division_time = cell.ID

    cbm_solver.cell_list = cells
    cbm_solver._build_event_queue()

    for i in range(5):
        assert hq.heappop(cbm_solver.event_queue)[1].ID == i


def test_update_event_queue():
    solver = cbmos.CBMSolver(lambda r: 0., scpi.solve_ivp)
    cell = cl.Cell(0, np.zeros((1, 3)))

    solver.event_queue = []
    solver._update_event_queue(cell)
    # check that event gets added
    assert len(solver.event_queue) == 1
    # check that it's the correct event
    event = solver.event_queue[0]
    assert event[0] == cell.division_time
    assert event[1] == cell
    # add more cells
    cell2 = cl.Cell(1, np.zeros((1, 3))+0.25)
    cell3 = cl.Cell(2, np.zeros((1, 3))+0.5)
    solver._update_event_queue(cell2)
    solver._update_event_queue(cell3)
    assert len(solver.event_queue) == 3
    # check that sorting is correct
    assert solver.event_queue[0][0] <= solver.event_queue[1][0]
    assert solver.event_queue[1][0] <= solver.event_queue[2][0]


def test_get_next_event():
    solver = cbmos.CBMSolver(lambda r: 0., scpi.solve_ivp)

    cell_list = [cl.Cell(i, np.array([0, 0, i]), 0.0, True) for i in [0, 1, 2]]
    solver.cell_list = cell_list
    solver._build_event_queue()
    assert len(solver.event_queue) == 3

    solver._get_next_event()
    assert len(solver.event_queue) == 2

    solver._get_next_event()
    assert len(solver.event_queue) == 1

    solver._get_next_event()
    assert len(solver.event_queue) == 0


def test_get_division_direction():
    for dim in [1, 2, 3]:
        cbm_solver = cbmos.CBMSolver(lambda r: 0., ef.solve_ivp, dim)

        mean_division_direction = cbm_solver._get_division_direction()
        assert mean_division_direction.shape == (dim,)
        assert np.isclose(np.linalg.norm(mean_division_direction), 1)

#        N = 1000
#        for i in range(N):
#            mean_division_direction += cbm_solver._get_division_direction()
#        assert np.all(abs(mean_division_direction/N) < 1)


def test_apply_division():
    dim = 3
    cbm_solver = cbmos.CBMSolver(ff.linear, ef.solve_ivp, dim)

    cell_list = [cl.Cell(i, [0, 0, i], proliferating=True) for i in range(5)]
    for i, cell in enumerate(cell_list):
        cell.division_time = cell.ID

    cbm_solver.cell_list = cell_list
    cbm_solver.next_cell_index = 5
    cbm_solver._build_event_queue()

    cbm_solver._apply_division(cell_list[0], 1)

    assert len(cbm_solver.cell_list) == 6
    assert cbm_solver.cell_list[5].ID == 5
    assert cbm_solver.next_cell_index == 6
    assert cell_list[0].division_time != 0

    assert np.isclose(
            np.linalg.norm(cell_list[0].position - cell_list[5].position),
            cbm_solver.separation)


def test_update_positions():
    dim = 3
    cbm_solver = cbmos.CBMSolver(ff.linear, ef.solve_ivp, dim)

    cell_list = [cl.Cell(i, [0, 0, i]) for i in range(5)]
    for i, cell in enumerate(cell_list):
        cell.division_time = cell.ID

    cbm_solver.cell_list = cell_list

    new_positions = [[0, i, i] for i in range(5)]
    cbm_solver._update_positions(new_positions)
    for i, cell in enumerate(cbm_solver.cell_list):
        print(cell.position)
        assert cell.position.tolist() == [0, i, i]


def test_simulate():
    dim = 1
    cbm_solver = cbmos.CBMSolver(ff.cubic, scpi.solve_ivp, dim)
    cell_list = [cl.Cell(0, [0]), cl.Cell(1, [1.0], 0.0, True)]
    cell_list[1].division_time = 1.05  # make sure not to divide at t_data

    N = 100
    t_data = np.linspace(0, 10, N) # stay away from 24 hours
    history = cbm_solver.simulate(cell_list, t_data, {}, {})

    assert len(history) == N

    assert len(history[10]) == 2
    assert np.isclose(abs(history[10][0].position - history[10][1].position), 1)

    assert len(history[-1]) == 3
    scells = sorted(history[-1], key=lambda c:c.position)
    assert np.isclose(abs(scells[0].position - scells[1].position), 1, atol=1e-03)
    assert np.isclose(abs(scells[1].position - scells[2].position), 1, atol=1e-03)

def test_two_events_at_once():
    dim = 1
    cbm_solver = cbmos.CBMSolver(ff.linear, scpi.solve_ivp, dim)
    cell_list = [cl.Cell(0, [0], proliferating=True), cl.Cell(1, [1.0], 0.0, True)]
    cell_list[0].division_time = 1.05
    cell_list[1].division_time = 1.05

    t_data = np.linspace(0, 10, 100)
    history = cbm_solver.simulate(cell_list, t_data, {}, {})

    assert len(history) == 100

def test_event_at_t_data():
    dim = 1
    cbm_solver = cbmos.CBMSolver(ff.linear, scpi.solve_ivp, dim)
    cell_list = [cl.Cell(0, [0], proliferating=True), cl.Cell(1, [1.0], 0.0, True)]
    cell_list[0].division_time = 1.0
    cell_list[1].division_time = 1.0

    t_data = np.linspace(0, 10, 101)
    history = cbm_solver.simulate(cell_list, t_data, {}, {})

    assert len(history) == len(t_data)

def test_no_division_skipped():

    dim = 1
    cbm_solver = cbmos.CBMSolver(ff.linear, scpi.solve_ivp, dim)
    cell_list = [cl.Cell(0, [0], proliferating=True), cl.Cell(1, [1.0], 0.0, True)]
    cell_list[0].division_time = 1.0
    cell_list[1].division_time = 1.0

    t_data = np.linspace(0, 30, 101)
    history = cbm_solver.simulate(cell_list, t_data, {}, {})

    eq = [hq.heappop(cbm_solver.event_queue) for i in range(len(cbm_solver.event_queue))]
    assert eq == sorted(eq)

    assert len(eq) == len(history[-1]) - 1

    for t, cells in zip(t_data, history):
        for c in cells:
            assert c.birthtime <= t
            assert c.division_time > t

def test_cell_list_copied():

    dim = 1
    cbm_solver_one = cbmos.CBMSolver(ff.linear, scpi.solve_ivp, dim)
    cbm_solver_two = cbmos.CBMSolver(ff.linear, scpi.solve_ivp, dim)

    cell_list = [cl.Cell(0, [0], proliferating=True), cl.Cell(1, [0.3], proliferating=True)]
    t_data = np.linspace(0, 1, 101)

    history_one = cbm_solver_one.simulate(cell_list, t_data, {}, {})
    history_two = cbm_solver_two.simulate(cell_list, t_data, {}, {})

    assert history_two[0][0].position == np.array([0])
    assert history_two[0][1].position == np.array([0.3])

def test_tdata():
    n = 100

    s = 1.0    # rest length
    tf = 1.0  # final time
    rA = 1.5   # maximum interaction distance

    params_cubic = {"mu": 6.91, "s": s, "rA": rA}

    solver_ef = cbmos.CBMSolver(ff.cubic, ef.solve_ivp, 1)
    t_data = np.linspace(0,1, n)
    cell_list = [cl.Cell(0, [0], proliferating=False), cl.Cell(1, [0.3], proliferating=False)]
    sols = solver_ef.simulate(cell_list, t_data, params_cubic, {'dt': 0.03})
    y = np.array([np.squeeze([clt[0].position, clt[1].position]) for clt in sols])

    assert y.shape == (n, 2)

def test_sparse_tdata():
    dim = 3
    solver_cubic = cbmos.CBMSolver(ff.cubic, ef.solve_ivp, dim)
    ancestor = [cl.Cell(0, np.zeros((dim,)), -5, True)]
    dt = 0.1
    t_f = 50
    t_data = np.linspace(0, t_f, 2) 
    tumor_cubic = solver_cubic.simulate(ancestor, t_data, {"mu":6.91}, {"dt":dt})

def test_seed():
    dim = 3
    cbm_solver = cbmos.CBMSolver(ff.logarithmic, ef.solve_ivp, dim)

    cell_list = [cl.Cell(0, [0, 0, 0], proliferating=True)]
    t_data = np.linspace(0, 100, 10)
    histories = [
            cbm_solver.simulate(cell_list, t_data, {}, {}, seed=seed)
            for seed in [0, 0, 1, None, None]]

    for cells in zip(*[history[-1] for history in histories]):
        assert cells[0].position.tolist() == cells[1].position.tolist()
        assert cells[0].position.tolist() != cells[2].position.tolist()
        assert cells[3].position.tolist() != cells[4].position.tolist()

def test_seed_division_time():
    logger = logging.getLogger()
    logs = io.StringIO()
    logger.addHandler(logging.StreamHandler(logs))

    dim = 3
    cbm_solver = cbmos.CBMSolver(ff.logarithmic, ef.solve_ivp, dim)

    cell_list = [cl.Cell(0, [0, 0, 0], proliferating=True)]
    t_data = np.linspace(0, 100, 10)

    history = [
            cbm_solver.simulate(cell_list, t_data, {}, {}, seed=seed)
            for seed in [0, 0, 1]]

    division_times = logs.getvalue().split("Starting new simulation\n")[1:]
    assert division_times[0] == division_times[1]
    assert division_times[0] != division_times[2]
