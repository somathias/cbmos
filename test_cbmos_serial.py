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

import force_functions as ff
import euler_forward as ef
import cell as cl


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
    #check that sorting is correct
    assert solver.event_queue[0][0] <= solver.event_queue[1][0]
    assert solver.event_queue[1][0] <= solver.event_queue[2][0]

def test_get_next_event():
    solver = cbmos.CBMSolver(lambda r: 0., scpi.solve_ivp)

    cell_list = [cl.Cell(i, np.array([0, 0, i]), 0.0, True) for i in [0, 1, 2]]
    solver._build_event_queue(cell_list)
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
        N = 1000
        for i in range(N):
            mean_division_direction += cbm_solver._get_division_direction()

        print(mean_division_direction)

        assert np.all(abs(mean_division_direction/N) < 1)





