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

@pytest.fixture
def two_cells():
    return np.array([[0., 0., 0.], [0.5, 0., 0.]]).reshape(-1)


def test_ode_force(two_cells):
    # Check that the force is made used of
    solver = cbmos.CBMSolver(lambda r: 0., scpi.solve_ivp)
    f = solver.ode_force({})
    assert not f(0., two_cells).any()
    solver = cbmos.CBMSolver(lambda r: 1., scpi.solve_ivp)
    f = solver.ode_force({})
    assert f(0., two_cells).any()

    # Check parameters are made used of
    solver = cbmos.CBMSolver(lambda r, p=1.: p, scpi.solve_ivp)
    f = solver.ode_force({})
    assert f(0., two_cells).any()
    f = solver.ode_force({'p': 0.})
    assert not f(0., two_cells).any()

    # Check force computation is correct
    solver = cbmos.CBMSolver(lambda r: 1., scpi.solve_ivp)
    f = solver.ode_force({})
    total_force = f(0., two_cells).reshape(2, 3)
    assert np.array_equal(total_force[0], np.array([1., 0., 0.]))
    assert np.array_equal(total_force[0], -total_force[1])
