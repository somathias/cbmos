#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unit tests of the CBM solver

"""
import io
import parse
import pytest
import logging
import heapq as hq
import numpy as np
import numpy.random as npr
import scipy.integrate as scpi

import cbmos.cbmodel as cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.cell as cl
import cbmos.events as ev

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def two_cells():
    return np.array([[0., 0., 0.], [0.5, 0., 0.]]).reshape(-1)


def test_ode_system(two_cells):
    # Check that the force is made use of
    solver = cbmos.CBModel(lambda r: 0., scpi.solve_ivp)
    f = solver._ode_system({})
    assert not f(0., two_cells).any()
    solver = cbmos.CBModel(lambda r: 1., scpi.solve_ivp)
    f = solver._ode_system({})
    assert f(0., two_cells).any()

    # Check parameters are made use of
    solver = cbmos.CBModel(lambda r, p=1.: p, scpi.solve_ivp)
    f = solver._ode_system({})
    assert f(0., two_cells).any()
    f = solver._ode_system({'p': 0.})
    assert not f(0., two_cells).any()

    # Check force computation is correct
    solver = cbmos.CBModel(lambda r: 1., scpi.solve_ivp)
    f = solver._ode_system({})
    total_force = f(0., two_cells).reshape(2, 3)
    assert np.array_equal(total_force[0], np.array([1., 0., 0.]))
    assert np.array_equal(total_force[0], -total_force[1])


def test_calculate_positions(two_cells):
    for dim in [1, 2, 3]:
        cbm_solver = cbmos.CBModel(ff.Linear(), ef.solve_ivp, dim)

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


def test_update_positions():
    dim = 3
    cbm_solver = cbmos.CBModel(ff.Linear(), ef.solve_ivp, dim)

    cell_list = [cl.Cell(i, [0, 0, i]) for i in range(5)]
    for i, cell in enumerate(cell_list):
        cell.division_time = cell.ID

    cbm_solver.cell_list = cell_list

    new_positions = [[0, i, i] for i in range(5)]
    cbm_solver._update_positions(new_positions)
    for i, cell in enumerate(cbm_solver.cell_list):
        print(cell.position)
        assert cell.position.tolist() == [0, i, i]


def test_simulate(caplog):
    dim = 1
    cbm_solver = cbmos.CBModel(ff.Cubic(), scpi.solve_ivp, dim)
    cell_list = [cl.ProliferatingCell(0, [0]), cl.ProliferatingCell(1, [1.0], 0.0, True)]
    cell_list[1].division_time = 1.05  # make sure not to divide at t_data

    event_list = [
            ev.CellDivisionEvent(cell)
            for cell in cell_list
            if cell.proliferating
            ]

    N = 100
    t_data = np.linspace(0, 10, N) # stay away from 24 hours
    t_data_sol, history = cbm_solver.simulate(
            cell_list, t_data, {}, {},
            raw_t=False, event_list=event_list)

    assert len(history) == N
    assert t_data_sol.tolist() == t_data.tolist()

    assert len(history[10]) == 2
    assert np.isclose(abs(history[10][0].position - history[10][1].position), 1)

    assert len(history[-1]) == 3
    scells = sorted(history[-1], key=lambda c:c.position)
    assert np.isclose(abs(scells[0].position - scells[1].position), 1, atol=1e-03)
    assert np.isclose(abs(scells[1].position - scells[2].position), 1, atol=1e-03)

def test_two_events_at_once():
    dim = 1
    cbm_solver = cbmos.CBModel(ff.Linear(), scpi.solve_ivp, dim)
    cell_list = [
            cl.ProliferatingCell(0, [0], proliferating=True),
            cl.ProliferatingCell(1, [1.0], 0.0, True)
            ]
    cell_list[0].division_time = 1.05
    cell_list[1].division_time = 1.05

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = np.linspace(0, 10, 100)
    _, history = cbm_solver.simulate(
            cell_list, t_data, {}, {},
            raw_t=False, event_list=[])

    assert len(history) == 100

def test_event_at_t_data():
    dim = 1
    cbm_solver = cbmos.CBModel(ff.Linear(), scpi.solve_ivp, dim)
    cell_list = [cl.ProliferatingCell(0, [0], proliferating=True), cl.ProliferatingCell(1, [1.0], 0.0, True)]
    cell_list[0].division_time = 1.0
    cell_list[1].division_time = 1.0

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = np.linspace(0, 10, 101)
    _, history = cbm_solver.simulate(
            cell_list, t_data, {}, {},
            raw_t=False, event_list=event_list)

    assert len(history) == len(t_data)

def test_no_division_skipped():
    dim = 1
    cbm_solver = cbmos.CBModel(ff.Linear(), scpi.solve_ivp, dim)
    cell_list = [cl.ProliferatingCell(0, [0], proliferating=True), cl.ProliferatingCell(1, [1.0], 0.0, True)]
    cell_list[0].division_time = 1.0
    cell_list[1].division_time = 1.0

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = np.linspace(0, 30, 101)
    _, history = cbm_solver.simulate(
            cell_list, t_data, {}, {},
            raw_t=False, event_list=event_list)

    eq = [hq.heappop(cbm_solver.queue._events) for i in range(len(cbm_solver.queue._events))]
    assert eq == sorted(eq)

    assert len(eq) == len(history[-1]) - 1

    for t, cells in zip(t_data, history):
        for c in cells:
            assert c.birthtime <= t
            assert c.division_time > t

def test_min_event_resolution():
    dim = 1
    cbm_solver = cbmos.CBModel(ff.Linear(), scpi.solve_ivp, dim)
    cell_list = [cl.ProliferatingCell(0, [0], proliferating=True), cl.ProliferatingCell(1, [1.0], 0.0, proliferating=True)]
    cell_list[0].division_time = 0.25
    cell_list[1].division_time = 0.25

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = [0, 0.4, 0.6, 1]
    _, history = cbm_solver.simulate(
            cell_list, t_data, {}, {},
            raw_t=False, event_list=event_list, min_event_resolution=0.5,
            )

    assert len(history[0]) == 2
    assert len(history[1]) == 2
    assert len(history[2]) == 4
    assert len(history[3]) == 4

def test_cell_list_copied():

    dim = 1
    cbm_solver_one = cbmos.CBModel(ff.Linear(), scpi.solve_ivp, dim)
    cbm_solver_two = cbmos.CBModel(ff.Linear(), scpi.solve_ivp, dim)

    cell_list = [cl.ProliferatingCell(0, [0], proliferating=True), cl.ProliferatingCell(1, [0.3], proliferating=True)]
    t_data = np.linspace(0, 1, 101)

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    _, history_one = cbm_solver_one.simulate(
            cell_list, t_data, {}, {},
            event_list=event_list)
    _, history_two = cbm_solver_two.simulate(
            cell_list, t_data, {}, {},
            event_list=event_list)

    assert history_two[0][0].position == np.array([0])
    assert history_two[0][1].position == np.array([0.3])

def test_tdata():
    n = 100

    s = 1.0    # rest length
    tf = 1.0  # final time
    rA = 1.5   # maximum interaction distance

    params_cubic = {"mu": 6.91, "s": s, "rA": rA}

    solver_ef = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, 1)
    t_data = np.linspace(0,1, n)
    cell_list = [cl.ProliferatingCell(0, [0], proliferating=False), cl.ProliferatingCell(1, [0.3], proliferating=False)]

    _, sols = solver_ef.simulate(
            cell_list, t_data, params_cubic, {'dt': 0.03},
            raw_t=False, event_list=[])
    y = np.array([np.squeeze([clt[0].position, clt[1].position]) for clt in sols])

    assert y.shape == (n, 2)

def test_tdata_raw():
    n = 100

    s = 1.0    # rest length
    tf = 1.0  # final time
    rA = 1.5   # maximum interaction distance

    params_cubic = {"mu": 6.91, "s": s, "rA": rA}

    solver_ef = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, 1)
    t_data = np.linspace(0,1, n)
    cell_list = [cl.ProliferatingCell(0, [0], proliferating=False), cl.ProliferatingCell(1, [0.3], proliferating=False)]

    t_data_sol, sols = solver_ef.simulate(
            cell_list, t_data, params_cubic, {'dt': 0.03},
            raw_t=True, event_list=[])

    assert len(t_data_sol) == len(sols)

def test_tdata_raw_division():
    n = 100

    s = 1.0    # rest length
    tf = 1.0  # final time
    rA = 1.5   # maximum interaction distance

    params_cubic = {"mu": 6.91, "s": s, "rA": rA}

    solver_ef = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, 1)
    t_data = np.linspace(0,50, n)
    cell_list = [
            cl.ProliferatingCell(0, [0], proliferating=True),
            cl.ProliferatingCell(1, [0.3], proliferating=True)
            ]

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data_sol, sols = solver_ef.simulate(
            cell_list, t_data, params_cubic, {'dt': 0.03},
            raw_t=True, event_list=event_list)

    assert len(sols[-1]) > len(cell_list) # Make sure some cells multiplied
    assert len(t_data_sol) == len(sols)
    assert all([t[0] < t[1] for t in zip(t_data_sol, t_data_sol[1:])])


def test_sparse_tdata():
    dim = 3
    solver_cubic = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim)
    ancestor = [cl.ProliferatingCell(0, np.zeros((dim,)), -5, True)]
    dt = 0.1
    t_f = 50
    t_data = np.linspace(0, t_f, 2)

    event_list = [ev.CellDivisionEvent(cell) for cell in ancestor]

    _, tumor_cubic = solver_cubic.simulate(
            ancestor, t_data, {"mu":6.91}, {"dt":dt},
            event_list=event_list)


def test_seed():
    dim = 3
    cbm_solver = cbmos.CBModel(ff.Logarithmic(), ef.solve_ivp, dim)

    cell_list = [cl.ProliferatingCell(0, [0, 0, 0], proliferating=True)]

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = np.linspace(0, 100, 10)
    histories = [
            cbm_solver.simulate(
                cell_list, t_data, {}, {},
                seed=seed, event_list=event_list)[1]
            for seed in [0, 0, 1, None, None]]

    for cells in zip(*[history[-1] for history in histories]):
        assert cells[0].position.tolist() == cells[1].position.tolist()
        assert cells[0].position.tolist() != cells[2].position.tolist()
        assert cells[3].position.tolist() != cells[4].position.tolist()


def test_seed_division_time(caplog):
    logger = logging.getLogger()
    logs = io.StringIO()
    logger.addHandler(logging.StreamHandler(logs))

    caplog.set_level(logging.DEBUG)

    dim = 3
    cbm_solver = cbmos.CBModel(ff.Logarithmic(), ef.solve_ivp, dim)

    cell_list = [cl.ProliferatingCell(0, [0, 0, 0], proliferating=True)]

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = np.linspace(0, 100, 10)

    history = [
            cbm_solver.simulate(
                cell_list, t_data, {}, {},
                seed=seed, event_list=event_list)[1]
            for seed in [0, 0, 1]]

    division_times = logs.getvalue().split("Starting new simulation\n")[1:]
    assert division_times[0] == division_times[1]
    assert division_times[0] != division_times[2]


def test_cell_dimension_exception():
    dim = 3
    cbm_solver = cbmos.CBModel(ff.Logarithmic(), ef.solve_ivp, dim)

    cell_list = [cl.ProliferatingCell(0, [0, 0], proliferating=True)]

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = np.linspace(0, 100, 10)

    with pytest.raises(AssertionError):
        cbm_solver.simulate(cell_list, t_data, {}, {}, event_list=event_list)


def test_cell_birth(caplog):
    logger = logging.getLogger()
    logs = io.StringIO()
    logger.addHandler(logging.StreamHandler(logs))

    caplog.set_level(logging.DEBUG)

    dim = 2
    cbm_solver = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim)

    cell_list = [
                cl.ProliferatingCell(0, [0, 0], -5.5, True,
                        division_time_generator=lambda t: 6 + t)]

    event_list = [ev.CellDivisionEvent(cell) for cell in cell_list]

    t_data = np.linspace(0, 1, 10)

    cbm_solver.simulate(cell_list, t_data, {}, {}, event_list=event_list)

    division_times = logs.getvalue()
    assert parse.search("Division event: t={:f}", division_times)[0] == 0.5


def test_cell_list_order():
    # 2D honeycomb mesh
    n_x = 5
    n_y = 5
    xcrds = [(2 * i + (j % 2)) * 0.5 for j in range(n_y) for i in range(n_x)]
    ycrds = [np.sqrt(3) * j * 0.5 for j in range(n_y) for i in range(n_x)]

    # make cell_list for the sheet
    sheet = [
            cl.ProliferatingCell(i, [x, y], -6.0, True, lambda t: 6 + t)
            for i, x, y in zip(range(n_x*n_y), xcrds, ycrds)]
    # delete cells to make it circular
    del sheet[24]
    del sheet[20]
    del sheet[19]
    del sheet[9]
    del sheet[4]
    del sheet[0]

    solver = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, 2)
    dt = 0.01
    t_data = np.arange(0, 3, dt)

    event_list = [ev.CellDivisionEvent(cell) for cell in sheet]

    _, history = solver.simulate(
            sheet, t_data, {"mu": 6.91}, {'dt': dt},
            seed=17, event_list=event_list)
    history = history[1:]  # delete initial data because that's less cells

    ids = [cell.ID for cell in history[0]]
    assert np.all([ids == [cell.ID for cell in clt] for clt in history[1:]])

def test_jacobian_1DN3():

    force = ff.Linear()
    force_prime = force.derive()

    y = np.array([1.0, 0.7, 2.5])[:, np.newaxis]

    model = cbmos.CBModel(force, ef.solve_ivp, 1)

    A = model.jacobian(y, {})

    # calculate manually
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    A12 = (y2-y1)**2/abs(y2-y1)**2\
            *(force_prime(abs(y2-y1)) - force(abs(y2-y1))/abs(y2-y1))\
            + force(abs(y2-y1))/abs(y2-y1)
    A13 = (y3-y1)**2/abs(y3-y1)**2\
            *(force_prime(abs(y3-y1)) - force(abs(y3-y1))/abs(y3-y1))\
            + force(abs(y3-y1))/abs(y3-y1)
    A23 = (y3-y2)**2/abs(y3-y2)**2\
            *(force_prime(abs(y3-y2)) - force(abs(y3-y2))/abs(y3-y2))\
            + force(abs(y3-y2))/abs(y3-y2)
    A2 = np.squeeze(np.array([[- (A12 +A13), A12, A13],[A12, -(A12 + A23), A23],[A13, A23, -(A13 +A23)]]))

    assert(np.all(A == A2))

def test_jacobian_2DN3():

    g = ff.Linear()
    g_prime = g.derive()

    y = np.array([[0., 0.], [0.7, 0.1], [0.3, -1.]])

    model = cbmos.CBModel(g, ef.solve_ivp, 2)


    A = model.jacobian(y, {})

    # calculate manually
    y1 = y[0, :][:, np.newaxis]
    y2 = y[1, :][:, np.newaxis]
    y3 = y[2, :][:, np.newaxis]

    r12 = y2 - y1
    norm_12 = np.linalg.norm(r12)
    r13 = y3 - y1
    norm_13 = np.linalg.norm(r13)
    r23 = y3 - y2
    norm_23 = np.linalg.norm(r23)

    A12 = r12@r12.transpose()/norm_12**2*(g_prime(norm_12) - g(norm_12)/norm_12) + g(norm_12)/norm_12 * np.eye(2)
    A13 = r13@r13.transpose()/norm_13**2*(g_prime(norm_13) - g(norm_13)/norm_13) + g(norm_13)/norm_13 * np.eye(2)
    A23 = r23@r23.transpose()/norm_23**2*(g_prime(norm_23) - g(norm_23)/norm_23) + g(norm_23)/norm_23 * np.eye(2)

    A2 = np.block([[- (A12 +A13), A12, A13],[A12, -(A12 + A23), A23],[A13, A23, -(A13 +A23)]])

    assert(np.all(A == A2))

def test_jacobian_3DN3():

    g = ff.Linear()
    g_prime = g.derive()

    y = np.array([[0., 0., 0.], [0.7, 0.1, -0.6], [0.3, -1., -2.0]])

    model = cbmos.CBModel(g, ef.solve_ivp, 3)
    A = model.jacobian(y, {})

    # calculate manually
    y1 = y[0, :][:, np.newaxis]
    y2 = y[1, :][:, np.newaxis]
    y3 = y[2, :][:, np.newaxis]

    r12 = y2 - y1
    norm_12 = np.linalg.norm(r12)
    r13 = y3 - y1
    norm_13 = np.linalg.norm(r13)
    r23 = y3 - y2
    norm_23 = np.linalg.norm(r23)

    A12 = r12@r12.transpose()/norm_12**2*(g_prime(norm_12) - g(norm_12)/norm_12) + g(norm_12)/norm_12 * np.eye(3)
    A13 = r13@r13.transpose()/norm_13**2*(g_prime(norm_13) - g(norm_13)/norm_13) + g(norm_13)/norm_13 * np.eye(3)
    A23 = r23@r23.transpose()/norm_23**2*(g_prime(norm_23) - g(norm_23)/norm_23) + g(norm_23)/norm_23 * np.eye(3)

    A2 = np.block([[- (A12 +A13), A12, A13],[A12, -(A12 + A23), A23],[A13, A23, -(A13 +A23)]])

    assert(np.all(A == A2))
