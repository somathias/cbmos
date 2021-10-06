import pytest
import logging
import numpy as np

import cbmos.cbmodel as cbmos
import cbmos.events as events
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.cell as cl

logging.basicConfig(level=logging.DEBUG)

def test_get_division_direction():
    for dim in [1, 2, 3]:
        cbmodel = cbmos.CBModel(lambda r: 0., ef.solve_ivp, dim)

        mean_division_direction = events.CellDivisionEvent._get_division_direction(None, cbmodel)
        assert mean_division_direction.shape == (dim,)
        assert np.isclose(np.linalg.norm(mean_division_direction), 1)

def test_apply_division():
    from cbmos.cbmodel._eventqueue import EventQueue
    from cbmos.events import CellDivisionEvent

    dim = 3
    cbmodel = cbmos.CBModel(ff.Linear(), ef.solve_ivp, dim)

    cell_list = [
            cl.Cell(i, [0, 0, i], proliferating=True, division_time=i + 1)
            for i in range(5)]
    for i, cell in enumerate(cell_list):
        cell.division_time = cell.ID

    cbmodel.cell_list = cell_list
    cbmodel.next_cell_index = 5
    cbmodel.queue = EventQueue([])
    event = CellDivisionEvent(cell_list[0])

    event.apply(cbmodel)

    assert len(cbmodel.cell_list) == 6
    assert cbmodel.cell_list[5].ID == 5
    assert cbmodel.next_cell_index == 6
    assert cell_list[0].division_time != 1

    assert np.isclose(
            np.linalg.norm(cell_list[0].position - cell_list[5].position),
            cbmodel.separation)
