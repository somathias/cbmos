import numpy as _np
import numpy.random as _npr
import logging as _logging

from . import cell as _cl


class Event:
    """
    Abstract class for implementing events (e.g. cell division, cell death).
    """
    def __init__(self, *args, **kwargs):
        """
            Note
            ----
            `self.tau`, the time when the event takes place, must be defined
            here.
        """
        raise NotImplementedError

    def apply(self, cbmodel):
        """
        Apply event to cell population

        Parameters
        ----------
            cbmodel: CBModel
                model to which the event is applied
        """
        raise NotImplementedError

    def __lt__(self, other):
        return self.tau < other.tau


class CellDivisionEvent(Event):
    """
    Divide `target_cell` into parent cell and child cell, placed `separation`
    cell diameters apart.
    """
    def __init__(self, target_cell):
        assert target_cell.proliferating

        self.tau = target_cell.division_time
        self.target_cell = target_cell

    def apply(self, cbmodel):
        #check that the parent cell has set its proliferating flag to True
        assert self.target_cell.proliferating


        division_direction = self._get_division_direction(cbmodel)
        updated_position_parent = self.target_cell.position \
            - 0.5 * cbmodel.separation * division_direction
        position_daughter = self.target_cell.position \
            + 0.5 * cbmodel.separation * division_direction

        daughter_cell = _cl.Cell(
                cbmodel.next_cell_index, position_daughter, birthtime=self.tau,
                proliferating=True,
                division_time_generator=self.target_cell.generate_division_time,
                parent_ID=self.target_cell.ID)
        cbmodel.next_cell_index += 1
        cbmodel.cell_list.append(daughter_cell)
        cbmodel._queue.push(CellDivisionEvent(daughter_cell))

        self.target_cell.position = updated_position_parent
        self.target_cell.division_time = self.target_cell.generate_division_time(self.tau)
        cbmodel._queue.push(CellDivisionEvent(self.target_cell))

        _logging.debug("Division event: t={}, direction={}".format(
            self.tau, division_direction))


    def _get_division_direction(self, cbmodel):
        if cbmodel.dim == 1:
            division_direction = _np.array([-1.0 + 2.0 * _npr.randint(2)])

        elif cbmodel.dim == 2:
            random_angle = 2.0 * _np.pi * _npr.rand()
            division_direction = _np.array([
                _np.cos(random_angle),
                _np.sin(random_angle)])

        elif cbmodel.dim == 3:
            u = _npr.rand()
            v = _npr.rand()
            random_azimuth_angle = 2 * _np.pi * u
            random_zenith_angle = _np.arccos(2 * v - 1)
            division_direction = _np.array([
                _np.cos(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.sin(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.cos(random_zenith_angle)])

        return division_direction

