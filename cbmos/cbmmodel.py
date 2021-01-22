import numpy as _np
import numpy.random as _npr
import heapq as _hq
import logging as _logging

from . import cell as _cl

_NU = 1


class CBMModel:
    def __init__(self, force, solver, dimension=3, separation=0.3, hpc=_np):
        self.force = force
        self.solver = solver
        self.dim = dimension
        self.separation = separation
        self.hpc = hpc

    def simulate(self, cell_list, t_data, force_args, solver_args, seed=None):
        """

        Note
        ----
        Cell ordering in the output can vary between timepoints.
        Cell indices need to be unique for the whole duration of the simulation.

        """

        _npr.seed(seed)

        for cell in cell_list:
                assert len(cell.position) == self.dim

        _logging.debug("Starting new simulation")

        t = t_data[0]
        t_end = t_data[-1]
        self.cell_list = [
                _cl.Cell(
                    cell.ID, cell.position, cell.birthtime,
                    cell.proliferating, cell.generate_division_time,
                    cell.division_time, cell.parent_ID)
                for cell in cell_list]

        self.next_cell_index = max(self.cell_list, key=lambda cell: cell.ID).ID + 1
        self.history = []
        self._save_data()

        # build event queue once, since independent of environment (for now)
        self._build_event_queue()

        while t < t_end:

            # generate next event
            tau, cell = self._get_next_event()

            if tau > t:
                # calculate positions until the last t_data smaller or equal to min(tau, t_end)
                t_eval = [t] \
                    + [time for time in t_data if t < time <= min(tau, t_end)]
                y0 = _np.array([cell.position for cell in self.cell_list]).reshape(-1)
                # only calculate positions if there is t_data before tau
                if len(t_eval) > 1:
                    sol = self._calculate_positions(t_eval, y0, force_args, solver_args)

                    # save data for all t_data points passed
                    for y_t in sol.y[:, 1:].T:
                        self._save_data(y_t.reshape(-1, self.dim))

                # continue the simulation until tau if necessary
                if tau > t_eval[-1] and tau <=t_end:
                    y0 = sol.y[:, -1] if len(t_eval) > 1 else y0
                    sol = self._calculate_positions([t_eval[-1], tau], y0, force_args, solver_args)

                    # save data for all t_data points passed
                    for y_t in sol.y[:, 1:].T:
                        self._save_data(y_t.reshape(-1, self.dim))

                # update the positions for the current time point
                self._update_positions(sol.y[:, -1].reshape(-1, self.dim).tolist())

            # apply event if tau <= t_end
            if tau <= t_end:
                self._apply_division(cell, tau)

            # update current time t to min(tau, t_end)
            t = min(tau, t_end)

        return self.history

    def _save_data(self, positions=None):
        """
        Note
        ----
        self.history has to be instantiated before the first call to _save_data
        as an empty list.
        """
        # copy correct positions to cell list that is stored
        if positions is not None:
            self.history.append([_cl.Cell(
                    cell.ID, pos, cell.birthtime, cell.proliferating,
                    cell.generate_division_time, cell.division_time,
                    cell.parent_ID)
                for cell, pos in zip(self.cell_list, positions)])
        else:
            self.history.append([_cl.Cell(
                    cell.ID, cell.position, cell.birthtime, cell.proliferating,
                    cell.generate_division_time, cell.division_time,
                    cell.parent_ID)
                for cell in self.cell_list])

    def _build_event_queue(self):
        events = [(cell.division_time, cell) for cell in self.cell_list]
        _hq.heapify(events)
        self.event_queue = events

    def _update_event_queue(self, cell):
        """
        Note
        ----
        The code assumes that all cell events are division events.
        """
        event = (cell.division_time, cell)
        _hq.heappush(self.event_queue, event)

    def _get_next_event(self):
        return _hq.heappop(self.event_queue)

    def _apply_division(self, cell, tau):
        """
        Note
        ----
        The code assumes that all cell events are division events,
        """

        #check that the parent cell has set its proliferating flag to True
        assert cell.proliferating

        division_direction = self._get_division_direction()
        updated_position_parent = cell.position -\
            0.5 * self.separation * division_direction
        position_daughter = cell.position +\
            0.5 * self.separation * division_direction

        daughter_cell = _cl.Cell(
                self.next_cell_index, position_daughter, birthtime=tau,
                proliferating=True,
                division_time_generator=cell.generate_division_time,
                parent_ID=cell.ID)
        self.next_cell_index = self.next_cell_index + 1
        self.cell_list.append(daughter_cell)
        self._update_event_queue(daughter_cell)

        cell.position = updated_position_parent
        cell.division_time = cell.generate_division_time(tau)
        self._update_event_queue(cell)

        _logging.debug("Division event: t={}, direction={}".format(
            tau, division_direction))

    def _get_division_direction(self):

        if self.dim == 1:
            division_direction = _np.array([-1.0 + 2.0 * _npr.randint(2)])

        elif self.dim == 2:
            random_angle = 2.0 * _np.pi * _npr.rand()
            division_direction = _np.array([
                _np.cos(random_angle),
                _np.sin(random_angle)])

        elif self.dim == 3:
            u = _npr.rand()
            v = _npr.rand()
            random_azimuth_angle = 2 * _np.pi * u
            random_zenith_angle = _np.arccos(2 * v - 1)
            division_direction = _np.array([
                _np.cos(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.sin(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.cos(random_zenith_angle)])
        return division_direction

    def _calculate_positions(self, t_eval, y0, force_args, solver_args):
        return self.solver(self._ode_force(force_args),
                           (t_eval[0], t_eval[-1]),
                           y0,
                           t_eval=t_eval,
                           **solver_args)

    def _update_positions(self, y):
        """
        Note
        ----
        The ordering in cell_list and sol.y has to match.
        """

        for cell, pos in zip(self.cell_list, y):
            cell.position = _np.array(pos)

    def _ode_force(self, force_args):
        """ Generate ODE force function from cell-cell force function

        Parameters
        ----------
        force: (r, **kwargs) -> float
            describes the force applying between two cells at distance r
        force_args:
            extra arguments for the force function

        Returns
        -------
        f: (t, y) -> dy/dt

        """
        def f(t, y):
            y_r = self.hpc.asarray(y).reshape((-1, self.dim))[:, :, self.hpc.newaxis] # shape (n, d, 1)
            cross_diff = y_r.transpose([2, 1, 0]) - y_r # shape (n, d, n)
            norm = self.hpc.sqrt((cross_diff**2).sum(axis=1))
            forces = self.force(norm, **force_args)\
                / (norm + self.hpc.diag(self.hpc.ones(y_r.shape[0])))
            total_force = (forces[:, self.hpc.newaxis, :] * cross_diff).sum(axis=2)

            fty = (_NU*total_force).reshape(-1)

            if self.hpc.__name__ == "cupy":
                return self.hpc.asnumpy(fty)
            else:
                return _np.asarray(fty)

        return f


if __name__ == "__main__":
    import warnings as wg

    from . import force_functions as ff
    from .solvers import euler_forward as ef

    dim = 1
    cbm_solver = CBMModel(ff.logarithmic, ef.solve_ivp, dim)

    cell_list = [_cl.Cell(0, [0], proliferating=True), _cl.Cell(1, [0.3], proliferating=True)]
    t_data = _np.linspace(0, 1, 101)


    wg.simplefilter("error", RuntimeWarning)

    try:
        history = cbm_solver.simulate(cell_list, t_data, {}, {})
    except RuntimeWarning:
        print('Caught RuntimeWarning.')
    print('Simulation done.')





