import numpy as np
import numpy.random as npr
import scipy.integrate as scpi
import heapq as hq
import warnings as wg

import force_functions as ff
import euler_forward as ef
import cell as cl

NU = 1


class CBMSolver:
    def __init__(self, force, solver, dimension=3, separation=0.3):
        self.force = force
        self.solver = solver
        self.dim = dimension
        self.separation = separation

    def simulate(self, cell_list, t_data, force_args, solver_args):
        """

        Note
        ----
        Cell ordering in the output can vary between timepoints.
        Cell indices need to be unique for the whole duration of the simulation.

        """

        t = t_data[0]
        t_end = t_data[-1]
        self.cell_list = [cl.Cell(
                            cell.ID, cell.position, cell.birthtime,
                            cell.proliferating, cell.division_time,
                            cell.parent_ID) for cell in cell_list]

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
                y0 = np.array([cell.position for cell in self.cell_list]).reshape(-1)
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
            self.history.append([cl.Cell(
                    cell.ID, pos, cell.birthtime, cell.proliferating,
                    cell.division_time, cell.parent_ID)
                for cell, pos in zip(self.cell_list, positions)])
        else:
            self.history.append([cl.Cell(
                    cell.ID, cell.position, cell.birthtime, cell.proliferating,
                    cell.division_time, cell.parent_ID)
                for cell in self.cell_list])

    def _build_event_queue(self):
        events = [(cell.division_time, cell) for cell in self.cell_list]
        hq.heapify(events)
        self.event_queue = events

    def _update_event_queue(self, cell):
        """
        Note
        ----
        The code assumes that all cell events are division events.
        """
        event = (cell.division_time, cell)
        hq.heappush(self.event_queue, event)

    def _get_next_event(self):
        return hq.heappop(self.event_queue)

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

        daughter_cell = cl.Cell(
                self.next_cell_index, position_daughter, birthtime=tau, proliferating=True, parent_ID=cell.ID)
        self.next_cell_index = self.next_cell_index + 1
        self.cell_list.append(daughter_cell)
        self._update_event_queue(daughter_cell)

        cell.position = updated_position_parent
        cell.division_time = cell.generate_division_time(tau)
        self._update_event_queue(cell)

    def _get_division_direction(self):

        if self.dim == 1:
            division_direction = np.array([-1.0 + 2.0 * npr.randint(2)])

        elif self.dim == 2:
            random_angle = 2.0 * np.pi * npr.rand()
            division_direction = np.array([
                np.cos(random_angle),
                np.sin(random_angle)])

        elif self.dim == 3:
            u = npr.rand()
            v = npr.rand()
            random_azimuth_angle = 2 * np.pi * u
            random_zenith_angle = np.arccos(2 * v - 1)
            division_direction = np.array([
                np.cos(random_azimuth_angle) * np.sin(random_zenith_angle),
                np.sin(random_azimuth_angle) * np.sin(random_zenith_angle),
                np.cos(random_zenith_angle)])
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
            cell.position = np.array(pos)

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
            y_r = y.reshape((-1, self.dim))
            tmp = np.repeat(y_r[:, :, np.newaxis], y_r.shape[0], axis=2)
            norm = np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1))
            forces = self.force(norm, **force_args)\
                / (norm + np.diag(np.ones(y_r.shape[0])))
            total_force = (np.repeat(forces[:, np.newaxis, :],
                                     self.dim, axis=1)
                           * (tmp.transpose()-tmp)).sum(axis=2)
            return (NU*total_force).reshape(-1)

        return f


if __name__ == "__main__":

    dim = 1
    cbm_solver = CBMSolver(ff.logarithmic, ef.solve_ivp, dim)

    cell_list = [cl.Cell(0, [0], proliferating=True), cl.Cell(1, [0.3], proliferating=True)]
    t_data = np.linspace(0, 1, 101)


    wg.simplefilter("error", RuntimeWarning)

    try:
        history = cbm_solver.simulate(cell_list, t_data, {}, {})
    except RuntimeWarning:
        print('Caught RuntimeWarning.')
    print('Simulation done.')





