import numpy as np
import numpy.random as npr
import scipy.integrate as scpi
import heapq as hq
import warnings as wg

import force_functions as ff
import euler_forward as ef
import cell

NU = 1


class CBMSolver:
    def __init__(self, force, solver, dimension=3):
        self.force = force
        self.solver = solver
        self.dim = dimension

    def simulate(self, cell_list, t_data, force_args, solver_args):
        """

        Note
        ----
        Cell ordering in the output can vary between timepoints.

        """

        t = t_data[0]
        t_end = t_data[-1]

        # build event queue once, since independent of environment (for now)
        event_queue = self.build_event_queue(cell_list)

        while t < t_end:

            # generate next event
            tau, cell = self.get_next_event(event_queue)

            # calculate positions until time min(tau, t_end)
            t_eval = [t] + [time for time in t_data if t < time < min(tau, t_end)] + [min(tau, t_end)]
            y0 = np.array([cell.position for cell in cell_list]).reshape(-1)
            sol = self.calculate_positions(t_eval, y0, force_args, solver_args)
            cell_list = self.update_cell_positions(cell_list, sol)

            # apply event if tau <= t_end
            if tau <= t_end :
                cell_list = self.apply_division(cell_list, cell)

            # update current time t to min(tau, t_end)
            t = min(tau, t_end)
            print(t)

        # TODO: build history (right now only the state at t_end is returned)
        wg.warn('TODO: build history')
        return cell_list



    def get_next_event(self, event_queue):
        return hq.heappop(event_queue)


    def build_event_queue(self, cells):
        events = [(cell.division_time, cell) for cell in cells]
        hq.heapify(events)
        return events

    def apply_division(self, cell_list, cell):
        raise NotImplementedError('The apply_event function is not yet implemented.')
        return cell_list

    def update_cell_positions(self, cell_list, sol):
        raise NotImplementedError('The update_cell_position function is not yet implemented.')
        return cell_list


    def calculate_positions(self, t_eval, y0, force_args, solver_args):
        return self.solver(self.ode_force(force_args),
                           (t_eval[0], t_eval[-1]),
                           y0,
                           t_eval=t_eval,
                           **solver_args)

    def ode_force(self, force_args):
        """ Generate ODE force function from cell-cell force function

        Parameters
        ----------
        force: (r, **kwargs) -> float
            describes the force applying between two cells at distance r
        force_args:
            extra arguments for the force function

        Returns
        -------
        (t, y) -> dy/dt

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

    cbm_solver = CBMSolver(ff.cubic, ef.solve_ivp)

    # three cells at rest
    cell_list = [cell.Cell(i, np.array([0,0,i])) for i in [0,1,2]]
    t_data = np.linspace(0, 5, 10)

    updated_cell_list = cbm_solver.simulate(
            cell_list, t_data, {'s': 1.0, 'mu': 1.0, 'rA': 1.5}, {})


