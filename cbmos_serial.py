import numpy as np
import numpy.random as npr
import scipy.integrate as scpi
import heapq as hq

import force_functions as ff
import euler_forward as ef

NU = 1


class CBMSolver:
    def __init__(self, force, solver, dimension=3):
        self.force = force
        self.solver = solver
        self.dim = dimension

    def simulate(self, cells, t_data, force_args, solver_args):
        """

        Note
        ----
        Cell ordering in the output can vary between timepoints.

        """

        t = t_data[0]
        t_end = t_data[-1]

        # build event queue once, since independent of environment (for now)
        event_queue = self.build_event_queue(cells)

        while t < t_end :

            # generate next event
            tau, cell = self.get_next_event(event_queue)

            # calculate positions until time min(tau, t_end)
            t_eval = [t] + [time for time in t_data if t < time < tau] + [tau]
            y0 = np.array([cell.position for cell in cells]).reshape(-1)
            self.calculate_positions(t_eval, y0, force_args, solver_args)

        # apply event if tau <= t_end

        # update current time t to min(tau, t_end)

        # if t<t_end go to first step

        return None



    def get_next_event(self, event_queue):
        return hq.heappop(event_queue)


    def build_event_queue(self, cells):
        events = [(cell.division_time, cell) for cell in cells]
        hq.heapify(events)
        return events




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

    T = np.linspace(0, 1, num=100)

    X, Y, Z = [4]*3
    y0 = np.array([[x,
                    y,
                    z] for x in range(X) for y in range(Y) for z in range(Z)],
                  dtype=np.float64).reshape(-1)

    sol = cbm_solver.calculate_positions(T, y0, {'s': 1.0, 'mu': 1.0, 'rA': 1.5}, {})

    print(sol.y)
