import numpy as np
import numpy.random as npr
import scipy.integrate as scpi

import force_functions as ff
import euler_forward as ef

NU = 1


class CBMSolver:
    def __init__(self, force, solver):
        self.force = force
        self.solver = solver

    def simulate(self, t_eval, y0, force_args, solver_args):
        return self.solver(self.ode_force(self.force, force_args),
                           (t_eval[0], t_eval[-1]),
                           y0,
                           t_eval=t_eval,
                           **solver_args)

    def ode_force(self, force, force_args):
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
            y_r = y.reshape((-1, 3))
            tmp = np.repeat(y_r[:, :, np.newaxis], y_r.shape[0], axis=2)
            norm = np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1))
            forces = force(norm, **force_args)\
                / (norm + np.diag(np.ones(y_r.shape[0])))
            total_force = (np.repeat(forces[:, np.newaxis, :], 3, axis=1)
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

    sol = cbm_solver.simulate(T, y0, {'s': 1.0, 'mu': 1.0, 'rA': 1.5}, {})

    print(sol.y)
