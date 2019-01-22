import numpy as np
import numpy.random as npr
import scipy.integrate as scpi

NU = 1

class CBMSolver:
    def __init__(self, force, solver):
        self.force = force
        self.solver = solver
    def simulate(self, t_eval, y0, force_args, solver_args):
        def f(t, y):
            y_r = y.reshape((-1, 3))
            tmp = np.repeat(y_r[:, :, np.newaxis], y_r.shape[0], axis=2)
            forces = self.force(np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1)), **force_args)
            total_force = (np.repeat(forces[:, np.newaxis, :], 3, axis=1)*(tmp - tmp.transpose())
                    ).sum(axis=2)
            return (NU*total_force).reshape(-1)

        return self.solver(f, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, **solver_args)

if __name__ == "__main__":
    @np.vectorize
    def force(r, S, M):
        if not r:
            return 0.

        return 4. * M * ((S/r)**12 - (S/r)**6)

    cbm_solver = CBMSolver(force, scpi.solve_ivp)

    T = np.linspace(0, 1, num=100)

    X, Y, Z = [4]*3
    y0 = np.array([[x, y, z] for x in range(X) for y in range(Y) for z in range(Z)],
            dtype=np.float64).reshape(-1)

    sol = cbm_solver.simulate(T, y0, {'S': 0.5, 'M': 3}, {})

    print(sol.y)
