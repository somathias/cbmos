import numpy as np
import numpy.random as npr
import scipy.integrate as scpi

NU = 1
X, Y, Z = [4]*3

@np.vectorize
def force(r):
    if not r:
        return 0.

    S = 0.5
    M = 3
    return 4. * M * ((S/r)**12 - (S/r)**6)

def f(t, y):
    y_r = y.reshape((-1, 3))
    tmp = np.repeat(y_r[:, :, np.newaxis], y_r.shape[0], axis=2)
    forces = force(np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1)))
    total_force = (np.repeat(forces[:, np.newaxis, :], 3, axis=1)*(tmp - tmp.transpose())
            ).sum(axis=2)
    return (NU*total_force).reshape(-1)


pop = np.array([[x, y, z] for x in range(X) for y in range(Y) for z in range(Z)],
        dtype=np.float64).reshape(-1)


T = np.linspace(0, 1, num=100)
sol = scpi.solve_ivp(f, (T[0], T[-1]), pop, t_eval=T)
print(sol.y)
