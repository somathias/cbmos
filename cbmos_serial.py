import numpy as np
import numpy.random as npr

DT = 0.01
NU = 1
X, Y, Z = [2]*3

@np.vectorize
def force(r):
    if not r:
        return 0.

    S = 0.5
    M = 0.01
    return 4. * M * ((S/r)**12 - (S/r)**6)

pop = np.array([[x, y, z] for x in range(X) for y in range(Y) for z in range(Z)],
        dtype=np.float64)

tmp = np.repeat(pop[:, :, np.newaxis], pop.shape[0], axis=2)

forces = force(np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1)))

total_force = (np.repeat(forces[:, np.newaxis, :], 3, axis=1)*(tmp - tmp.transpose())
        ).sum(axis=2)

print(total_force)
