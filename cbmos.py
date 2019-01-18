import numpy as np
import dask.bag as db
import numpy.random as npr

DT = 0.01
NU = 1
X, Y, Z = 2, 2, 2

class Cell:
    def __init__(self, coords):
        self.id = hash(tuple(coords))
        self.coords = coords
    def __eq__(self, other):
        return self.id == other.id
    def __hash__(self):
        return self.id
    def __repr__(self):
        return "Cell({})".format(self.coords)
    def move(self, dv):
        self.coords += dv
        return self

def distance(ci, cj):
    return np.sqrt(sum((ci.coords - cj.coords)**2))

def force(cij):
    ci, cj = cij
    r = distance(ci, cj)
    S = 0.5
    M = 0.01
    return -4 * M * ((S/r)**12 - (S/r)**6) * (cj.coords-ci.coords) / r

pop = db.from_sequence([Cell(np.array([x, y, z], dtype=np.float64) + npr.uniform(0, 0.1, 3)) for x in range(X) for y in range(Y) for z in range(Z)])

for _ in range(200):
    forces = pop.product(pop).filter(lambda cij: cij[0].id != cij[1].id)\
            .map(lambda cij: (cij[0], force(cij))).foldby(lambda cif: cif[0], lambda cif, cjf: (cif[0], cif[1] + cjf[1]))
    pop = forces.map(lambda cif: cif[1][0].move(DT*cif[1][1]/NU))

cells = pop.compute()

for cell in cells:
    print(cell)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y, z = [[c.coords[i] for c in cells] for i in range(3)]
ax.scatter(x, y, z)
plt.show()
