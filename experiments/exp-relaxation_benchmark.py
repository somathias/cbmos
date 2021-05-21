"""
Relaxation benchmark: the simulation is run until cell relax. Simulation
Wall time is then recorded

Usage:
    python3 exp-relaxation_benchmark.py <output.json>
"""
import numpy as np

import cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.cell as cl

import time
import json
import sys

import cupy as cp

if len(sys.argv) < 2:
    raise IOError("Must provide output file")

dim = 2 # let's have a two-dimensional model
DT = 0.1
seed = 1
cbmmodel_numpy = cbmos.CBMModel(ff.Gls(), ef.solve_ivp, dim, hpc_backend=np)
cbmmodel_cupy = cbmos.CBMModel(ff.Gls(), ef.solve_ivp, dim, hpc_backend=cp)

params_gls = {'mu': 1.95, 'a':-2*np.log(0.002/1.95)}

n_run = 5

data = {}

for n in [2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100]:
    n_x = n
    n_y = n
    scl = 0.8
    xcrds = [(2 * i + (j % 2)) * 0.5 * scl for j in range(n_y) for i in range(n_x)]
    ycrds = [np.sqrt(3) * j * 0.5 * scl for j in range(n_y) for i in range(n_x)]

    cell_list = []
    for i, (x, y) in enumerate(zip(xcrds, ycrds)):
        cell_list.append(cl.Cell(i, [x, y], proliferating=False))

    t_data = np.linspace(0, 1, 101)

    time_box = []
    for _ in range(n_run):
        start_box = time.time()
        history = cbmmodel_numpy.simulate(cell_list, t_data, params_gls, {"dt": DT}, seed=seed, box=True)
        stop_box = time.time()
        time_box.append(stop_box - start_box)

    time_np = []
    for _ in range(n_run):
        start_np = time.time()
        history = cbmmodel_numpy.simulate(cell_list, t_data, params_gls, {"dt": DT}, seed=seed, box=False)
        stop_np = time.time()
        time_np.append(stop_np - start_np)

    time_cp = []
    #Burnin
    cbmmodel_cupy.simulate(cell_list, t_data, params_gls, {"dt": DT}, seed=seed, box=False)
    for _ in range(n_run):
        start_cp = time.time()
        history = cbmmodel_cupy.simulate(cell_list, t_data, params_gls, {"dt": DT}, seed=seed, box=False)
        stop_cp = time.time()
        time_cp.append(stop_cp - start_cp)

    data[n**2] = {
        'box': time_box,
        'np': time_np,
        'cp': time_cp,
        }

with open(sys.argv[1], 'w') as f:
    json.dump(data, f)
