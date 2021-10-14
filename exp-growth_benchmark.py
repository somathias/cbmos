"""
Growth benchmark: the simulation is run until a fixed simulation time. Simulation
Wall time is then recorded

Usage:
    python3 exp-growth_benchmark.py <output.json> <sep> <dim> <hpc_backend=cp>
"""
import numpy as np
import numpy.random as npr

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

separation = float(sys.argv[2])
dimension = int(sys.argv[3])

#dim = 2 # let's have a two-dimensional model
DT = 0.1
seed = 1

if bool(sys.argv[4]) == True:
    cbmmodel = cbmos.CBMModel(ff.Cubic(), ef.solve_ivp, dimension,
                              separation, hpc_backend=cp)
else:
    cbmmodel = cbmos.CBMModel(ff.Cubic(), ef.solve_ivp, dimension,
                              separation, hpc_backend=np)


s = 1.0
rA = 1.5
params_cubic = {"mu": 5.70, "s": s, "rA": rA}

n_run = 5

rate = 1.5
npr.seed(seed)
cell_list = [
    cl.Cell(
        0, np.zeros(dimension),
        proliferating=True, division_time_generator=lambda t: npr.exponential(rate*(t+1.0)) + t)
    ]

data = {}

for tf in [10, 20, 30]:

    t_data = [0.0, tf]

    # fixed time stepping
    time_fixed_dt = []
    n_fixed_dt = []
    if bool(sys.argv[4]) == True:
        #burn-in
        cbmmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed)
    for _ in range(n_run):
        start_fixed_dt = time.time()
        ts, history = cbmmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed)
        stop_fixed_dt = time.time()
        time_fixed_dt.append(stop_fixed_dt - start_fixed_dt)
        n_fixed_dt.append(len(history[-1]))

    # global adaptivity (accuracy only)
    # global adaptivity
    # local adaptivity

#    time_np = []
#    for _ in range(n_run):
#        start_np = time.time()
#        ts, history = cbmmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed)
#        stop_np = time.time()
#        time_np.append(stop_np - start_np)
#
#    time_cp = []
#    #Burnin
#    cbmmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed)
#    for _ in range(n_run):
#        start_cp = time.time()
#        ts, history = cbmmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed)
#        stop_cp = time.time()
#        time_cp.append(stop_cp - start_cp)

    data[tf] = {
        'fixed_dt': time_fixed_dt,
        'n_fixed_dt' : n_fixed_dt#,
#        'np': time_np,
#        'cp': time_cp,
        }

with open(sys.argv[1], 'w') as f:
    json.dump(data, f)

