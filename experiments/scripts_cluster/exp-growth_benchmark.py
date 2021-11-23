"""
Growth benchmark: the simulation is run until a fixed simulation time. Simulation
Wall time is then recorded

Usage:
    python3 exp-growth_benchmark.py <output> <sep> <dim> <rate>
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

#import cupy as cp

if len(sys.argv) < 2:
    raise IOError("Must provide output file")

separation = float(sys.argv[2])
dimension = int(sys.argv[3])
if len(sys.argv) > 4:
    rate = float(sys.argv[4])
else:
    rate = 1.5

#dim = 2 # let's have a two-dimensional model
seed = 1

cbmodel = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dimension,
                              separation, hpc_backend=np)


s = 1.0
rA = 1.5
mu = 5.70
params_cubic = {"mu": mu, "s": s, "rA": rA}
DT = 0.01

tf = 1000.0
#tf = 100.0
t_data = [0.0, tf]

eps = 0.01
eta = 0.0001

n_run = 5
#n_run = 2

npr.seed(seed)
cell_list = [
    cl.Cell(
        0, np.zeros(dimension),
        proliferating=True, division_time_generator=lambda t: npr.exponential(rate*(t+1.0)) + t)
    ]

#n_target_cell_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_target_cell_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#n_target_cell_counts = [10, 20]

# global adaptivity (accuracy only)
data = {}

for i in range(n_run):
    ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta}, seed=seed, n_target_cells=n_target_cell_counts, throw_away_history=True)
    data[i] = cbmodel.target_cell_count_checkpoints

with open(sys.argv[1]+'_glob_adap_acc.json', 'w') as f:
    json.dump(data, f)

# global adaptivity
data = {}
for i in range(n_run):
    ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta, "jacobian": cbmodel.jacobian, "force_args": params_cubic}, seed=seed, n_target_cells=n_target_cell_counts,  throw_away_history=True)
    data[i] = cbmodel.target_cell_count_checkpoints

with open(sys.argv[1]+'_glob_adap_stab.json', 'w') as f:
    json.dump(data, f)

# local adaptivity
data = {}
for i in range(n_run):
    ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta, "jacobian": cbmodel.jacobian, "force_args": params_cubic, "local_adaptivity": True}, seed=seed, n_target_cells=n_target_cell_counts,  throw_away_history=True)
    data[i] = cbmodel.target_cell_count_checkpoints

with open(sys.argv[1]+'_local_adap.json', 'w') as f:
    json.dump(data, f)

# fixed time stepping
data = {}
for i in range(n_run):
    ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed, n_target_cells=n_target_cell_counts, throw_away_history=True)
    data[i] =  cbmodel.target_cell_count_checkpoints

with open(sys.argv[1]+'_fixed_dt.json', 'w') as f:
    json.dump(data, f)

