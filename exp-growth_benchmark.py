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

#import cupy as cp

if len(sys.argv) < 2:
    raise IOError("Must provide output file")

separation = float(sys.argv[2])
dimension = int(sys.argv[3])

#dim = 2 # let's have a two-dimensional model
seed = 1

#if bool(sys.argv[4]) == True:
#    cbmodel = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dimension,
#                              separation, hpc_backend=cp)
#else:
cbmodel = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dimension,
                              separation, hpc_backend=np)


s = 1.0
rA = 1.5
mu = 5.70
params_cubic = {"mu": mu, "s": s, "rA": rA}
DT = 0.01

eps = 0.01
eta = 0.0001

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
        cbmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed)
    for _ in range(n_run):
        start_fixed_dt = time.time()
        ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"dt": DT}, seed=seed)
        stop_fixed_dt = time.time()
        time_fixed_dt.append(stop_fixed_dt - start_fixed_dt)
        n_fixed_dt.append(len(history[-1]))

    # global adaptivity (accuracy only)
    time_global_adap_acc = []
    n_global_adap_acc = []
    if bool(sys.argv[4]) == True:
        #burn-in
        cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta}, seed=seed)
    for _ in range(n_run):
        start_global_adap_acc = time.time()
        ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta}, seed=seed)
        stop_global_adap_acc = time.time()
        time_global_adap_acc.append(stop_global_adap_acc - start_global_adap_acc)
        n_global_adap_acc.append(len(history[-1]))

    # global adaptivity
    time_global_adap_stab = []
    n_global_adap_stab = []
    if bool(sys.argv[4]) == True:
        #burn-in
        cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta, "jacobian": cbmodel.jacobian, "force_args": params_cubic}, seed=seed)
    for _ in range(n_run):
        start_global_adap_stab = time.time()
        ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta, "jacobian": cbmodel.jacobian, "force_args": params_cubic}, seed=seed)
        stop_global_adap_stab = time.time()
        time_global_adap_stab.append(stop_global_adap_stab - start_global_adap_stab)
        n_global_adap_stab.append(len(history[-1]))

    # local adaptivity
    time_local_adap = []
    n_local_adap = []
    if bool(sys.argv[4]) == True:
        #burn-in
        cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta, "jacobian": cbmodel.jacobian, "force_args": params_cubic, "local_adaptivity": True}, seed=seed)
    for _ in range(n_run):
        start_local_adap = time.time()
        ts, history = cbmodel.simulate(cell_list, t_data, params_cubic, {"eps": eps, "eta": eta, "jacobian": cbmodel.jacobian, "force_args": params_cubic, "local_adaptivity": True}, seed=seed)
        stop_local_adap = time.time()
        time_local_adap.append(stop_local_adap - start_local_adap)
        n_local_adap.append(len(history[-1]))


    data[tf] = {
        'fixed_dt': time_fixed_dt,
        'n_fixed_dt' : n_fixed_dt,
        'global_adap_acc': time_global_adap_acc,
        'n_global_adap_acc' : n_global_adap_acc,
        'global_adap_stab': time_global_adap_stab,
        'n_global_adap_stab' : n_global_adap_stab,
        'local_adap': time_local_adap,
        'n_local_adap' : n_local_adap
#        'np': time_np,
#        'cp': time_cp,
        }

with open(sys.argv[1], 'w') as f:
    json.dump(data, f)

