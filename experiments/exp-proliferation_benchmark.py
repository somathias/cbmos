"""
Proliferation benchmark: the simulation is run for a given wall time and the
number of cells at the end of the simulation are counted.

Record data as { alg: [(time, cells, steps)] }

Usage:
    python3 exp-proliferation_benchmark.py <output.json>
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

dim = 2 # let's have a two-dimensional model
DT = float(sys.argv[2])
min_event_resolution = DT
seeds = list(range(5))
cbmmodel_numpy = cbmos.CBMModel(ff.Gls(), ef.solve_ivp, dim, hpc_backend=np)
cbmmodel_cupy = cbmos.CBMModel(ff.Gls(), ef.solve_ivp, dim, hpc_backend=cp)

params_gls = {'mu': 1.95, 'a':-2*np.log(0.002/1.95)}

data = {}

cell_list = [
        cl.Cell(
            0, [0., 0.],
            proliferating=True, division_time_generator=lambda t: npr.exponential(float(sys.argv[3])) + t)
        ]

t_data = np.linspace(0, 100, num=100)

for exec_time in [2**i for i in range(9)]:
    print(f"Box {exec_time}s")
    n_cell_box = []
    for seed in seeds:
        _, history = cbmmodel_numpy.simulate(
                cell_list, t_data,
                params_gls, {"dt": DT},
                seed=seed, box=True, max_execution_time=exec_time,
                min_event_resolution=min_event_resolution)

        # Make sure the simulation stopped before it reached the end of t_data
        assert len(history) != len(t_data)
        data.setdefault('box', []).append(
                (cbmmodel_numpy.last_exec_time, len(history[-1]), len(history))
                )

    print(f"NumPy {exec_time}s")
    n_cell_np = []
    for seed in seeds:
        _, history = cbmmodel_numpy.simulate(
                cell_list, t_data,
                params_gls, {"dt": DT},
                seed=seed, box=False, max_execution_time=exec_time,
                min_event_resolution=min_event_resolution)

        # Make sure the simulation stopped before it reached the end of t_data
        assert len(history) != len(t_data)
        n_cell_np.append(len(history[-1]))
        data.setdefault('np', []).append(
                (cbmmodel_numpy.last_exec_time, len(history[-1]), len(history))
                )

    print(f"CuPy {exec_time}s")
    n_cell_cp = []
    try:
        _, history = cbmmodel_cupy.simulate(
                cell_list, t_data,
                params_gls, {"dt": DT},
                seed=0, box=False, max_execution_time=exec_time,
                min_event_resolution=min_event_resolution)
    except cp.cuda.memory.OutOfMemoryError:
        pass

    for seed in seeds:
        try:
            _, history = cbmmodel_cupy.simulate(
                    cell_list, t_data,
                    params_gls, {"dt": DT},
                    seed=seed, box=False, max_execution_time=exec_time,
                    min_event_resolution=min_event_resolution)
        except cp.cuda.memory.OutOfMemoryError:
            n_cell_cp.append(None)
            continue

        # Make sure the simulation stopped before it reached the end of t_data
        assert len(history) != len(t_data)
        data.setdefault('cp', []).append(
                (cbmmodel_numpy.last_exec_time, len(history[-1]), len(history))
                )

with open(sys.argv[1], 'w') as f:
    json.dump(data, f)
