#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:31:43 2021

@author: Sonja Mathias
"""

import numpy as np
import scipy as scp
import scipy.interpolate as sci_interp
import scipy.integrate as sci_integr

import cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.solvers.euler_backward as eb
import cbmos.cell as cl

import time
import json
import sys

import cupy as cp

data = {}

# Simulation parameters
s = 1.0    # rest length
rA = 1.5   # maximum interaction distance
n = 10

dt = 1e-4
eps_max = 1e-2

#dt_values = [0.001*1.25**n for n in range(0, 25)]
dt_values = [0.001*1.25**n for n in range(25, 31)]
data['dt_values'] = dt_values


seed = 17

params_cubic = {"mu": 5.70, "s": s, "rA": rA}
muR = 9.1
ratio = 0.21
params_poly = {'muA': muR*ratio,
               'muR': muR,
               'rA': rA,
               'rR': 1.0/(1.0-np.sqrt(ratio)/3.0),
               'n': 1.0,
               'p': 1.0}
params_gls = {'mu': 1.95, 'a': 7.51}
params = {'cubic': params_cubic, 'pw. quad.': params_poly, 'GLS': params_gls}

force_names = ['cubic', 'pw. quad.', 'GLS']
solver_names = ['EF', 'EB']

def calculate_reference_solutions(cell_list,  # initial cell populations
                                  model_dicts,  # models
                                  params,  # force function parameters
                                  dt_ref,  # time step size
                                  tf):  # final time
    t_data_ref = np.arange(0, tf, dt_ref)
    ref_sol_dicts = {}
    for solver in solver_names:
        models = model_dicts[solver]
        ref_sols = {}
        for force in force_names:
            (t_data, history) = models[force].simulate(cell_list,
                                                       t_data_ref,
                                                       params[force],
                                                       {'dt': dt_ref},
                                                       seed=seed)
            ref_traj = np.array([[cell.position for cell in cell_list]
                                 for cell_list in history])  # (N,n_cells,dim)
            ref_sols[force] = (t_data, ref_traj)
        ref_sol_dicts[solver] = ref_sols
    return ref_sol_dicts


def do_convergence_study(cell_list,  # initial cell configuration
                               model_dicts,  # models
                               params,  # force function parameters
                               ref_sol_dicts,  # reference solutions
                               dt_values,  # dt values
                               tf,  # final simulation time
                               seed):  # seed
    sol_dicts = {}
    for solver in solver_names:
        sol = {'cubic': [], 'pw. quad.': [], 'GLS': []}
        for dt in dt_values:
            t_data = np.arange(0,tf,dt)
            for force in force_names:
                (t, history) = model_dicts[solver][force].simulate(cell_list,
                                                                       t_data,
                                                                       params[force],
                                                                       {'dt': dt},
                                                                       seed=seed)
                traj = np.array([[cell.position for cell in cell_list] for cell_list in history]) # (N, n_cells, dim)
                t_ref = ref_sol_dicts[solver][force][0]
                traj_ref = ref_sol_dicts[solver][force][1]
                interp = sci_interp.interp1d(t, traj, axis=0,
                                             bounds_error=False,
                                             fill_value=tuple(traj[[0, -1], :, :]))(t_ref[:])
                error = (np.linalg.norm(interp - traj_ref, axis=0)
                         / np.linalg.norm(traj_ref, axis=0)).mean(axis=0)
                sol[force].append(error.tolist())
        sol_dicts[solver] = sol
    return sol_dicts


def do_timed_convergence_study(cell_list,  # initial cell configuration
                               model_dicts,  # models
                               params,  # force function parameters
                               dt_values,  # dt values
                               dt_ref,  # reference time step size
                               tf,  # final simulation time
                               eps_max, # threshold value EB
                               n):  # number of repetitions for calculating the average wall time
    params_solver_ref = {'EF': {'dt': dt_ref}, 'EB': {'dt': dt_ref, 'eps_max': eps_max}}
    t_data_ref = np.arange(0, tf, dt_ref)
    sol_dicts = {}
    for solver in solver_names:
        sol = {'cubic': [], 'pw. quad.': [], 'GLS': []}
        for force in force_names:
            for j in np.arange(n):
                (t_ref, history_ref) = model_dicts[solver][force].simulate(cell_list,
                                                   t_data_ref,
                                                   params[force],
                                                   params_solver_ref[solver],
                                                   seed=j)
                traj_ref = np.array([[cell.position for cell in cell_list]
                             for cell_list in history_ref])  # (N,n_cells,dim)

                exec_times = []
                errors = []
                for dt in dt_values:
                    params_solver = {'EF': {'dt': dt}, 'EB': {'dt': dt, 'eps_max': eps_max}}
                    t_data = np.arange(0, tf, dt)

                    # time the calculation of the trajectories
                    start = time.time()
                    (t, history) = model_dicts[solver][force].simulate(cell_list,
                                                                       t_data,
                                                                       params[force],
                                                                       params_solver[solver],
                                                                       seed=j)
                    exec_time = time.time() - start
                    exec_times.append(exec_time)

                    traj = np.array([[cell.position for cell in cell_list] for cell_list in history]) # (N, n_cells, dim)
                    interp = sci_interp.interp1d(t, traj, axis=0,
                                             bounds_error=False,
                                             fill_value=tuple(traj[[0, -1], :, :]))(t_ref[:])
                    error = (np.linalg.norm(interp - traj_ref, axis=0)
                         / np.linalg.norm(traj_ref, axis=0)).mean(axis=0)
                    errors.append(error.tolist())
                sol[force].append((errors, exec_times)) # errors/exec_times collect the errors/exec times for all dt values in a single repetition
        sol_dicts[solver] = sol
    return sol_dicts



# 1D models for relaxation between daughter cells and two adhering cells
models_ef_1d = {'pw. quad.': cbmos.CBMModel(ff.PiecewisePolynomial(),
                                            ef.solve_ivp, 1),
                'cubic': cbmos.CBMModel(ff.Cubic(), ef.solve_ivp, 1),
                'GLS': cbmos.CBMModel(ff.Gls(), ef.solve_ivp, 1)}
models_eb_1d = {'pw. quad.': cbmos.CBMModel(ff.PiecewisePolynomial(),
                                            eb.solve_ivp, 1),
                'cubic': cbmos.CBMModel(ff.Cubic(), eb.solve_ivp, 1),
                'GLS': cbmos.CBMModel(ff.Gls(), eb.solve_ivp, 1)}
model_dicts_1d = {'EF': models_ef_1d, 'EB': models_eb_1d}

# 2d models for small monolayer using numpy
dim = 2
models_ef_2d = {'pw. quad.': cbmos.CBMModel(ff.PiecewisePolynomial(),
                                            ef.solve_ivp, dim),
                'cubic': cbmos.CBMModel(ff.Cubic(), ef.solve_ivp, dim),
                'GLS': cbmos.CBMModel(ff.Gls(), ef.solve_ivp, dim)}
models_eb_2d = {'pw. quad.': cbmos.CBMModel(ff.PiecewisePolynomial(),
                                            eb.solve_ivp, dim),
                'cubic': cbmos.CBMModel(ff.Cubic(), eb.solve_ivp, dim),
                'GLS': cbmos.CBMModel(ff.Gls(), eb.solve_ivp, dim)}
model_dicts_2d = {'EF': models_ef_2d, 'EB': models_eb_2d}

# 2d models for large monolayer using cupy
dim = 2
models_ef_2d_cp = {'pw. quad.': cbmos.CBMModel(ff.PiecewisePolynomial(), ef.solve_ivp, dim, hpc_backend=cp),
             'cubic': cbmos.CBMModel(ff.Cubic(), ef.solve_ivp, dim, hpc_backend=cp),
             'GLS': cbmos.CBMModel(ff.Gls(), ef.solve_ivp, dim, hpc_backend=cp)}
models_eb_2d_cp = {'pw. quad.': cbmos.CBMModel(ff.PiecewisePolynomial(), eb.solve_ivp, dim, hpc_backend=cp),
             'cubic': cbmos.CBMModel(ff.Cubic(), eb.solve_ivp, dim, hpc_backend=cp),
             'GLS': cbmos.CBMModel(ff.Gls(), eb.solve_ivp, dim, hpc_backend=cp)}
model_dicts_2d_cp = {'EF': models_ef_2d_cp, 'EB': models_eb_2d_cp}

#------------------
#  relaxation between daughter cells
# ------------------
daughter_cells = [cl.Cell(0, [0], proliferating=False),
                  cl.Cell(1, [0.3], proliferating=False)]

tf = 1.0

sol_dicts_relax = do_timed_convergence_study(daughter_cells,
                                             model_dicts_1d,
                                             params,
                                             dt_values,
                                             dt,
                                             tf,
                                             eps_max,
                                             n)

data['relax_1d'] = sol_dicts_relax

## ---------------
## Adhering cells
## ----------------
#
adhering_cells = [cl.Cell(0, [0], proliferating=False),
                  cl.Cell(1, [1.15], proliferating=False)]

tf = 3.0
sol_dicts_adhesion = do_timed_convergence_study(adhering_cells,
                                                model_dicts_1d,
                                                params,
                                                dt_values,
                                                dt,
                                                tf,
                                                eps_max,
                                                n)

data['adhesion_1d'] = sol_dicts_adhesion


# ---------------
# Small monolayer
# ----------------

# Initial condition

# 2D honeycomb mesh
n_x = 5
n_y = 5
xcrds = [(2 * i + (j % 2)) * 0.5 for j in range(n_y) for i in range(n_x)]
ycrds = [np.sqrt(3) * j * 0.5 for j in range(n_y) for i in range(n_x)]

# make cell_list for the sheet
sheet = [cl.Cell(i, [x, y], -6.0, True, lambda t: 6 + t)
         for i, x, y in zip(range(n_x*n_y), xcrds, ycrds)]
# delete cells to make it circular
del sheet[24]
del sheet[20]
del sheet[19]
del sheet[9]
del sheet[4]
del sheet[0]

# prepare consistent initial data
solver_scipy = cbmos.CBMModel(ff.PiecewisePolynomial(),
                              sci_integr.solve_ivp, dim)
t_data_init = [0, 0.0001]
(t_data, initial_sheet) = solver_scipy.simulate(sheet, t_data_init,
                                                {'muA': 0.21*9.1,
                                                 'muR': 9.1, 'rA': rA,
                                                 'rR': 1.0/(1.0-np.sqrt(0.21)/3.0),
                                                 'n': 1.0, 'p': 1.0
                                                },
                                                {}, seed=seed)
initial_sheet = initial_sheet[-1]

tf = 3.0
sol_dicts_small_monolayer = do_timed_convergence_study(initial_sheet,
                                                       model_dicts_2d,
                                                       params,
                                                       dt_values,
                                                       dt,
                                                       tf, eps_max,
                                                       n)

data['small_monolayer'] = sol_dicts_small_monolayer

# large monolayer
n_x = 20
n_y = 20
scl = 0.8
xcrds = [(2 * i + (j % 2)) * 0.5 * scl for j in range(n_y) for i in range(n_x)]
ycrds = [np.sqrt(3) * j * 0.5 * scl for j in range(n_y) for i in range(n_x)]

cell_list = []
for i, (x, y) in enumerate(zip(xcrds, ycrds)):
    cell_list.append(cl.Cell(i, [x, y], proliferating=False))

tf = 3.0
sol_dicts_large_monolayer = do_timed_convergence_study(cell_list,
                                                       model_dicts_2d_cp,
                                                       params,
                                                       dt_values,
                                                       dt,
                                                       tf,
                                                       eps_max,
                                                       n)

data['large_monolayer'] = sol_dicts_large_monolayer

with open(sys.argv[1], 'w') as f:
    json.dump(data, f)
