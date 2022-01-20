#!/usr/bin/env python
# coding: utf-8

# # Local adaptive time-stepping for a locally compressed monolayer due to a single proliferation event

# In[1]:


import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as scpi

import os
import sys
import json

import cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.cell as cl
import cbmos.utils as ut

if len(sys.argv) < 2:
    raise IOError("Must provide output file")

if len(sys.argv) < 3:
    seed=67
else:
    seed = int(sys.argv[2])
npr.seed(seed)

# Simulation parameters
s = 1.0    # rest length
tf = 4.0 # final time
rA = 1.5   # maximum interaction distance
separation = 0.3 # initial separation between daughter cells
t_data = [0, tf]

dim = 3

force = 'cubic'
# parameters fitted to relaxation time t=1.0h
params_cubic = {"mu": 5.70, "s": s, "rA": rA}


#sheet = ut.setup_locally_compressed_spheroid(6,6,6)

def make_hcp_mesh(n_x, n_y, n_z, scaling=1.0):
# 3D honeycomb mesh

    coords = [((2 * i_x + ((i_y + i_z) % 2)) * 0.5 * scaling,
             np.sqrt(3) * (i_y + (i_z % 2) / 3.0) * 0.5 * scaling,
             np.sqrt(6) * i_z / 3.0 * scaling)
            for i_x in range(n_x) for i_y in range(n_y) for i_z in range(n_z)
            ]


    # make cell_list for the sheet
    sheet = [cl.Cell(i, [x,y,z]) for i, (x, y, z) in enumerate(coords)]


    # find middle index, move cell there and add second daughter cells
    m = (n_x*n_y)*(n_z//2)+n_x*(n_y//2)+n_x//2
    coords = list(sheet[m].position)

    # get division direction
    u = npr.rand()
    v = npr.rand()
    random_azimuth_angle = 2 * np.pi * u
    random_zenith_angle = np.arccos(2 * v - 1)
    division_direction = np.array([
                np.cos(random_azimuth_angle) * np.sin(random_zenith_angle),
                np.sin(random_azimuth_angle) * np.sin(random_zenith_angle),
                np.cos(random_zenith_angle)])

    # update positions
    updated_position_parent = coords - 0.5 * separation * division_direction
    sheet[m].position = updated_position_parent

    position_daughter = coords + 0.5 * separation * division_direction

    # add daughter cell
    next_cell_index = len(sheet)
    daughter_cell = cl.Cell(next_cell_index, position_daughter)
    sheet.append(daughter_cell)

    return sheet

sheet = make_hcp_mesh(6,6,6)

data = {}

eps = 0.01
cbmodel = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim, separation)
algorithms = ['glob_adap_acc', 'glob_adap_stab' ,  'local_adap', 'fixed_dt' ]
exec_times = {}
F_evaluations = {}
A_evaluations = {}
ts_s = {}

n = 10


out = 'glob_adap_acc'
for i in range(n):
    try:
        os.remove('exec_times'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, "out": out, "measure_wall_time": True}, seed=seed)

    with open('exec_times'+out+'.txt', 'r') as f:
        if i==0:
            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
        else:
            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
    if i==0:
        with open('F_evaluations'+out+'.txt', 'r') as f:
            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
exec_times[out] = list(exec_times[out]/n)
F_evaluations[out] = list(F_evaluations[out])
ts_s[out] = ts
A_evaluations[out] = list(np.zeros(len(ts)))

out = 'glob_adap_stab'
for i in range(n):
    try:
        os.remove('exec_times'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('A_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, "out": out,"jacobian": cbmodel.jacobian, "force_args": params_cubic, "always_calculate_Jacobian": True, "measure_wall_time": True}, seed=seed)

    with open('exec_times'+out+'.txt', 'r') as f:
        if i==0:
            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
        else:
            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
    if i==0:
        with open('F_evaluations'+out+'.txt', 'r') as f:
            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
exec_times[out] = list(exec_times[out]/n)
F_evaluations[out] = list(F_evaluations[out])
A_evaluations[out] = list(A_evaluations[out])
ts_s[out] = ts

out = 'glob_adap_stab_stop_Jac_update'
for i in range(n):
    try:
        os.remove('exec_times'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('A_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, "out": out,"jacobian": cbmodel.jacobian, "force_args": params_cubic, "measure_wall_time": True}, seed=seed)

    with open('exec_times'+out+'.txt', 'r') as f:
        if i==0:
            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
        else:
            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
    if i==0:
        with open('F_evaluations'+out+'.txt', 'r') as f:
            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
exec_times[out] = list(exec_times[out]/n)
F_evaluations[out] = list(F_evaluations[out])
A_evaluations[out] = list(A_evaluations[out])
ts_s[out] = ts

# local adaptivity
out = 'local_adap'
for i in range(n):
    try:
        os.remove('exec_times'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('A_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, "out": out, "jacobian": cbmodel.jacobian, "force_args": params_cubic, "always_calculate_Jacobian": True, "local_adaptivity": True, "measure_wall_time": True}, seed=seed)

    with open('exec_times'+out+'.txt', 'r') as f:
        if i==0:
            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
        else:
            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
    if i==0:
        with open('F_evaluations'+out+'.txt', 'r') as f:
            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
exec_times[out] = list(exec_times[out]/n)
F_evaluations[out] = list(F_evaluations[out])
A_evaluations[out] = list(A_evaluations[out])
ts_s[out] = ts

out = 'local_adap_stop_Jac_update'
for i in range(n):
    try:
        os.remove('exec_times'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('A_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, "out": out, "jacobian": cbmodel.jacobian, "force_args": params_cubic, "local_adaptivity": True, "measure_wall_time": True}, seed=seed)

    with open('exec_times'+out+'.txt', 'r') as f:
        if i==0:
            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
        else:
            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
    if i==0:
        with open('F_evaluations'+out+'.txt', 'r') as f:
            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
exec_times[out] = list(exec_times[out]/n)
F_evaluations[out] = list(F_evaluations[out])
A_evaluations[out] = list(A_evaluations[out])
ts_s[out] = ts

# fixed time stepping
out = 'fixed_dt'
#DT = 0.01 #I can make this more specific to this set up
DT = ts_s['glob_adap_acc'][1] - ts_s['glob_adap_acc'][0]
print('DT='+str(DT))
for i in range(n):

    try:
        os.remove('exec_times'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"dt": DT, "out": out, "measure_wall_time": True}, seed=seed)

    with open('exec_times'+out+'.txt', 'r') as f:
        if i==0:
            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
        else:
            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
    if i==0:
        with open('F_evaluations'+out+'.txt', 'r') as f:
            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
exec_times[out] = list(exec_times[out]/n)
ts_s[out] = ts
F_evaluations[out] = list(F_evaluations[out])
A_evaluations[out] = list(np.zeros(len(ts)))



data['exec_times'] = exec_times
data['ts_s'] = ts_s
data['F_evals'] = F_evaluations
data['A_evals'] = A_evaluations

with open(sys.argv[1]+'.json', 'w') as f:
    json.dump(data, f)


