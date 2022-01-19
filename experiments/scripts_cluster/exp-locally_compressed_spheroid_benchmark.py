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


sheet = ut.setup_locally_compressed_spheroid(6,6,6)

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
            F_evaluations[out] = np.array(np.loadtxt(f))
exec_times[out] = exec_times[out]/n
ts_s[out] = ts
A_evaluations[out] = np.vstack([ts, np.zeros(len(ts))]).T

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
            F_evaluations[out] = np.array(np.loadtxt(f))
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))
exec_times[out] = exec_times[out]/n
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
            F_evaluations[out] = np.array(np.loadtxt(f))
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))
exec_times[out] = exec_times[out]/n
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
            F_evaluations[out] = np.array(np.loadtxt(f))
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))
exec_times[out] = exec_times[out]/n
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
            F_evaluations[out] = np.array(np.loadtxt(f))
        with open('A_evaluations'+out+'.txt', 'r') as f:
            A_evaluations[out] = np.array(np.loadtxt(f))
exec_times[out] = exec_times[out]/n
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
            F_evaluations[out] = np.array(np.loadtxt(f))
exec_times[out] = exec_times[out]/n
ts_s[out] = ts
A_evaluations[out] = np.vstack([ts, np.zeros(len(ts))]).T

data['exec_times'] = exec_times
data['ts_s'] = ts_s
data['F_evals'] = F_evaluations
data['A_evals'] = A_evaluations

with open(sys.argv[1]+'.json', 'w') as f:
    json.dump(data, f)


