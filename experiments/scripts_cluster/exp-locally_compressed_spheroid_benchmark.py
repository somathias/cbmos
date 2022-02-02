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
import cbmos.solvers.euler_backward as eb
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


sheet = ut.setup_locally_compressed_spheroid(10, 10, 10, seed=seed)
#sheet = ut.setup_locally_compressed_spheroid(3, 3, 3, seed=seed)


data = {}

eps = 0.01
eta = 0.0001
cbmodel = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim, separation)
cbmodel_eb = cbmos.CBModel(ff.Cubic(), eb.solve_ivp, dim, separation)
#algorithms = ['glob_adap_acc', 'glob_adap_stab' ,  'local_adap', 'fixed_dt' ]
exec_times = {}
F_evaluations = {}
A_evaluations = {}
ts_s = {}

n = 25

out = 'EF_glob_adap_acc'
for i in range(n):
    try:
        os.remove('exec_times'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+out+'.txt')
    except FileNotFoundError:
        print('Nothing to delete.')
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, 'eta': eta, "out": out, "measure_wall_time": True}, seed=seed)

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

out = 'EF_glob_adap_stab'
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
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, 'eta': eta, "out": out,"jacobian": cbmodel.jacobian, "force_args": params_cubic, "always_calculate_Jacobian": True, "measure_wall_time": True}, seed=seed)

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

#out = 'glob_adap_stab_stop_Jac_update'
#for i in range(n):
#    try:
#        os.remove('exec_times'+out+'.txt')
#    except FileNotFoundError:
#        print('Nothing to delete.')
#    try:
#        os.remove('F_evaluations'+out+'.txt')
#    except FileNotFoundError:
#        print('Nothing to delete.')
#    try:
#        os.remove('A_evaluations'+out+'.txt')
#    except FileNotFoundError:
#        print('Nothing to delete.')
#    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, 'eta': eta, "out": out,"jacobian": cbmodel.jacobian, "force_args": params_cubic, "measure_wall_time": True}, seed=seed)
#
#    with open('exec_times'+out+'.txt', 'r') as f:
#        if i==0:
#            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
#        else:
#            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
#    if i==0:
#        with open('F_evaluations'+out+'.txt', 'r') as f:
#            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
#        with open('A_evaluations'+out+'.txt', 'r') as f:
#            A_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
#exec_times[out] = list(exec_times[out]/n)
#F_evaluations[out] = list(F_evaluations[out])
#A_evaluations[out] = list(A_evaluations[out])
#ts_s[out] = ts

# local adaptivity
out = 'EF_local_adap'
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
    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, 'eta': eta,  "out": out, "jacobian": cbmodel.jacobian, "force_args": params_cubic, "always_calculate_Jacobian": True, "local_adaptivity": True, "measure_wall_time": True}, seed=seed)

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

#out = 'local_adap_stop_Jac_update'
#for i in range(n):
#    try:
#        os.remove('exec_times'+out+'.txt')
#    except FileNotFoundError:
#        print('Nothing to delete.')
#    try:
#        os.remove('F_evaluations'+out+'.txt')
#    except FileNotFoundError:
#        print('Nothing to delete.')
#    try:
#        os.remove('A_evaluations'+out+'.txt')
#    except FileNotFoundError:
#        print('Nothing to delete.')
#    ts, history = cbmodel.simulate(sheet, t_data, params_cubic, {"eps": eps, 'eta': eta, "out": out, "jacobian": cbmodel.jacobian, "force_args": params_cubic, "local_adaptivity": True, "measure_wall_time": True}, seed=seed)
#
#    with open('exec_times'+out+'.txt', 'r') as f:
#        if i==0:
#            exec_times[out] = np.array(np.loadtxt(f))[:, 1]
#        else:
#            exec_times[out] += np.array(np.loadtxt(f))[:, 1]
#    if i==0:
#        with open('F_evaluations'+out+'.txt', 'r') as f:
#            F_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
#        with open('A_evaluations'+out+'.txt', 'r') as f:
#            A_evaluations[out] = np.array(np.loadtxt(f))[:, 1]
#exec_times[out] = list(exec_times[out]/n)
#F_evaluations[out] = list(F_evaluations[out])
#A_evaluations[out] = list(A_evaluations[out])
#ts_s[out] = ts

out = 'EB_global_adap'
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
    ts, history = cbmodel_eb.simulate(sheet, t_data, params_cubic, {"eps": eps, 'eta': eta, "out": out, "measure_wall_time": True}, seed=seed)

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

out = 'EB_global_adap_jac'
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
    ts, history = cbmodel_eb.simulate(sheet, t_data, params_cubic, {"eps": eps, 'eta': eta, 'jacobian': cbmodel_eb.jacobian, 'force_args': params_cubic, "out": out, "measure_wall_time": True}, seed=seed)

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
DT = ts_s['EF_glob_adap_acc'][1] - ts_s['EF_glob_adap_acc'][0]
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


