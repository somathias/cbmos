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
    size = 10
else:
    size = int(sys.argv[2])
print(size)


if len(sys.argv) < 4:
    eps = 0.01
else:
    eps = float(sys.argv[3])
print(eps)

if len(sys.argv) < 5:
    m = 7
else:
    m = int(sys.argv[4])
print(m)

seed=17
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


sheet = ut.setup_locally_compressed_spheroid(size, size, size, seed=seed)
print(len(sheet))

algorithms = ['EF_glob_adap_acc', 'EF_glob_adap_stab' ,  'EF_local_adap', 'EF_local_adap_stab', 'EB_global_adap', 'fixed_dt' ]
#algorithms = ['EF_glob_adap_stab', 'EF_local_adap', 'EF_local_adap_stab']

models = {'EF_glob_adap_acc': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_glob_adap_stab': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_local_adap': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_local_adap_stab': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EB_global_adap': cbmos.CBModel(ff.Cubic(), eb.solve_ivp, dim),
          'fixed_dt': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim) }

eta = 1e-4

params = {'EF_glob_adap_acc': {'eps':eps, 'eta': eta},
          'EF_glob_adap_stab': {'eps':eps, 'eta': eta, 'jacobian': models['EF_glob_adap_stab'].jacobian, 'force_args': params_cubic, 'always_calculate_Jacobian': True},
          'EF_local_adap': {'eps':eps, 'eta': eta, 'local_adaptivity': True, 'm0': m, 'dim': dim, 'rA': rA},
          'EF_local_adap_stab': {'eps':eps, 'eta': eta, 'jacobian': models['EF_local_adap_stab'].jacobian, 'force_args': params_cubic,
                            'always_calculate_Jacobian': True, 'local_adaptivity': True, 'm0': m, 'dim': dim, 'rA': rA},
          'EB_global_adap': {'eps':eps, 'eta': eta, 'jacobian': models['EB_global_adap'].jacobian, 'force_args': params_cubic},
          #'fixed_dt': {'dt': 0.011758452836496444}
          'fixed_dt': {}
         }

for alg in algorithms:
    params[alg]['out'] = alg
    params[alg]['measure_wall_time'] = True


n = 20

dt_f = 0.007132468456522988

for alg in algorithms:
    # burn in
    try:
        os.remove('exec_times'+alg+'.txt')
    except FileNotFoundError:
        pass
        #print('Nothing to delete.')
    try:
        os.remove('F_evaluations'+alg+'.txt')
    except FileNotFoundError:
        pass
        #print('Nothing to delete.')
    try:
        os.remove('A_evaluations'+alg+'.txt')
    except FileNotFoundError:
        pass
        #print('Nothing to delete.')

    if alg is 'fixed_dt':
        params[alg]['dt'] = dt_f
        print(dt_f)

    ts, history = models[alg].simulate(sheet, t_data, params_cubic, params[alg], seed=seed)

    if alg is 'EF_glob_adap_acc':
        dt_f = ts[1] - ts[0]
        print(dt_f)

    data = {}
    for i in range(n):
        try:
            os.remove('exec_times'+alg+'.txt')
        except FileNotFoundError:
            pass
            #print('Nothing to delete.')
        try:
            os.remove('F_evaluations'+alg+'.txt')
        except FileNotFoundError:
            pass
            #print('Nothing to delete.')
        try:
            os.remove('A_evaluations'+alg+'.txt')
        except FileNotFoundError:
            pass
            #print('Nothing to delete.')

        ts, history = models[alg].simulate(sheet, t_data, params_cubic, params[alg], seed=seed)

        with open('exec_times'+alg+'.txt', 'r') as f:
            exec_times = np.array(np.loadtxt(f))[:, 1]
            if i==0:
                data['exec_times'] = exec_times
                data['min_exec_times'] =  exec_times
                data['max_exec_times'] =  exec_times
            else:
                data['exec_times'] += exec_times
                data['min_exec_times'] =  np.minimum(data['min_exec_times'], exec_times)
                data['max_exec_times'] =  np.maximum(data['max_exec_times'], exec_times)

    data['exec_times'] = list(data['exec_times']/n)
    data['min_exec_times'] = list(data['min_exec_times'])
    data['max_exec_times'] = list(data['max_exec_times'])


    with open('F_evaluations'+alg+'.txt', 'r') as f:
        data['F_evals'] = list(np.array(np.loadtxt(f))[:, 1])
    if alg in ['EF_glob_adap_stab', 'EF_local_adap_stab', 'EB_global_adap']:
        with open('A_evaluations'+alg+'.txt', 'r') as f:
            data['A_evals'] = list(np.array(np.loadtxt(f))[:, 1])
    else:
        data['A_evals'] = list(np.zeros(len(ts)))

    data['ts'] = ts

    with open(sys.argv[1]+'_'+alg+'.json', 'w') as f:
        json.dump(data, f)

    print('Done with '+alg)
