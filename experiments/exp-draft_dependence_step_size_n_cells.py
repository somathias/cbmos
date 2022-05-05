#!/usr/bin/env python
# coding: utf-8

# # Single proliferation event within larger spheroid

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as scpi

import os

import cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.solvers.euler_backward as eb
import cbmos.cell as cl
import cbmos.utils as ut

plt.style.use('seaborn-whitegrid')
plt.style.use('tableau-colorblind10')
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (6.75, 5),
          'lines.linewidth': 3.0,
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large',
          'legend.fontsize': 'xx-large',
          'font.size': 11,
          'font.family': 'serif',
          'mathtext.fontset': 'dejavuserif',
          'axes.titlepad': 12,
          'axes.labelpad': 12}
plt.rcParams.update(params)


# In[2]:


# Simulation parameters
s = 1.0    # rest length
tf = 5.0  # final time
rA = 1.5   # maximum interaction distance

dim = 3
seed = 67

t_data = [0,tf]

#force_names = ['cubic', 'pw. quad.', 'GLS']
force = 'cubic'
# parameters fitted to relaxation time t=1.0h
params_cubic = {"mu": 5.70, "s": s, "rA": rA}

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = {'cubic': defcolors[0], 'pw. quad.': defcolors[5], 'GLS': defcolors[6]}


# In[3]:


#algorithms
algorithms = ['EF_glob_adap_acc', 'EF_glob_adap_stab' ,  'EF_local_adap', 'EB_global_adap' ]

models = {'EF_glob_adap_acc': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_glob_adap_stab': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_local_adap': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EB_global_adap': cbmos.CBModel(ff.Cubic(), eb.solve_ivp, dim) }

eta = 1e-4

params = {'EF_glob_adap_acc': {'eta': eta},
          'EF_glob_adap_stab': {'eta': eta, 'jacobian': models['EF_glob_adap_stab'].jacobian, 'force_args': params_cubic,
                                'always_calculate_Jacobian': True},
          'EF_local_adap': {'eta': eta, 'jacobian': models['EF_local_adap'].jacobian, 'force_args': params_cubic,
                            'always_calculate_Jacobian': True, 'local_adaptivity': True, 'm0': 14, 'dim': dim, 'rA': rA},
          'EB_global_adap': {'eta': eta, 'jacobian': models['EB_global_adap'].jacobian, 'force_args': params_cubic}
         }

# ##  Dependence of initial step dt_0 and dt at steady state on spheroid size
#
# Can I run this locally, or should I run this on rackham? Can I maybe run up to 1000 as a script locally? If necessary if I allocate more RAM to the VM?
#
# I want to
# - consider all algorithms
# - over a range of spheroid sizes
# - average over several seeds and hence cell division directions
# - calculate dt_0 as ts[1]-ts[0] and dt_st = ts[-2] - ts[-3], since the last time step may be cut short to reach tf exactly. Might need to average over several last steps.
#
# Do I want to plot the level distribution of the locally adaptive algorithm here?

# In[6]:


eps = 0.005
for alg in algorithms:
    params[alg]['eps'] = eps


# In[7]:


dt_0s = {}
dt_sts = {}

sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
n_seed = 5

for alg in algorithms:

    dt_0s[alg] = []
    dt_sts[alg] = []

    for l in sizes:

        dt_0 = 0.
        dt_st = 0.

        for seed in range(n_seed):
            sheet = ut.setup_locally_compressed_spheroid(l,l,l, seed=seed)
            ts, history = models[alg].simulate(sheet, t_data, params_cubic, params[alg], seed=seed)

            dt_0 += ts[1]-ts[0]
            dt_st += ts[-2]-ts[-3]

        dt_0s[alg].append(dt_0/n_seed)
        dt_sts[alg].append(dt_st/n_seed)


# In[18]:


import json
with open('dt_0s_eps0.005_m14.json', 'w') as f:
    json.dump(dt_0s, f)

with open('dt_sts_eps0.005_m14.json', 'w') as f:
    json.dump(dt_sts, f)


