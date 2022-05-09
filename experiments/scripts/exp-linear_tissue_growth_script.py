#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as scpi
import scipy.interpolate as sci

import os
import json
import sys
from datetime import datetime

import cbmos
import cbmos.force_functions as ff
import cbmos.solvers.euler_forward as ef
import cbmos.solvers.euler_backward as eb
import cbmos.cell as cl
import cbmos.utils as ut
import cbmos.events as ev

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

from matplotlib import cm
import matplotlib.colors as mcolors

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if len(sys.argv) < 2:
    raise IOError("Must provide output file")

if len(sys.argv) < 3:
    size = 10
else:
    size = int(sys.argv[2])
print(size)

if len(sys.argv) < 4:
    time_between_events = 0.5
else:
    time_between_events = float(sys.argv[3])
print(time_between_events)


seed=17
npr.seed(seed)

# In[2]:


s = 1.0
rA = 1.5
params_cubic = {'mu': 5.7, 's': s, 'rA': rA}
dim = 3

tf = 1500.0 # in order to reach 1000 cells (starting at 512)



#algorithms
algorithms = ['EF_glob_adap_acc', 'EF_glob_adap_stab' ,  'EF_local_adap', 'EB_global_adap', 'fixed_dt' ]
#algorithms = ['EF_glob_adap_acc', 'EF_glob_adap_stab' ,  'EF_local_adap' ]


models = {'EF_glob_adap_acc': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_glob_adap_stab': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_local_adap': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EB_global_adap': cbmos.CBModel(ff.Cubic(), eb.solve_ivp, dim),
          'fixed_dt': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim) }

eta = 1e-4
eps = 0.005
dt_f = 0.007132468456522988


params = {'EF_glob_adap_acc': {'eps': eps, 'eta': eta},
          'EF_glob_adap_stab': {'eps': eps, 'eta': eta,
                                'jacobian': models['EF_glob_adap_stab'].jacobian, 'force_args': params_cubic,
                                #'calculate_eigenvalues': True
                               },
          'EF_local_adap': {'eps': eps, 'eta': eta,
                            'jacobian': models['EF_local_adap'].jacobian, 'force_args': params_cubic,
                            'local_adaptivity': True, 'm0': 14,
                            #'calculate_eigenvalues': True
                            'dim': dim,
                            'rA': rA
                           },
          'EB_global_adap': {'eps': eps, 'eta': eta, 'jacobian': models['EB_global_adap'].jacobian, 'force_args': params_cubic},
          'fixed_dt': { 'dt': dt_f}
         }


#start with tissue
coords = ut.generate_hcp_coordinates(size, size, size)
tissue = [cl.Cell(i, [x, y, z]) for i, (x, y, z) in enumerate(coords)]

event_times = np.arange(0.0, tf, time_between_events)

events = [ev.PickRandomCellToDivideEvent(time) for time in event_times]

initial_cell_count = size**3
target_cell_counts = [initial_cell_count + 10]
max_execution_time = 40*60 # 40 minutes in seconds

repetitions = 4

for alg in algorithms:
    print(alg)

    if alg is 'fixed_dt':
        params[alg]['dt'] = dt_f
        print(dt_f)

    data = {}
    counts = []
    for i in range(repetitions):
        ts, _ = models[alg].simulate(tissue,
                                          [0, tf],
                                          params_cubic,
                                          params[alg],
                                          seed=seed,
                                          event_list=events,
                                          n_target_cells=target_cell_counts,
                                          max_execution_time=max_execution_time,
                                          throw_away_history=True
                                         )
        counts.append(models[alg].target_cell_count_checkpoints)
    if alg is 'EF_glob_adap_acc':
        dt_f = ts[1] - ts[0]
        print(dt_f)


    data['ts'] = ts
    data['counts'] = counts

    with open(sys.argv[1]+'_'+str(size)+'_'+str(time_between_events)+'_'+alg+'.json', 'w') as f:
            json.dump(data, f)

    print('Done with '+alg)





