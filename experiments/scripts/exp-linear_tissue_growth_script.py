#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as scpi
import scipy.interpolate as sci

import os 
import json

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


# In[2]:


s = 1.0
rA = 1.5
params_cubic = {'mu': 5.7, 's': s, 'rA': rA}
dim = 3
seed = 67
tf = 1500.0 # in order to reach 1000 cells (starting at 512)
frequency = 1.0


# In[3]:


cell_list = ut.setup_locally_compressed_monolayer(6, 6, seed=seed)


# In[4]:


ut.plot_2d_population(cell_list)


# In[5]:


model = cbmos.CBModel(ff.Cubic(), ef.solve_ivp, 2)


# In[6]:


ts, history = model.simulate(cell_list, [0, 3], params_cubic, {})


# In[7]:


ut.plot_2d_population(history[-1])


# In[4]:


#ancestor = [cl.Cell(0, [0.0, 0.0, 0.0])]

#start with tissue (~500 cells?)
coords = ut.generate_hcp_coordinates(8, 8, 8)
tissue = [cl.Cell(i, [x, y, z]) for i, (x, y, z) in enumerate(coords)]

event_times = np.arange(0.0, tf, frequency)

events = [ev.PickRandomCellToDivideEvent(time) for time in event_times]


# In[5]:


8*8*8


# In[6]:


target_cell_counts = [520, 530, 540, 550]
max_execution_time = 10*60 # 10 minutes in seconds


# In[7]:


#algorithms
#algorithms = ['EF_glob_adap_acc', 'EF_glob_adap_stab' ,  'EF_local_adap', 'EB_global_adap' ]
algorithms = ['EF_glob_adap_acc', 'EF_glob_adap_stab' ,  'EF_local_adap' ]


models = {'EF_glob_adap_acc': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),
          'EF_glob_adap_stab': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim),  
          'EF_local_adap': cbmos.CBModel(ff.Cubic(), ef.solve_ivp, dim), 
          'EB_global_adap': cbmos.CBModel(ff.Cubic(), eb.solve_ivp, dim) }

eta = 1e-4
eps = 0.005

params = {'EF_glob_adap_acc': {'eps': eps, 'eta': eta},
          'EF_glob_adap_stab': {'eps': eps, 'eta': eta, 
                                'jacobian': models['EF_glob_adap_stab'].jacobian, 'force_args': params_cubic, 
                                'always_calculate_Jacobian': True, 
                                #'calculate_eigenvalues': True
                               }, 
          'EF_local_adap': {'eps': eps, 'eta': eta, 
                            'jacobian': models['EF_local_adap'].jacobian, 'force_args': params_cubic, 
                            'always_calculate_Jacobian': True, 
                            'local_adaptivity': True, 'm0': 14, 
                            #'calculate_eigenvalues': True
                            #'dim': dim, 
                            'rA': rA 
                           }, 
          'EB_global_adap': {'eps': eps, 'eta': eta, 'jacobian': models['EB_global_adap'].jacobian, 'force_args': params_cubic}
         }


labels = {'EF_glob_adap_acc': 'Alg. I (GAFE)', 
          'EF_glob_adap_stab': 'Alg. II (GAFES)' ,  
          'EF_local_adap': 'Alg. III (LAFE)', 
          'EB_global_adap': 'Alg. IV (GABE)'
         }


# In[15]:


data = {}
for alg in algorithms:
    print(alg)
    ts, history = models[alg].simulate(tissue, 
                                      [0, tf], 
                                      params_cubic, 
                                      params[alg],  
                                      seed=seed, 
                                      event_list=events,
                                      n_target_cells=target_cell_counts,
                                      max_execution_time=max_execution_time,
                                      throw_away_history=True
                                     )
    data[alg] = np.array(models[alg].target_cell_count_checkpoints)


# In[9]:


data['EF_glob_adap_acc']


# In[10]:


#algorithms = ['EF_glob_adap_acc']


# In[16]:


for alg in algorithms:
    plt.plot(data[alg][:, 1], data[alg][:, 2], label=labels[alg])
    
plt.legend()


# In[18]:


#ancestor = [cl.Cell(0, [0.0, 0.0, 0.0])]

#start with tissue (~500 cells?)
size = 9
coords = ut.generate_hcp_coordinates(size, size, size)
tissue = [cl.Cell(i, [x, y, z]) for i, (x, y, z) in enumerate(coords)]

frequency = 0.5
event_times = np.arange(0.0, tf, frequency)

events = [ev.PickRandomCellToDivideEvent(time) for time in event_times]

target_cell_counts = [730, 740, 750, 760]
max_execution_time = 10*60 # 10 minutes in seconds


# In[19]:


9**3


# In[22]:


data = {}
for alg in algorithms:
    print(alg)
    ts, history = models[alg].simulate(tissue, 
                                      [0, tf], 
                                      params_cubic, 
                                      params[alg],  
                                      seed=seed, 
                                      event_list=events,
                                      n_target_cells=target_cell_counts,
                                      max_execution_time=max_execution_time,
                                      throw_away_history=True
                                     )
    plt.plot(ts[:-2], np.diff(ts[:-1]))
    data[alg] = np.array(models[alg].target_cell_count_checkpoints)


# In[25]:


plt.plot(ts[:-2], np.diff(ts[:-1]))


# In[23]:


for alg in algorithms:
    plt.plot(data[alg][:, 1], data[alg][:, 2], label=labels[alg])
    
plt.legend()


# In[26]:


#ancestor = [cl.Cell(0, [0.0, 0.0, 0.0])]

#start with tissue (~500 cells?)
size = 10
coords = ut.generate_hcp_coordinates(size, size, size)
tissue = [cl.Cell(i, [x, y, z]) for i, (x, y, z) in enumerate(coords)]

frequency = 0.5
event_times = np.arange(0.0, tf, frequency)

events = [ev.PickRandomCellToDivideEvent(time) for time in event_times]

target_cell_counts = [1010, 1020, 1030, 1040]
max_execution_time = 10*60 # 10 minutes in seconds


# In[27]:


data = {}
for alg in algorithms:
    print(alg)
    ts, history = models[alg].simulate(tissue, 
                                      [0, tf], 
                                      params_cubic, 
                                      params[alg],  
                                      seed=seed, 
                                      event_list=events,
                                      n_target_cells=target_cell_counts,
                                      max_execution_time=max_execution_time,
                                      throw_away_history=True
                                     )
    plt.plot(ts[:-2], np.diff(ts[:-1]))
    data[alg] = np.array(models[alg].target_cell_count_checkpoints)


# GAFE is using so much larger steps... How does this affect the accuracy?

# In[28]:


for alg in algorithms:
    plt.plot(data[alg][:, 1], data[alg][:, 2], label=labels[alg])
    
plt.legend()


# Calculate slopes and add fixed time stepping

# In[29]:


#start with tissue (~500 cells?)
size = 10
coords = ut.generate_hcp_coordinates(size, size, size)
tissue = [cl.Cell(i, [x, y, z]) for i, (x, y, z) in enumerate(coords)]

frequency = 0.3
event_times = np.arange(0.0, tf, frequency)

events = [ev.PickRandomCellToDivideEvent(time) for time in event_times]

target_cell_counts = [1010, 1020, 1030]
max_execution_time = 10*60 # 10 minutes in seconds


# In[30]:


data = {}
for alg in algorithms:
    print(alg)
    ts, history = models[alg].simulate(tissue, 
                                      [0, tf], 
                                      params_cubic, 
                                      params[alg],  
                                      seed=seed, 
                                      event_list=events,
                                      n_target_cells=target_cell_counts,
                                      max_execution_time=max_execution_time,
                                      throw_away_history=True
                                     )
    plt.plot(ts[:-2], np.diff(ts[:-1]))
    data[alg] = np.array(models[alg].target_cell_count_checkpoints)


# In[31]:


for alg in algorithms:
    plt.plot(data[alg][:, 1], data[alg][:, 2], label=labels[alg])
    
plt.legend()


# The smaller the time between cell divisions, the more the local time stepping algorithm pays off compared to GAFES. Can it ever reach GAFE? There also has to be a lower limit when fixed time stepping becomes more efficient. What if after every time step a cell divides somewhere? I guess the time between divisions should not be smaller than sqrt(m)*dt_GAFE. Or else LAFE cannot benefit. 
# Right now m=14, dt_GAFE = 0.007, ie. the minimal time should be no smaller than 0.026. Which is an order of magnitude smaller than what I tried out so far.

# In[32]:


np.sqrt(14)*0.007


# In[33]:


#start with tissue (~500 cells?)
size = 10
coords = ut.generate_hcp_coordinates(size, size, size)
tissue = [cl.Cell(i, [x, y, z]) for i, (x, y, z) in enumerate(coords)]

frequency = 0.03
event_times = np.arange(0.0, tf, frequency)

events = [ev.PickRandomCellToDivideEvent(time) for time in event_times]

target_cell_counts = [1010, 1020, 1030]
max_execution_time = 10*60 # 10 minutes in seconds


# In[34]:


data = {}
for alg in algorithms:
    print(alg)
    ts, history = models[alg].simulate(tissue, 
                                      [0, tf], 
                                      params_cubic, 
                                      params[alg],  
                                      seed=seed, 
                                      event_list=events,
                                      n_target_cells=target_cell_counts,
                                      max_execution_time=max_execution_time,
                                      throw_away_history=True
                                     )
    plt.plot(ts[:-2], np.diff(ts[:-1]))
    data[alg] = np.array(models[alg].target_cell_count_checkpoints)


# Okay, LAFE is taking weird steps here.

# In[35]:


for alg in algorithms:
    plt.plot(data[alg][:, 1], data[alg][:, 2], label=labels[alg])
    
plt.legend()


# I guess one can never beat GAFE... 

# Can I do adding 20 cells for a population of 5000 cells locally? Or do I run out of memory there? (And would have to run it on rackham?)

# In[12]:


#start with tissue (~500 cells?)
size = 13 #2197
coords = ut.generate_hcp_coordinates(size, size, size)
tissue = [cl.Cell(i, [x, y, z]) for i, (x, y, z) in enumerate(coords)]

frequency = 0.05
event_times = np.arange(0.0, tf, frequency)

events = [ev.PickRandomCellToDivideEvent(time) for time in event_times]

target_cell_counts = [2200, 2210, 2220]
max_execution_time = 10*60 # 10 minutes in seconds


# In[13]:


data = {}
for alg in algorithms:
    print(alg)
    ts, history = models[alg].simulate(tissue, 
                                      [0, tf], 
                                      params_cubic, 
                                      params[alg],  
                                      seed=seed, 
                                      event_list=events,
                                      n_target_cells=target_cell_counts,
                                      max_execution_time=max_execution_time,
                                      throw_away_history=True
                                     )
    plt.plot(ts[:-2], np.diff(ts[:-1]))
    data[alg] = np.array(models[alg].target_cell_count_checkpoints)


# In[ ]:


for alg in algorithms:
    plt.plot(data[alg][:, 1], data[alg][:, 2], label=labels[alg])
    
plt.legend()


# In[11]:


13*13*13


# In[ ]:




