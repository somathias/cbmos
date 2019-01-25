#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:13:35 2019

@author: Sonja Mathias
"""
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
import matplotlib.pyplot as plt

plt.style.use('seaborn')


dt = 0.01

def solve_ivp(fun, t_span, y0, t_eval=None):
    
    t0, tf = float(t_span[0]), float(t_span[1])
    
    t = t0
    y = y0
    
    ts = [t]
    ys = [y]
    
    while t + dt <= tf:
        y = y + dt*fun(t,y)
        t = t + dt
                
        ts.append(t)
        ys.append(y) 

    return OdeResult(t=ts, y=ys)

if __name__ == "__main__":  
    
    # stability region for Euler forward for this problem is h<2/50=0.04
    @np.vectorize
    def fun(t,y):
        return -50*y
    
    t_span = (0,1)
    y0 = 1
    
    sol = solve_ivp(fun, t_span, y0 )
    
    plt.figure()
    plt.plot(sol.t, sol.y)
    
    
        