#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:57:42 2019

@author: Sonja Mathias
"""

import numpy as np


## Linear spring
def linear_spring(r, **kwargs): 
    """
    Linear spring force function.
    
    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0  
    
    """
    if 'mu' in kwargs:
        mu = kwargs['mu']
    else:
        mu = 1.0
    if 's' in kwargs:
        s = kwargs['s']
    else:
        s = 1.0
    mu*(r-s)

## Morse
def morse(r, **kwargs): 
    """
    Morse potential. (slope at r = s  is 4*a*m)
    
    Parameters:
      m: maximum value, default 1.0
      a: controls the bredth of the potential, default 1.0
      s: rest length, default 1.0  
    
    """
    if 'm' in kwargs:
        m = kwargs['m']
    else:
        m = 1.0
    if 'a' in kwargs:
        a = kwargs['a']
    else:
        a = 1.0
    if 's' in kwargs:
        s = kwargs['s']
    else:
        s = 1.0
    - m*(np.exp(-2*a*(r-s-np.log(2)/a)) -2*np.exp(-a*(r-s-np.log(2)/a)))
    
## Lennard-Jones
def lennard_jones(r, **kwargs): 
    """
    Lennard-Jones potential
    
    Parameters:
      m: maximum value, default 1.0
      s: rest length, default 1.0  
    
    """
    if 'm' in kwargs:
        m = kwargs['m']
    else:
        m = 1.0
    if 's' in kwargs:
        s = kwargs['s']
    else:
        s = 1.0
   -4*m*(np.power(s/r,12)-np.power(s/r,6))
    

## Lennard-Jones potential
# m: maximum value
lennard_jones = lambda m, r: -4*m*(np.power(s/r,12)-np.power(s/r,6))

## Generalized linear spring
#rc = s+0.5 # r_cut
gls = lambda mu, a, r: np.where(r<=s, mu*np.log(1+(r-s)), mu*(r-s)*np.exp(-a*(r-s)))
#gls2 = lambda muR, muA, a, r: np.where(r<=s, muR*np.log(1+(r-s)), np.where(r<=rc, muA*(r-s)*np.exp(-a*(r-s)), 0.))

logarithmic = lambda mu, r: np.where(r<=s, mu*np.log(1+(r-s)),0)

linear_logarithmic_product = lambda mu, r: np.where(r<=s, -mu*(r-s)*np.log(1+(r-s)),0)

linear_exponential_product = lambda mu, a, r: mu*(r-s)*np.exp(-a*(r-s))

## hard-core model
h_N = 0.3 # sum of incompressible nuclei of two cells
hard_core = lambda mu, r: np.where(r<=s-h_N, np.inf, np.where(r<=s, mu*(r-s)/(r-(s-h_N)), 0.))
hard_core2 = lambda mu, r: np.where(r<=h_N, np.inf, np.where(r<=s, mu*(r-s)/(r-h_N), 0.))


## Polynomial law for repulsion (PhysiCell)
r_R = s+0.2
polynomial_repulsion = lambda mu, n, r : np.where(r<=r_R, -mu*(1-r/r_R)**(n+1), 0.)
## Polynomial law for adhesion (PhysiCell)
# r_A: maximum adhesive distance
r_A = s+0.5
polynomial_adhesion = lambda mu, n, r : np.where(r<=r_A, mu*(1-r/r_A)**(n+1), 0.)

## cubic law (MecaGen)
rm = s+0.5
cubic_law = lambda muR, muA, r: np.where(r<=s, muR*(r-rm)**2*(r-s), np.where(r<=rm, muA*(r-rm)**2*(r-s), 0.)) 

