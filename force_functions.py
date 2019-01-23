#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:57:42 2019

@author: Sonja Mathias
"""

import numpy as np


## Linear spring
def linear_spring(r, parameters={}): 
    """
    Linear spring force function.
    
    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0  
    
    """
    if "mu" in parameters:
        mu = parameters["mu"]
    else:
        mu = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    return mu*(r-s)

## Morse
def morse(r, parameters={}): 
    """
    Morse potential. (slope at r = s  is 4*a*m)
    
    Parameters:
      m: maximum value, default 1.0
      a: controls the bredth of the potential, default 1.0
      s: rest length, default 1.0  
    
    """
    if "m" in parameters:
        m = parameters["m"]
    else:
        m = 1.0
    if "a" in parameters:
        a = parameters["a"]
    else:
        a = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    return - m*(np.exp(-2*a*(r-s-np.log(2)/a)) -2*np.exp(-a*(r-s-np.log(2)/a)))
    
## Lennard-Jones
def lennard_jones(r, parameters={}): 
    """
    Lennard-Jones potential
    
    Parameters:
      m: maximum value, default 1.0
      s: rest length, default 1.0  
    
    """
    if "m" in parameters:
        m = parameters["m"]
    else:
        m = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    return -4*m*(np.power(s/r,12)-np.power(s/r,6))
    
## Linear-exponential
def linear_exponential(r, parameters={}): 
    """
    Linear exponential force function
    
    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0  
      a: controls the bredth of the potential, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5
 
    
    """
    if "mu" in parameters:
        mu = parameters["mu"]
    else:
        mu = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    if "a" in parameters:
        a = parameters["a"]
    else:
        a = 1.0
    if "rA" in parameters:
        rA = parameters["rA"]
    else:
        rA = 1.5
    return np.where(r<=rA, mu*(r-s)*np.exp(-a*(r-s)), 0.)

## cubic
def cubic(r, parameters={}): 
    """
    Cubic force function
    
    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0  
      rA: maximum interaction distance (cutoff value), default 1.5
 
    
    """
    if "mu" in parameters:
        mu = parameters["mu"]
    else:
        mu = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    if "rA" in parameters:
        rA = parameters["rA"]
    else:
        rA = 1.0
    return np.where(r<=rA, mu*(r-rA)**2*(r-s), 0.)
    
## general polynomial
def general_polynomial(r, parameters={}): 
    """
    General polynomial force function
    
    Parameters:
      muA: spring stiffness coefficient for adhesion, default 1.0
      muR: spring stiffness coefficient for repulsion, default 1.0
      rA: maximum adhesive interaction distance (cutoff value), default 1.5
      rR: maximum repulsive interaction distance (cutoff value), default 1.5
      n: exponent adhesive part
      m: exponent repulsive part 
    
    """
    if "muA" in parameters:
        muA = parameters["muA"]
    else:
        muA = 1.0
    if "muR" in parameters:
        muR = parameters["muR"]
    else:
        muR = 1.0
    if "rA" in parameters:
        rA = parameters["rA"]
    else:
        rA = 1.0
    if "rR" in parameters:
        rR = parameters["rR"]
    else:
        rR = 1.0
    if "n" in parameters:
        n = parameters["n"]
    else:
        n = 1.0
    if "p" in parameters:
        p = parameters["p"]
    else:
        p = 1.0
    return np.where(r<=rR, muA*(1-r/rA)(n+1)+muR*(1-r/rR)(p+1), np.where(r<=rA, muA*(1-r/rA)(n+1), 0.))

## logarithmic
def logarithmic(r, parameters={}): 
    """
    Logarithmic force function
    
    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0  
    
    """
    if "mu" in parameters:
        mu = parameters["mu"]
    else:
        mu = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    return np.where(r<=s, mu*np.log(1+(r-s)),0.)
    
## linear-logarithmic
def linear_logarithmic(r, parameters={}): 
    """
    Linear logarithmic force function
    
    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0  
    
    """
    if "mu" in parameters:
        mu = parameters["mu"]
    else:
        mu = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    return np.where(r<=s, -mu*(r-s)*np.log(1+(r-s)),0.)

## hard-core model
def hard_core(r, parameters={}): 
    """
    Hard-core model force function
    
    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0  
      rN: radius of nucleus, default 0.3
    
    """
    if "mu" in parameters:
        mu = parameters["mu"]
    else:
        mu = 1.0
    if "s" in parameters:
        s = parameters["s"]
    else:
        s = 1.0
    if "rN" in parameters:
        rN = parameters["rN"]
    else:
        rN = 0.3
    return np.where(r<=s-2*rN, np.inf, np.where(r<=s, mu*(r-s)/(r-(s-2*rN)), 0.))








