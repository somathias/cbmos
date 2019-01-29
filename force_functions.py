#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:57:42 2019

@author: Sonja Mathias
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


# Linear spring
@np.vectorize
def linear_spring(r, mu=1.0, s=1.0):
    """
    Linear spring force function.

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if not r:
        return 0.
    return mu*(r-s)


# Morse
@np.vectorize
def morse(r, m=1.0, a=1.0, s=1.0):
    """
    Morse potential. (slope at r = s  is 4*a*m)

    Parameters:
      m: maximum value, default 1.0
      a: controls the bredth of the potential, default 1.0
      s: rest length, default 1.0

    """
    if not r:
        return 0.
    return - m*(np.exp(-2*a*(r-s-np.log(2)/a))-2*np.exp(-a*(r-s-np.log(2)/a)))


# Lennard-Jones
@np.vectorize
def lennard_jones(r, m=1.0, s=1.0):
    """
    Lennard-Jones potential

    Parameters:
      m: maximum value, default 1.0
      s: rest length, default 1.0

    """
    if not r:
        return 0.
    return -4*m*(np.power(s/r, 12)-np.power(s/r, 6))


# Linear-exponential
@np.vectorize
def linear_exponential(r, mu=15.0, s=1.0, a=5.0, rA=1.5):
    """
    Linear exponential force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0
      a: controls the bredth of the potential, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5


    """
    if not r:
        return 0.
    return np.where(r <= rA, mu*(r-s)*np.exp(-a*(r-s)), 0.)


# cubic
@np.vectorize
def cubic(r, mu=50.0, s=1.0, rA=1.5):
    """
    Cubic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5


    """
    if not r:
        return 0.
    return np.where(r <= rA, mu*(r-rA)**2*(r-s), 0.)


# general polynomial
@np.vectorize
def general_polynomial(r, muA=40.0, muR=160.0, rA=1.5, rR=1.2, n=1.0, p=1.0):
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
    if not r:
        return 0.
    return np.where(r <= rR, muA*(1-r/rA)**(n+1)-muR*(1-r/rR)**(p+1),
                    np.where(r <= rA, muA*(1-r/rA)**(n+1), 0.))


# logarithmic
@np.vectorize
def logarithmic(r, mu=1.0, s=1.0):
    """
    Logarithmic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if not r:
        return 0.
    return np.where(r <= s, mu*np.log(1+(r-s)), 0.)


# linear-logarithmic
@np.vectorize
def linear_logarithmic(r, mu=1.0, s=1.0):
    """
    Linear logarithmic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if not r:
        return 0.
    return np.where(r <= s, -mu*(r-s)*np.log(1+(r-s)), 0.)


# hard-core model
@np.vectorize
def hard_core(r, mu=1.0, s=1.0, rN=0.3):
    """
    Hard-core model force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0
      rN: radius of nucleus, default 0.3

    """
    if not r:
        return 0.
    return np.where(r <= s-2*rN, np.inf,
                    np.where(r <= s, mu*(r-s)/(r-(s-2*rN)), 0.))


if __name__ == "__main__":

    x_vals = np.linspace(0.8, 2, 200)

    plt.figure()
    plt.plot(x_vals, linear_exponential(x_vals),
             label='linear-exponential (f_max fitted, r_cut small)')
    plt.plot(x_vals, morse(x_vals), label='Morse')
    plt.plot(x_vals, lennard_jones(x_vals), label='LJ')
    plt.plot(x_vals, cubic(x_vals), label='cubic')
    plt.plot(x_vals, general_polynomial(x_vals),
             label='polynomial, n=1 ($\mu_A/\mu_R$ fixed, f_max fitted)')
    plt.plot((1.5, 1.5), (-0.5, 1.5), linestyle='-', color='grey', alpha=0.5)
    plt.text(1.525, -0.35, 'maximum adhesive distance', color='grey')
    plt.plot(1.0, 0.0, linestyle='', marker='o', color='grey')
    plt.text(0.8350, -0.25, 'rest length', color='grey')
    plt.ylim((-2.5, 2.5))
    plt.xlabel('Cell-cell distance in cell diameters')
    plt.ylabel('Force intensity F')
    plt.legend()
