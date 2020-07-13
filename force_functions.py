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
def linear(r, mu=1.0, s=1.0, rA=1.5):
    """
    Linear spring force function.

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    return np.where(r < rA, mu*(r-s), 0.)

# Linear spring - derivative
def linear_prime(r, mu=1.0, s=1.0, rA=1.5):
    """
    Derivative of the linear spring force function.

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    return np.where(r < rA, mu, 0.)


# Morse
def morse(r, m=1.0, a=5.0, s=1.0, rA=1.5):
    """
    Morse potential. (slope at r = s  is 4*a*m)

    Parameters:
      m: maximum value, default 1.0
      a: controls the bredth of the potential, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    return np.where(r < rA, - m*(np.exp(-2*a*(r-s-np.log(2)/a))-2*np.exp(-a*(r-s-np.log(2)/a))), 0.)

# Morse - derivative
def morse_prime(r, m=1.0, a=5.0, s=1.0, rA=1.5):
    """
    Derivative of the Morse potential.

    Parameters:
      m: maximum value, default 1.0
      a: controls the bredth of the potential, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    return np.where(r < rA, - 2*a*m*(np.exp(-2*a*(r-s-np.log(2)/a))-np.exp(-a*(r-s-np.log(2)/a))), 0.)


# Lennard-Jones
def lennard_jones(r, m=1.0, s=1.0, rA=1.5):
    """
    Lennard-Jones potential

    Parameters:
      m: maximum value, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    return np.where(r < rA, -4*m*(np.power(s/r, 12)-np.power(s/r, 6)), 0.)

# Lennard-Jones - derivative
def lennard_jones_prime(r, m=1.0, s=1.0, rA=1.5):
    """
    Derivative of the Lennard-Jones potential

    Parameters:
      m: maximum value, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    return np.where(r < rA, -4*m*(-12/r*np.power(s/r, 12)+6/r*np.power(s/r, 6)), 0.)


# Linear-exponential
def linear_exponential(r, mu=15.0, s=1.0, a=5.0, rA=1.5):
    """
    Linear exponential force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0
      a: controls the bredth of the potential, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5


    """
    if r is None:
        return 0.
    return np.where(r < rA, mu*(r-s)*np.exp(-a*(r-s)), 0.)

# Linear-exponential -derivative
def linear_exponential_prime(r, mu=15.0, s=1.0, a=5.0, rA=1.5):
    """
    Derivative of the linear exponential force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0
      a: controls the bredth of the potential, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5


    """
    if r is None:
        return 0.
    return np.where(r < rA, mu*(1-a*(r-s))*np.exp(-a*(r-s)), 0.)


# cubic
def cubic(r, mu=50.0, s=1.0, rA=1.5):
    """
    Cubic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5


    """
    if r is None:
        return 0.
    return np.where(r < rA, mu*(r-rA)**2*(r-s), 0.)

# cubic - derivative
def cubic_prime(r, mu=50.0, s=1.0, rA=1.5):
    """
    Derivative of the cubic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5


    """
    if r is None:
        return 0.
    return np.where(r < rA, mu*(r-rA)*(2*(r-s)+r-rA), 0.)


# general polynomial
def piecewise_polynomial(r, muA=40.0, muR=160.0, rA=1.5, rR=1.2, n=1.0, p=1.0):
    """
    Piecewise polynomial force function

    Parameters:
      muA: spring stiffness coefficient for adhesion, default 1.0
      muR: spring stiffness coefficient for repulsion, default 1.0
      rA: maximum adhesive interaction distance (cutoff value), default 1.5
      rR: maximum repulsive interaction distance (cutoff value), default 1.5
      n: exponent adhesive part
      m: exponent repulsive part

    """
    if r is None:
        return 0.
    return np.where(r <= rR, muA*(1-r/rA)**(n+1)-muR*(1-r/rR)**(p+1),
                    np.where(r < rA, muA*(1-r/rA)**(n+1), 0.))

# piecewise polynomial - derivative
def piecewise_polynomial_prime(r, muA=40.0, muR=160.0, rA=1.5, rR=1.2, n=1.0, p=1.0):
    """
    Derivative of the piecewise polynomial force function

    Parameters:
      muA: spring stiffness coefficient for adhesion, default 1.0
      muR: spring stiffness coefficient for repulsion, default 1.0
      rA: maximum adhesive interaction distance (cutoff value), default 1.5
      rR: maximum repulsive interaction distance (cutoff value), default 1.5
      n: exponent adhesive part
      m: exponent repulsive part

    """
    if r is None:
        return 0.
    return np.where(r <= rR, -muA/rA*(n+1)*(1-r/rA)**n+muR/rR*(p+1)*(1-r/rR)**p,
                    np.where(r < rA, -muA/rA*(n+1)*(1-r/rA)**n, 0.))

# logarithmic
def logarithmic(r, mu=1.0, s=1.0):
    """
    Logarithmic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    r[r==0] = 0.0001  # get away from zero - this is an awful hack!
    return np.where(r < s, mu*np.log(1+(r-s)), 0.)

# logarithmic - derivative
def logarithmic_prime(r, mu=1.0, s=1.0):
    """
    Derivative of the logarithmic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    r[r==0] = 0.0001  # get away from zero - this is an awful hack!
    return np.where(r < s, mu/(1+(r-s)), 0.)

# linear-logarithmic
def linear_logarithmic(r, mu=1.0, s=1.0):
    """
    Linear logarithmic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
    r[r==0] = 0.0001  # get away from zero - this is an awful hack!
    return np.where(r < s, -mu*(r-s)*np.log(1+(r-s)), 0.)

# linear-logarithmic -derivative
def linear_logarithmic_prime(r, mu=1.0, s=1.0):
    """
    Derviative of the linear logarithmic force function

    Parameters:
      mu: spring stiffness coefficient, default 1.0
      s: rest length, default 1.0

    """
    if r is None:
        return 0.
        r[r==0] = 0.0001  # get away from zero - this is an awful hack!
    return np.where(r < s, -mu*np.log(1+(r-s))-mu*(r-s)/(1+(r-s)), 0.)


## hard-core model
#def hard_core(r, mu=1.0, s=1.0, rN=0.3):
#    """
#    Hard-core model force function
#
#    Parameters:
#      mu: spring stiffness coefficient, default 1.0
#      s: rest length, default 1.0
#      rN: radius of nucleus, default 0.3
#
#    """
#    if r is None:
#        return 0.
#    return np.where(r <= s-2*rN, np.inf,
#                    np.where(r < s, mu*(r-s)/(r-(s-2*rN)), 0.))
#
#
#def hertz(r, mu=1.0, s=1.0):
#    """
#    (Simplified) Hertz force law for elastic contact.
#
#    Parameters:
#      mu: coefficient, default 1.0
#      s: rest length, default 1.0
#
#    """
#    if r is None:
#        return 0.
#    return np.where(r < s, mu*np.sign(r-s)*(np.abs(r-s))**(3/2), 0.)

def gls(r, mu=1.0, s=1.0, a=5.0, rA=1.5):
    """
    Generalized linear spring using logarithmic for repulsion and linear-
    exponential for adhesion.

    Parameters:
      mu: coefficient, default 1.0
      s: rest length, default 1.0
      a: controls the bredth of the potential, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5

    """
    if r is None:
        return 0.
    r[r==0] = 0.0001  # get away from zero - this is an awful hack! Plus it does not allow for single value evaluation
    return np.where(r < s, mu*np.log(1+(r-s)), np.where(r < rA, mu*(r-s)*np.exp(-a*(r-s)), 0))

def gls_prime(r, mu=1.0, s=1.0, a=5.0, rA=1.5):
    """
    Generalized linear spring using logarithmic for repulsion and linear-
    exponential for adhesion.

    Parameters:
      mu: coefficient, default 1.0
      s: rest length, default 1.0
      a: controls the bredth of the potential, default 1.0
      rA: maximum interaction distance (cutoff value), default 1.5

    """
    if r is None:
        return 0.
    r[r==0] = 0.0001  # get away from zero - this is an awful hack! Plus it does not allow for single value evaluation
    return np.where(r < s, mu/(1+(r-s)), np.where(r < rA, mu*(1-a*(r-s))*np.exp(-a*(r-s)), 0))


if __name__ == "__main__":



    x_vals = np.linspace(0.0, 1.8, 200)

#    plt.figure()
#    plt.plot(x_vals, linear(x_vals),
#             label='linear')
#    plt.plot(x_vals, linear_exponential(x_vals),
#             label='linear-exponential (f_max fitted, r_cut small)')
#    plt.plot(x_vals, morse(x_vals), label='Morse')
#    plt.plot(x_vals, lennard_jones(x_vals), label='LJ')
#    plt.plot(x_vals, cubic(x_vals), label='cubic')
#    plt.plot(x_vals, piecewise_polynomial(x_vals),
#             label='polynomial, n=1 ($\mu_A/\mu_R$ fixed, f_max fitted)')
#    plt.plot((1.5, 1.5), (-0.5, 1.5), linestyle='-', color='grey', alpha=0.5)
#    plt.text(1.525, -0.35, 'maximum adhesive distance', color='grey')
#    plt.plot(1.0, 0.0, linestyle='', marker='o', color='grey')
#    plt.text(0.8350, -0.25, 'rest length', color='grey')
#    plt.ylim((-2.5, 2.5))
#    plt.xlabel('Cell-cell distance in cell diameters')
#    plt.ylabel('Force intensity F')
#    plt.legend()
