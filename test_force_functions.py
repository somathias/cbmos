#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:43:56 2018

@author: Sonja Mathias
"""

import numpy as np
import matplotlib.pyplot as plt
import force_functions as ff

plt.style.use('seaborn')



s = 1.0 # equilibrium rest length, set to 1 cell diameter 
rA = s+0.5 # set maximum interaction distance to 1.5 cell diameter
rR = s+0.2 # set maximum repulsive interaction distance to 1.2 cell diameter
rN = 0.3 # radius of nuclei for hard-core model


linear_spring = lambda mu, r: ff.linear_spring(r, {"mu":mu, "s":s})

morse = lambda m, a, r: ff.morse(r, {"m":m, "a":a,  "s":s})

lennard_jones = lambda m, r: ff.lennard_jones(r, {"m":m,  "s":s})

linear_exponential_product = lambda mu, a, r: ff.linear_exponential(r,  {"mu":mu, "a":a,  "s":s, "rA":rA})

cubic_law = lambda mu, r: ff.cubic(r, {"mu":mu, "s":s, "rA":rA})

logarithmic = lambda mu, r: ff.logarithmic(r, {"mu":mu,  "s":s})

linear_logarithmic_product = lambda mu, r: ff.linear_logarithmic(r, {"mu":mu,  "s":s })

hard_core = lambda mu, r: ff.hard_core(r, {"mu":mu, "s":s, "rN":rN})

polynomial_repulsion = lambda mu, m, r : np.where(r<=rR, -mu*(1-r/rR)**(m+1), 0.)

polynomial_adhesion = lambda mu, n, r : np.where(r<=rA, mu*(1-r/rA)**(n+1), 0.)

polynomial = lambda muA, muR, n, p, r: ff.general_polynomial(r, {"muA":muA, "muR":muR, "rA":rA, "rR":rR, "n":n, "p":p})


## Parameters
m = 2.0 # maximum value for Morse and LJ
aMorse = 8.0 # Morse parameter, controls the location of the maximum
muGLSA = 120.0
#muGLSR = 240.0 
aGLS = -2*np.log(0.002/muGLSA)
muCUBER = 79*1.3697
muCUBEA = 79*1.3697
muRATR = 168.0
muRATA = 0.25*muRATR


plt.figure()
x_vals = np.linspace(0.8,2,200)
plt.plot(x_vals, linear_exponential_product(muGLSA, aGLS, x_vals), label='linear-exponential (f_max fitted, r_cut small)')
#plt.plot(x_vals, morse(m,18, x_vals), label='Morse')
plt.plot(x_vals, lennard_jones(m,x_vals), label='LJ')
plt.plot(x_vals, cubic_law(muCUBER, x_vals), label='cubic')
plt.plot(x_vals, polynomial_adhesion(muRATA, 1, x_vals)+ polynomial_repulsion(muRATR, 1, x_vals), label='polynomial, n=1 ($\mu_A/\mu_R$ fixed, f_max fitted)')
plt.plot((1.5, 1.5), (-0.5, 1.5), linestyle='-', color='grey', alpha=0.5)
plt.text(1.525,-0.35,'maximum adhesive distance', color='grey')
plt.plot(1.0, 0.0, linestyle='', marker='o', color='grey')
plt.text(0.8350,-0.25,'rest length', color='grey')

plt.ylim((-2.5,2.5))
plt.xlabel('Cell-cell distance in cell diameters')
plt.ylabel('Force intensity F')
plt.legend()
#plt.savefig('comparison_GLS_LJ_cubic_poly.png')
#plt.savefig('comparison_GLS_LJ_cubic_poly.pdf')

plt.figure()
muRATR = 168.0
muRATA = 0.25*muRATR
x_vals = np.linspace(0.8,2,200)
plt.plot(x_vals, lennard_jones(m,x_vals), label='LJ', linestyle='--')
plt.plot(x_vals, morse(m,4.0, x_vals), label='Morse (fitted)', linestyle='-.')
plt.plot(x_vals, linear_exponential_product(32.5, 6, x_vals), label='linear-exponential (fitted)')
plt.plot(x_vals, cubic_law(muCUBER, x_vals), label='cubic')
plt.plot(x_vals, polynomial_adhesion(muRATA, 1, x_vals)+ polynomial_repulsion(muRATR, 1, x_vals), label='polynomial, n=1 ($\mu_A/\mu_R$ fixed, f_max fitted)')
plt.plot((1.5, 1.5), (-0.5, 1.5), linestyle='-', color='grey', alpha=0.5)
plt.text(1.525,-0.35,'maximum adhesive distance', color='grey')
plt.plot(1.0, 0.0, linestyle='', marker='o', color='grey')
plt.text(0.8350,-0.45,'rest length', color='grey')
plt.ylim((-2.5,2.5))
plt.xlabel('Cell-cell distance in cell diameters')
plt.ylabel('Force intensity F')
plt.legend()
#plt.savefig('comparison_fitted_to_cubic.png')
#plt.savefig('comparison_fitted_to_cubic.pdf')
#
plt.figure()
x_vals = np.linspace(0.8,2,200)
plt.plot(x_vals, polynomial_adhesion(muRATA, 1, x_vals), label='polynomial adhesion only, n=1')
plt.plot(x_vals, polynomial_repulsion(muRATR, 1, x_vals), label='polynomial repulsion only, n=1')
plt.plot(x_vals, polynomial_adhesion(muRATA, 1, x_vals)+ polynomial_repulsion(muRATR, 1, x_vals), label='polynomial, n=1 ($\mu_A/\mu_R$ fixed, f_max fitted)')
plt.xlabel('Cell-cell distance in cell diameters')
plt.ylabel('Force intensity F')
plt.plot((1.5, 1.5), (-0.5, 1.5), linestyle='-', color='grey', alpha=0.5)
plt.text(1.525,-0.535,'maximum adhesive \ndistance', color='grey')
plt.plot(1.0, 0.0, linestyle='', marker='o', color='grey')
plt.text(0.87350,-0.55,'rest length', color='grey')
plt.legend()
plt.ylim((-2,3))
#plt.savefig('polynomial_force_law.png')
#plt.savefig('polynomial_force_law.pdf')
#
#
plt.figure()
x_vals = np.linspace(0.5,1.2,200)
#plt.plot(x_vals, gls(muGLSA, aGLS, x_vals), label='GLS (f_max fitted, r_cut small)')
plt.plot(x_vals, linear_exponential_product(muGLSA, aGLS, x_vals), label='linear-exponential (f_max fitted, r_cut small)')
plt.plot(x_vals, logarithmic(muGLSA, x_vals), label='logarithmic')
plt.plot(x_vals, cubic_law(muCUBER,  x_vals), label='cubic')
plt.plot(x_vals, polynomial_adhesion(muRATA, 1, x_vals)+polynomial_repulsion(muRATR, 1, x_vals), label='polynomial, n=1 ($\mu_A/\mu_R$ fixed, f_max fitted)')
plt.plot(x_vals, linear_logarithmic_product(muGLSA, x_vals), label='linear-logarithmic')
plt.plot(x_vals, hard_core(1.0, x_vals), label='hard core model')
#plt.plot(x_vals, hard_core2(1.0, x_vals), label='hard core model 2')
plt.plot(1.0, 0.0, linestyle='', marker='o', color='grey')
plt.text(1.020,-1.1,'rest length', color='grey')
plt.xlabel('Cell-cell distance in cell diameters')
plt.ylabel('Force intensity F')
plt.legend()
plt.ylim((-10,2.5))
#plt.savefig('comparison_repulsion.png')
#plt.savefig('comparison_repulsion.pdf')