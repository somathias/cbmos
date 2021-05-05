#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as _np


# Linear spring
class Linear:
    def __init__(self):
        pass

    def __call__(self, r, mu=1.0, s=1.0, rA=1.5):
        """
        Linear spring force function.

        Parameters:
          mu: spring stiffness coefficient, default 1.0
          s: rest length, default 1.0

        """
        if r is None:
            return 0.
        return _np.where(r < rA, mu*(r-s), 0.)

    def derive(self):
        def fp(r, mu=1.0, s=1.0, rA=1.5):
            if r is None:
                return 0.
            return _np.where(r < rA, mu, 0.)

        return fp


# Morse
class Morse:
    def __init__(self):
        pass
    def __call__(self, r, m=1.0, a=5.0, s=1.0, rA=1.5):
        """
        Morse potential. (slope at r = s  is 4*a*m)

        Parameters:
          m: maximum value, default 1.0
          a: controls the bredth of the potential, default 1.0
          s: rest length, default 1.0

        """
        if r is None:
            return 0.
        return _np.where(r < rA, - m*(_np.exp(-2*a*(r-s-_np.log(2)/a))-2*_np.exp(-a*(r-s-_np.log(2)/a))), 0.)

    def derive(self):
        def fp(r, m=1.0, a=5.0, s=1.0, rA=1.5):
            if r is None:
                return 0.
            return _np.where(r < rA, - 2*a*m*(_np.exp(-2*a*(r-s-_np.log(2)/a))-_np.exp(-a*(r-s-_np.log(2)/a))), 0.)

        return fp


# Lennard-Jones
class LennardJones:
    def __init__(self):
        pass

    def __call__(self, r, m=1.0, s=1.0, rA=1.5):
        """
        Lennard-Jones potential

        Parameters:
          m: maximum value, default 1.0
          s: rest length, default 1.0

        """
        if r is None:
            return 0.
        return _np.where(r < rA, -4*m*(_np.power(s/r, 12)-_np.power(s/r, 6)), 0.)

    def derive(self):
        def fp(r, m=1.0, s=1.0, rA=1.5):
            if r is None:
                return 0.
            return _np.where(r < rA, -4*m*(-12/r*_np.power(s/r, 12)+6/r*_np.power(s/r, 6)), 0.)

        return fp


# Linear-exponential
class LinearExponential:
    def __init__(self):
        pass

    def __call__(self, r, mu=15.0, s=1.0, a=5.0, rA=1.5):
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
        return _np.where(r < rA, mu*(r-s)*_np.exp(-a*(r-s)), 0.)

    def derive(self):
        def fp(r, mu=1.0, s=1.0, a=5.0, rA=1.5):
            if r is None:
                return 0.
            return _np.where(r < rA, mu*(1-a*(r-s))*np.exp(-a*(r-s)), 0.)

        return fp


# cubic
class Cubic:
    def __init__(self):
        pass

    def __call__(self, r, mu=50.0, s=1.0, rA=1.5):
        """
        Cubic force function

        Parameters:
          mu: spring stiffness coefficient, default 1.0
          s: rest length, default 1.0
          rA: maximum interaction distance (cutoff value), default 1.5


        """
        if r is None:
            return 0.
        return _np.where(r < rA, mu*(r-rA)**2*(r-s), 0.)

    def derive(self):
        def fp(r, mu=50.0, s=1.0, rA=1.5):
            if r is None:
                return 0.
            return _np.where(r < rA, mu*(r-rA)*(2*(r-s)+r-rA), 0.)

        return fp


# general polynomial
class PiecewisePolynomial:
    def __init__(self):
        pass

    def __call__(self, r, muA=40.0, muR=160.0, rA=1.5, rR=1.2, n=1.0, p=1.0):
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
        return _np.where(r <= rR, muA*(1-r/rA)**(n+1)-muR*(1-r/rR)**(p+1),
                        _np.where(r < rA, muA*(1-r/rA)**(n+1), 0.))

    def derive(self):
        def fp(r, muA=40.0, muR=160.0, rA=1.5, rR=1.2, n=1.0, p=1.0):
            if r is None:
                return 0.
            return _np.where(r <= rR, -muA/rA*(n+1)*(1-r/rA)**n+muR/rR*(p+1)*(1-r/rR)**p,
                    _np.where(r < rA, -muA/rA*(n+1)*(1-r/rA)**n, 0.))

        return fp


# logarithmic
class Logarithmic:
    def __init__(self):
        pass

    def __call__(self, r, mu=1.0, s=1.0):
        """
        Logarithmic force function

        Parameters:
          mu: spring stiffness coefficient, default 1.0
          s: rest length, default 1.0

        """
        if r is None:
            return 0.
        r[r==0] = 0.0001  # get away from zero - this is an awful hack!
        return _np.where(r < s, mu*_np.log(1+(r-s)), 0.)

    def derive(self):
        def fp(r, mu=1.0, s=1.0):
            if r is None:
                return 0.
            r[r==0] = 0.0001  # get away from zero - this is an awful hack!
            return _np.where(r < s, mu/(1+(r-s)), 0.)
        return fp

# linear-logarithmic
class LinearLogarithmic:
    def __init__(self):
        pass

    def __call__(self, r, mu=1.0, s=1.0):
        """
        Linear logarithmic force function

        Parameters:
          mu: spring stiffness coefficient, default 1.0
          s: rest length, default 1.0

        """
        if r is None:
            return 0.
        r[r==0] = 0.0001  # get away from zero - this is an awful hack!
        return _np.where(r < s, -mu*(r-s)*_np.log(1+(r-s)), 0.)

    def derive(self):
        def fp(r, mu=1.0, s=1.0):
            if r is None:
                return 0.
            r[r==0] = 0.0001  # get away from zero - this is an awful hack!
            return _np.where(r < s, -mu*_np.log(1+(r-s))-mu*(r-s)/(1+(r-s)), 0.)
        return fp


# hard-core model
class HardCore:
    def __init__(self):
        pass
    def __call__(self, r, mu=1.0, s=1.0, rN=0.3):
        """
        Hard-core model force function

        Parameters:
          mu: spring stiffness coefficient, default 1.0
          s: rest length, default 1.0
          rN: radius of nucleus, default 0.3

        """
        if r is None:
            return 0.
        return _np.where(r <= s-2*rN, _np.inf,
                        _np.where(r < s, mu*(r-s)/(r-(s-2*rN)), 0.))

    def derive(self):
        raise NotImplementedError


class Hertz:
    def __init__(self):
        pass
    def __call__(self, r, mu=1.0, s=1.0):
        """
        (Simplified) Hertz force law for elastic contact.

        Parameters:
          mu: coefficient, default 1.0
          s: rest length, default 1.0

        """
        if r is None:
            return 0.
        return _np.where(r < s, mu*_np.sign(r-s)*(_np.abs(r-s))**(3/2), 0.)

    def derive(self):
        raise NotImplementedError

class Gls:
    def __init__(self):
        pass

    def __call__(self, r, mu=1.0, s=1.0, a=5.0, rA=1.5):
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
        return _np.where(r < s, mu*_np.log(1+(r-s)), _np.where(r < rA, mu*(r-s)*_np.exp(-a*(r-s)), 0))

    def derive(self):
        def fp(r, mu=1.0, s=1.0, a=5.0, rA=1.5):
            if r is None:
                return 0.
            r[r==0] = 0.0001  # get away from zero - this is an awful hack! Plus it does not allow for single value evaluation
            return _np.where(r < s, mu/(1+(r-s)), _np.where(r < rA, mu*(1-a*(r-s))*_np.exp(-a*(r-s)), 0))
        return fp



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.style.use('seaborn')


    x_vals = _np.linspace(0.0, 1.8, 200)

    print(Hertz(x_vals))

    plt.figure()
    plt.plot(x_vals, Hertz(x_vals), label='hertz')
#
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
