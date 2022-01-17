#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import cbmos.force_functions as ff

def test_linear():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([0., 0., - 0.5, 0.25])
    assert (ff.Linear()(x) == y).all()


def test_linear_derivative():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    assert (ff.Linear().derive()(x) == y).all()

def test_cubic():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([0., 0., -25.0, 50.0*(-0.25)**2*0.25])
    assert (ff.Cubic()(x) == y).all()


def test_cubic_derivative():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([50.0*0.25, 0.0, 100.0, -50.0*0.25*0.25])
    assert (ff.Cubic().derive()(x) == y).all()

#
#def test_piecewise_polynomial():
#    x = np.array([1.0, 1.5, 0.5, 1.25])
#    y = np.array([0., 0., -25.0, 50.0*(-0.25)**2*0.25])
#    assert (ff.PiecewisePolynomial()(x) == y).all()
#
#
#def test_piecewise_polynomial_derivative():
#    x = np.array([1.0, 1.5, 0.5, 1.25])
#    y = np.array([50.0*0.25, 0.0, 100.0, -50.0*0.25*0.25])
#    assert (ff.PiecewisePolynomial().derive()(x) == y).all()


def test_hertz():
    assert ff.Hertz()(1.0) == 0.
    assert ff.Hertz()(5/9) == -8/27



def test_gls():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([0., 0., - np.log(2), 0.25*np.exp(-5.0*0.25)])
    assert (ff.Gls()(x) == y).all()



def test_linearexponential():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([0., 0., - 7.5*np.exp(2.5), 15.0*0.25*np.exp(-5.0*0.25)])
    assert (ff.LinearExponential()(x) == y).all()


def test_linearexponential_derivative():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([15.0, 0., 15.0*3.5*np.exp(2.5), -15.0*0.25*np.exp(-5.0*0.25)])
    assert (ff.LinearExponential().derive()(x) == y).all()
