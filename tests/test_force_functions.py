#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:18:39 2019

@author: kubuntu1804
"""
import numpy as np

import cbmos.force_functions as ff


def test_hertz():
    assert ff.Hertz()(1.0) == 0.
    assert ff.Hertz()(5/9) ==-8/27

def test_gls():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([0., 0., - np.log(2), 0.25*np.exp(-5.0*0.25)])
    assert (ff.Gls()(x) == y).all()
#    assert ff.gls(1.0) == 0.
#    assert ff.gls(1.5) == 0.
#    assert ff.gls(0.5) == - np.log(2)
#    assert ff.gls(1.25) == 0.25*np.exp(-5.0*0.25)

def test_linearexponential():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([0., 0., - 7.5*np.exp(2.5), 15.0*0.25*np.exp(-5.0*0.25)])
    assert (ff.LinearExponential()(x) == y).all()

def test_linearexponential_derivative():
    x = np.array([1.0, 1.5, 0.5, 1.25])
    y = np.array([15.0, 0., 15.0*3.5*np.exp(2.5), -15.0*0.25*np.exp(-5.0*0.25)])
    assert (ff.LinearExponential().derive()(x) == y).all()