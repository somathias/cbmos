#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import cbmos.solvers.geshgorin as gs

@np.vectorize
def func(t, y):
    return -50*y

def jacobian(y, fa):
    return -50*np.eye(len(y))

def test_dN6():
    # doesn't matter as A is independent of y for test equation
    y = np.array([0.3, 0.6, 4.5, 0.25, 1.6, 2.3])

    (m, xi, rho) = gs.estimate_eigenvalues(y, jacobian)

    assert(np.all(rho==0.0))
    assert(np.all(xi==-50.0))


