#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import cbmos.solvers.heun as he


@np.vectorize
def func(t, y):
    return -50*y


def jacobian(y, fa):
    return -50*np.eye(len(y))


def test_no_overstep():
    t_span = (0, 1)
    y0 = np.array([1, 1])

    # fixed time step
    sol = he.solve_ivp(func, t_span, y0, dt=0.03)
    assert sol.t[-1] == t_span[1]
