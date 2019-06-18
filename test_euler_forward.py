#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:45:08 2019

@author: Sonja Mathias
"""
import numpy as np
import euler_forward as ef


@np.vectorize
def func(t, y):
    return -50*y


def test_y_shape():

    t_span = (0, 1)
    y0 = np.array([1, 1])

    sol = ef.solve_ivp(func, t_span, y0)

    assert len(sol.t) == 100
    assert sol.y.shape == (2, 100)
