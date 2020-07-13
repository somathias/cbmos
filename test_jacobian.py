#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:15:10 2020

@author: Sonja Mathias
"""

import pytest
import numpy as np

import force_functions as ff
import jacobian as jac


def test_jacobian_1DN3():

    force = ff.linear
    force_prime = ff.linear_prime

    y = np.array([1.0, 0.7, 2.5])[:, np.newaxis]

    A = jac.jacobian(y, 1, force, force_prime)

    # calculate manually
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    A12 = (y2-y1)**2/abs(y2-y1)**2\
            *(force_prime(abs(y2-y1)) - force(abs(y2-y1))/abs(y2-y1))\
            + force(abs(y2-y1))/abs(y2-y1)
    A13 = (y3-y1)**2/abs(y3-y1)**2\
            *(force_prime(abs(y3-y1)) - force(abs(y3-y1))/abs(y3-y1))\
            + force(abs(y3-y1))/abs(y3-y1)
    A23 = (y3-y2)**2/abs(y3-y2)**2\
            *(force_prime(abs(y3-y2)) - force(abs(y3-y2))/abs(y3-y2))\
            + force(abs(y3-y2))/abs(y3-y2)
    A2 = np.squeeze(np.array([[- (A12 +A13), A12, A13],[A12, -(A12 + A23), A23],[A13, A23, -(A13 +A23)]]))

    assert(np.all(A == A2))

def test_jacobian_2DN3():

    g = ff.linear
    g_prime = ff.linear_prime

    y = np.array([[0., 0.], [0.7, 0.1], [0.3, -1.]])

    A = jac.jacobian(y, 2, g, g_prime)

    # calculate manually
    y1 = y[0, :][:, np.newaxis]
    y2 = y[1, :][:, np.newaxis]
    y3 = y[2, :][:, np.newaxis]

    r12 = y2 - y1
    norm_12 = np.linalg.norm(r12)
    r13 = y3 - y1
    norm_13 = np.linalg.norm(r13)
    r23 = y3 - y2
    norm_23 = np.linalg.norm(r23)

    A12 = r12@r12.transpose()/norm_12**2*(g_prime(norm_12) - g(norm_12)/norm_12) + g(norm_12)/norm_12 * np.eye(2)
    A13 = r13@r13.transpose()/norm_13**2*(g_prime(norm_13) - g(norm_13)/norm_13) + g(norm_13)/norm_13 * np.eye(2)
    A23 = r23@r23.transpose()/norm_23**2*(g_prime(norm_23) - g(norm_23)/norm_23) + g(norm_23)/norm_23 * np.eye(2)

    A2 = np.block([[- (A12 +A13), A12, A13],[A12, -(A12 + A23), A23],[A13, A23, -(A13 +A23)]])

    assert(np.all(A == A2))

def test_jacobian_3DN3():

    g = ff.linear
    g_prime = ff.linear_prime

    y = np.array([[0., 0., 0.], [0.7, 0.1, -0.6], [0.3, -1., -2.0]])

    A = jac.jacobian(y, 3, g, g_prime)

    # calculate manually
    y1 = y[0, :][:, np.newaxis]
    y2 = y[1, :][:, np.newaxis]
    y3 = y[2, :][:, np.newaxis]

    r12 = y2 - y1
    norm_12 = np.linalg.norm(r12)
    r13 = y3 - y1
    norm_13 = np.linalg.norm(r13)
    r23 = y3 - y2
    norm_23 = np.linalg.norm(r23)

    A12 = r12@r12.transpose()/norm_12**2*(g_prime(norm_12) - g(norm_12)/norm_12) + g(norm_12)/norm_12 * np.eye(3)
    A13 = r13@r13.transpose()/norm_13**2*(g_prime(norm_13) - g(norm_13)/norm_13) + g(norm_13)/norm_13 * np.eye(3)
    A23 = r23@r23.transpose()/norm_23**2*(g_prime(norm_23) - g(norm_23)/norm_23) + g(norm_23)/norm_23 * np.eye(3)

    A2 = np.block([[- (A12 +A13), A12, A13],[A12, -(A12 + A23), A23],[A13, A23, -(A13 +A23)]])

    assert(np.all(A == A2))







