#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:01:31 2021

@author: kubuntu1804
"""
import numpy as _np


def estimate_eigenvalues(y, jacobian, force_args={}, dim=2):

    A = jacobian(y, force_args)
    A_diag = _np.diag(_np.diag(A))

    rho = _np.sum(_np.abs(A), axis=1) - A_diag

    # make complement matrix with all diagonal elements of the submatrix set to zero
    u = A
    for i in range(dim):
        u[i::dim, i::dim] = 0.0

    xi = _np.sum(A - u, axis=1) - A_diag

    minKXiminusRho = _np.minimum(xi-rho)


    return (minKXiminusRho, xi, rho)
