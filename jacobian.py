#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:40:33 2020

@author: Sonja Mathias
"""

import numpy as np


def jacobian(y, dim, g, gprime):
#    dim = 3
    y_r = y.reshape((-1, dim))
    n = y_r.shape[0]
    tmp = np.repeat(y_r[:, :, np.newaxis], n, axis=2)
    norm = np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1))
    r_hat = np.moveaxis(tmp - tmp.transpose(), 1, 2)

    # Step 1
    def rrT(r):
        r = r[:, np.newaxis]
        return r@r.transpose()

    B = np.apply_along_axis(rrT, 2, r_hat)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Ignore divide by 0 warnings
        # All NaNs are removed below

        # add normalization
        B = B / ((norm*norm)[:, :, np.newaxis, np.newaxis]
            .repeat(B.shape[2], axis=2).repeat(B.shape[3], axis=3))

        B = (
                B*(
                (gprime(norm)-g(norm)/norm)[:, :, np.newaxis, np.newaxis]
                .repeat(B.shape[2], axis=2).repeat(B.shape[3], axis=3))
            + (np.identity(dim)[np.newaxis, np.newaxis, :, :]
                .repeat(B.shape[0], axis=0).repeat(B.shape[1], axis=1))*
                (g(norm)/norm)[:, :, np.newaxis, np.newaxis]
                .repeat(B.shape[2], axis=2).repeat(B.shape[3], axis=3))

        B[np.isnan(B)] = 0

    # Step 2: compute the diagonal
    B[range(n), range(n), :, :] = - B.sum(axis=0)

    # Step 3: Build block matrix
    return B.reshape(n, n, dim, dim).swapaxes(1, 2).reshape(dim*n, -1)


