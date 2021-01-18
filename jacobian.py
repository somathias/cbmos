#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:40:33 2020

@author: Sonja Mathias
"""

import numpy as np


def jacobian(y, dim, g, gprime):
#    dim = 3
    y_r = np.expand_dims(y.reshape((-1, dim)), axis=-1)
    n = y_r.shape[0]
    cross_diff = y_r - y_r.transpose([2, 1, 0]) # shape (n, d, n)
    norm = np.sqrt((cross_diff**2).sum(axis=1))
    r_hat = np.expand_dims(np.moveaxis(cross_diff, 1, 2), axis=-1) # shape (n, n, d, 1)

    B = r_hat @ r_hat.transpose([0, 1, 3, 2]) # shape (n, n, d, d)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Ignore divide by 0 warnings
        # All NaNs are removed below

        # add normalization
        B = B / np.expand_dims(norm*norm, axis=(2, 3))

        B = (
                B*np.expand_dims(gprime(norm)-g(norm)/norm, axis=(2, 3))
                + np.expand_dims(np.identity(dim), axis=(0, 1))
                    * np.expand_dims(g(norm)/norm, axis=(2, 3))
                )

        B[np.isnan(B)] = 0

    # Step 2: compute the diagonal
    B[range(n), range(n), :, :] = - B.sum(axis=0)

    # Step 3: Build block matrix
    return B.reshape(n, n, dim, dim).swapaxes(1, 2).reshape(dim*n, -1)


