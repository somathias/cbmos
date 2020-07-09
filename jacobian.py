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
    r_hat = np.rollaxis(tmp-tmp.transpose(), 2, 0)

    # Step 1
    def rrT(r):
        r = r[:, np.newaxis]
        return r@r.transpose()

    B = np.apply_along_axis(rrT, 2, r_hat)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Ignore divide by 0 warnings
        # All NaNs are removed below
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


def jacobian_1D(y, force, force_prime):
    abs_diff = np.sqrt((y - y.transpose())**2)
    A = (y - y.transpose())/(abs_diff + np.eye(len(y)))\
        * (y - y.transpose())/(abs_diff + np.eye(len(y)))\
        * (force_prime(abs_diff) - force(abs_diff)/(abs_diff + np.eye(len(y))))\
        + force(abs_diff)/(abs_diff + np.eye(len(y)))
    dia = np.diag_indices(len(y))
    A[dia] = A[dia] - A.sum(axis=1)
    return A


def test1DN3(y1, y2, y3, force, force_prime):
    A12 = (y2-y1)**2/abs(y2-y1)**2\
            *(force_prime(abs(y2-y1)) - force(abs(y2-y1))/abs(y2-y1))\
            + force(abs(y2-y1))/abs(y2-y1)
    A13 = (y3-y1)**2/abs(y3-y1)**2\
            *(force_prime(abs(y3-y1)) - force(abs(y3-y1))/abs(y3-y1))\
            + force(abs(y3-y1))/abs(y3-y1)
    A23 = (y3-y2)**2/abs(y3-y2)**2\
            *(force_prime(abs(y3-y2)) - force(abs(y3-y2))/abs(y3-y2))\
            + force(abs(y3-y2))/abs(y3-y2)
    A = np.array([[- (A12 +A13), A12, A13],[A12, -(A12 + A23), A23],[A13, A23, -(A13 +A23)]])
    return np.squeeze(A)

if __name__ == "__main__":

    import force_functions as ff

    force = ff.linear
    force_prime = ff.linear_prime
    y = np.array([1.0, 0.7, 2.5])[:, np.newaxis]

    jac = jacobian_1D(y, force, force_prime)
    test = test1DN3(y[0], y[1], y[2], force, force_prime)

    print(np.all(jac == test))

    jac2 = jacobian(y, 1, force, force_prime)
    print(np.all(jac2 == test))

    y = np.array([[0, 0, 0], [1, 2, 3], [8, -1, 5], [10, 11, 12]])
    jac3 = jacobian(y, 3, force, force_prime)

