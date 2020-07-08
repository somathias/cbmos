#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:40:33 2020

@author: Sonja Mathias
"""

import numpy as np


def A_1D(y, force, force_prime):
    abs_diff = np.sqrt((y - y.transpose())**2)
    A = (y - y.transpose())/(abs_diff + np.eye(len(y)))\
        * (y - y.transpose())/(abs_diff + np.eye(len(y)))\
        * (force_prime(abs_diff) - force(abs_diff)/(abs_diff + np.eye(len(y))))\
        + force(abs_diff)/(abs_diff + np.eye(len(y)))
    dia = np.diag_indices(len(y))
    A[dia] = A[dia] - A.sum(axis=1)
    return A

#def A(y, dim, force, force_prime):
#    y_r = y.reshape((-1, dim))
#    tmp = np.repeat(y_r[:, :, np.newaxis], y_r.shape[0], axis=2)
#    norm = np.sqrt(((tmp - tmp.transpose())**2).sum(axis=1))
#    r_hat = (tmp - tmp.transpose())/(norm + np.eye(len(y)))
#
#    A = r_hat@r_hat.transpose()




def A_test1DN3(y1, y2, y3, force, force_prime):
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

    jacobian = A_1D(y, force, force_prime)
    test = A_test1DN3(y[0], y[1], y[2], force, force_prime)

    print(np.all(jacobian == test))