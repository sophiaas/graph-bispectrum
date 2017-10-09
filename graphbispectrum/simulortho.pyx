# encoding: utf-8
# cython: profile=False
from __future__ import division

import collections
import itertools
import numpy as np
import random
import time

from util import memoize

cimport cython
cimport numpy as np

from libc.math cimport cos, sin

DTYPE = np.float
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
cdef void _equation(np.ndarray[dtype=DTYPE_t, ndim=2] A, np.ndarray[dtype=DTYPE_t, ndim=2] B, np.ndarray[dtype=DTYPE_t, ndim=1] result):

    cdef np.ndarray[dtype=DTYPE_t, ndim=2] M = B[:2, 2:].dot(A[:2, 2:].T)
    cdef np.ndarray[dtype=DTYPE_t, ndim=2] N = B[2:, :2].T.dot(A[2:, :2])

    cdef DTYPE_t y = (
        A[1, 0] * B[1, 1] -
        A[1, 1] * B[0, 1] -
        A[1, 1] * B[1, 0] +
        A[0, 1] * B[1, 1] -
        A[0, 1] * B[0, 0] +
        A[0, 0] * B[0, 1] -
        A[1, 0] * B[0, 0] +
        A[0, 0] * B[1, 0]
    )

    cdef DTYPE_t x = (
        N[1, 0] + M[1, 0] -
        N[0, 1] - M[0, 1]
    )

    cdef DTYPE_t w = -(
        N[1, 1] + M[1, 1] +
        N[0, 0] + M[0, 0]
    )

    cdef DTYPE_t v = 2 * (
        A[1, 1] * B[0, 0] +
        A[0, 0] * B[1, 1] -
        A[1, 0] * B[1, 0] -
        A[0, 0] * B[0, 0] -
        A[1, 1] * B[1, 1] -
        A[0, 1] * B[0, 1] -
        A[0, 1] * B[1, 0] -
        A[1, 0] * B[0, 1]
    )

    cdef DTYPE_t u = 2 * (
        A[1, 0] * B[0, 0] -
        A[0, 0] * B[0, 1] +
        A[0, 1] * B[0, 0] -
        A[0, 0] * B[1, 0] +
        A[1, 1] * B[0, 1] -
        A[1, 0] * B[1, 1] +
        A[1, 1] * B[1, 0] -
        A[0, 1] * B[1, 1]
    )

    result[0] = y
    result[1] = x
    result[2] = w
    result[3] = v
    result[4] = u


@cython.boundscheck(False)
cdef np.ndarray[dtype=DTYPE_t, ndim=2] givens_rotation(np.ndarray[dtype=DTYPE_t, ndim=2] R, int i, int j, DTYPE_t theta):
    for k in xrange(R.shape[0]):
        for l in xrange(R.shape[1]):
            R[k, l] = int(k == l)

    R[i, i] = cos(theta)
    R[i, j] = sin(theta)
    R[j, i] = -sin(theta)
    R[j, j] = cos(theta)
    return R


@cython.boundscheck(False)
cdef make_permutation(int i, int j, int n):
    cdef np.ndarray[dtype=DTYPE_t, ndim=2] Pij = np.identity(n)
    Pij[0, 0] = 0
    Pij[1, 1] = 0
    Pij[i, i] = 0
    Pij[j, j] = 0
    Pij[0, i] = 1
    Pij[1, j] = 1
    Pij[i, 0] = 1
    Pij[j, 1] = 1
    return Pij


@cython.boundscheck(False)
def error(np.ndarray[dtype=DTYPE_t, ndim=2] A1, np.ndarray[dtype=DTYPE_t, ndim=2] B1, np.ndarray[dtype=DTYPE_t, ndim=2] A2, np.ndarray[dtype=DTYPE_t, ndim=2] B2, np.ndarray[dtype=DTYPE_t, ndim=2] R):
    cdef DTYPE_t objective = 0
    cdef int n = A1.shape[0]
    cdef np.ndarray[dtype=DTYPE_t, ndim=2] T1 = R.dot(B1).dot(R.T) - A1
    cdef np.ndarray[dtype=DTYPE_t, ndim=2] T2 = R.dot(B2).dot(R.T) - A2

    for i in xrange(n):
        for j in xrange(n):
            objective += T1[i, j] ** 2 + T2[i, j] ** 2

    return objective


@cython.boundscheck(False)
def simultaneously_orthogonalize(
        np.ndarray A1, np.ndarray A2, np.ndarray B1, np.ndarray B2,
        DTYPE_t atol=1.0e-6, int max_iter=1000):

    """ See http://perso.telecom-paristech.fr/~cardo4o/Algo/Joint_Diag/joint_diag.m.

        http://perso.telecom-paristech.fr/~cardoso/jfbib.html

        http://perso.telecom-paristech.fr/~cardoso/Algo/Jade/jade.py
    """
    cdef int n = A1.shape[0]
    print "N=", n
    cdef np.ndarray C = np.identity(n)
    cdef np.ndarray R = np.identity(n)
    cdef np.ndarray I = np.identity(n)
    cdef np.ndarray OB1 = B1
    cdef np.ndarray OB2 = B2

    timers = collections.defaultdict(lambda: 0)

    # def eval_trig_poly(coeffs, theta):
    #     y, x, w, v, u = coeffs
    #     s = np.sin(theta)
    #     c = np.cos(theta)
    #     return u * s ** 2 + v * s * c + w * s + x * c + y

    # def eval_trig_poly2(coeffs, theta):
    #     y, x, w, v, u = coeffs
    #     s = np.sin(theta)
    #     c = np.cos(theta)
    #     return (
    #         -x ** 2 + y ** 2 +
    #         (2 * y * w - 2 * x * v) * s +
    #         (2 * u * y - v ** 2 + w ** 2 + x ** 2) * s ** 2 +
    #         (2 * u * w + 2 * x * v) * s ** 3 +
    #         (u ** 2 + v ** 2) * s ** 4
    #     )

    # def ghetto_derivative(f, theta):
    #     return (f(theta + 0.0000000001) - f(theta)) / 0.0000000001

    # def plot(coeffs, i=0, j=1):
    #     import matplotlib.pyplot as plt
    #     thetas = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)
    #     vals1 = -2 * np.array([eval_trig_poly(coeffs, t) for t in thetas])
    #     f = lambda t: error(givens_rotation(i, j, t))
    #     vals2 = np.array([ghetto_derivative(f, t) for t in thetas])
    #     print thetas.shape, vals1.shape
    #     plt.plot(thetas, vals1)
    #     plt.plot(thetas, vals2)
    #     plt.show()
    #     import time
    #     time.sleep(100000)

    # i, j = 2, 4
    # Pij = make_permutation(i, j)
    # A1p = Pij.T.dot(A1).dot(Pij)
    # B1p = Pij.T.dot(B1).dot(Pij)

    # A2p = Pij.T.dot(A2).dot(Pij)
    # B2p = Pij.T.dot(B2).dot(Pij)
    # coeffs = tuple(
    #     _equation(A1p, B1p) + _equation(A2p, B2p)
    # )
    # plot(coeffs, i, j)

    cdef DTYPE_t theta = 0
    cdef DTYPE_t last_error = 0
    cdef DTYPE_t new_error = 0
    cdef int itr = 0
    # max_itr = 10
    count_theta = 0
    ijs = list(itertools.product(xrange(n), xrange(n)))
    # random.shuffle(ijs)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] result1 = np.zeros(5)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] result2 = np.zeros(5)

    while itr < max_iter and (not new_error or new_error > atol):
        if last_error and (last_error / new_error) < 1.005:
            print "** Bump **"
            B1 = OB1
            B2 = OB2
            C = np.identity(n)
            random.shuffle(ijs)

        last_error = error(A1, B1, A2, B2, I)
        print itr, last_error
        itr += 1

        for i, j in ijs:
            if i == j:
                continue

            # Apply transformation to A and B
            t1 = time.time()
            Pij = make_permutation(i, j, n)
            A1p = Pij.T.dot(A1).dot(Pij)
            B1p = Pij.T.dot(B1).dot(Pij)

            A2p = Pij.T.dot(A2).dot(Pij)
            B2p = Pij.T.dot(B2).dot(Pij)
            timers["prepare_matrices"] += time.time() - t1

            t1 = time.time()
            _equation(A1p, B1p, result1)
            _equation(A2p, B2p, result2)
            y, x, w, v, u = result1 + result2
            timers["_equation"] += time.time() - t1

            t1 = time.time()
            poly = (
                -x ** 2 + y ** 2,
                2 * y * w - 2 * x * v,
                2 * u * y - v ** 2 + w ** 2 + x ** 2,
                2 * u * w + 2 * x * v,
                u ** 2 + v ** 2
            )
            timers["make_poly"] += time.time() - t1

            t1 = time.time()
            PC = np.polynomial.polynomial.polycompanion(poly)
            timers["make_companion"] += time.time() - t1

            t1 = time.time()
            d, V = np.linalg.eig(PC)
            timers["eig"] += time.time() - t1
            t1 = time.time()
            d = list(np.arcsin(filter(
                lambda c: np.isreal(c) and -1 <= c <= 1, d)))
            timers["arcsin"] += time.time() - t1

            t1 = time.time()
            errors = map(lambda t: error(A1, B1, A2, B2, givens_rotation(R, i, j, t)), d)
            min_theta_idx = np.argmin(errors)
            min_error = errors[min_theta_idx]
            theta = d[min_theta_idx] if min_error <= new_error else 0
            if theta == 0:
                count_theta += 1
            timers["argmin"] += time.time() - t1

            # print "*" * 80
            # print eval_trig_poly(coeffs, theta)
            # print eval_trig_poly2(coeffs, theta)

            t1 = time.time()
            R = givens_rotation(R, i, j, theta)
            C = R.dot(C)
            B1 = R.dot(B1).dot(R.T)
            B2 = R.dot(B2).dot(R.T)
            timers["apply"] += time.time() - t1
            new_error = error(A1, B1, A2, B2, I)
        print count_theta
        count_theta = 0

    print error(A1, B1, A2, B2, I)
    print timers
    return C
