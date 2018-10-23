from __future__ import division

import collections
import itertools
import numpy as np
import random
import time

from method_decorator import method_decorator
from numpy.linalg import svd, eigh
from scipy import sparse
from scipy.linalg import qr
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse.linalg import eigsh, svds
from math import atan2

MemoKey = collections.namedtuple("MemoKey", ["args", "kwargs"])


class memodict(dict):
    def __init__(self, f):
        super(memodict, self).__init__()
        self.f = f

    def __missing__(self, key):
        ret = self[key] = self.f(*key.args, **dict(key.kwargs))
        return ret


def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    return memodict(f).__getitem__


class memoize_method(method_decorator):

    def __call__(self, *args, **kwargs):
        obj = self.obj
        key = MemoKey(args=args, kwargs=tuple(kwargs.items()))
        if not hasattr(obj, "_cache"):
            obj._cache = {}
        name = self.func.__name__
        if name not in obj._cache:
            obj._cache[name] = memodict(self.func)
        return obj._cache[name][key]


def direct_sum(Ms):
    dim = sum((M.shape[0] for M in Ms))
    result = np.zeros((dim, dim))
    o = 0
    for M in Ms:
        d = M.shape[0]
        result[o:o + d, o:o + d] = M
        o += d
    return result


def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        result.append(np.array(s))
    return result


def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-8, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def nullspace_qr(A):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the QR
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    Q, R = qr(A.T)
    ns = Q[:, R.shape[1]:].conj()
    return ns


def sparse_nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the QR
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """
    s, v = eigsh(A.T.dot(A), k=3, ncv=8, which="SM")
    cols = []
    for i in xrange(v.shape[1]):
        col = v[:, i]
        if np.linalg.norm(A.dot(col)) < atol:
            col = col.T
            col.resize((col.shape[0], 1))
            cols.append(sparse.lil_matrix(col))
    return cols


def simultaneously_diagonalize(A, atol=1.0e-12, max_iter=50, ensure_ordered_diag=True, unitary=False):
    """ See http://perso.telecom-paristech.fr/~cardo4o/Algo/Joint_Diag/joint_diag.m.

        http://perso.telecom-paristech.fr/~cardoso/jfbib.html

        http://perso.telecom-paristech.fr/~cardoso/Algo/Jade/jade.py

        unitary = False:  do real version of algorithm
        unitary = True:  do complex version

    """
    m, nm = A.shape
    n = int(nm/m)
    if unitary == True:
        V = np.identity(m, dtype=np.complex128)
        B = np.array([[1, 0, 0], [0, 1, 1], [0, (0.-1.0j), (0.+1.0j)]], dtype=np.complex128)    # for complex version
    else:
        V = np.identity(m)  # dtype=np.float128)

    encore = True
    sweep = 0  # number of passes thru matrix applying Givens transforms
    updates = 0  # Total number of rotations
    upds = 0  # Number of rotations in a given seep

    # Joint diagonalization
    while encore:
        encore = False
        sweep += 1
        upds  = 0
        for p in range(m-1):
            for q in range(p+1, m):
                Ip = np.arange(p, m*n, m)
                Iq = np.arange(q, m*n, m)

                # computation of Givens angle
                if unitary == False:  # real Givens rotations applied
                    g = np.vstack([A[p,Ip] - A[q,Iq], A[p,Iq] + A[q,Ip]])
                    gg = np.dot(g, g.T)
                    ton = gg[0,0] - gg[1,1]
                    toff = gg[0,1] + gg[1,0]
                    theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                    c = np.cos(theta)
                    s = np.sin(theta)
                else:
                    g = np.vstack([A[p,Ip] - A[q,Iq], A[p,Iq], A[q,Ip]])
                    D, vcp = np.linalg.eig(np.real(B.dot(g.dot(g.conj().T)).dot(B.conj().T)))
                    K = np.argsort(D)
                    angles  = vcp[:,K[2]]
                    if angles[0] < 0:
                        angles = -angles
                    c = np.sum(np.sqrt(0.5 + angles[0] / 2.))  # weird: have to add np.sum since sometimes angles is weird format
                    s = np.sum(0.5 * (angles[1] - (0.+1.j) * angles[2])/c)

                # Givens update of A, V  --> G*AG, VG
                if (np.abs(s) > atol) and (sweep < max_iter):
                    encore = True
                    upds = upds + 1
                    G = np.array([[c, -np.conj(s)] , [s, c] ])
                    pair = (p,q)
                    V[:, pair] = V[:, pair].dot(G)
                    A[pair,:] = G.conj().T.dot(A[pair,:])
                    A[:,np.concatenate([Ip,Iq])] = np.append(c*A[:,Ip]+s*A[:,Iq], -np.conj(s)*A[:,Ip]+c*A[:,Iq], axis=1)
        updates = updates + upds

#        To see progress on off-diagonal sums, uncomment:
        # On = 0.
        # Diag = np.zeros(m, dtype=np.complex128)
        # Range = np.arange(m)
        # for im in range(n):
        #     Diag = np.diag(A[:,Range])
        #     On = On + (Diag*Diag.conj()).sum(axis=0)
        #     Range = Range + m
        # Off = (np.multiply(A,A.conj()).sum(axis=0)).sum(axis=0) - On
        # print Off


        # enforce ordering of diagonal entries
        if (ensure_ordered_diag == True) and (sweep < max_iter):
            diags_to_order = 0  # np.random.randint(n)  # sweep % n
            if unitary == True:
                Re = np.real(np.diagonal(A[:, diags_to_order*m:m*(diags_to_order + 1) ]))
                Im = np.imag(np.diagonal(A[:, diags_to_order*m:m*(diags_to_order + 1) ]))
                GhTo = .2123892 * Re - .42319424 * Im   # any random numbers here will do
            else:
                GhTo = np.diagonal(A[:, diags_to_order*m:m*(diags_to_order + 1)])
            idx = GhTo.argsort()
            V = V[:,idx]
            idx1 = idx.copy()
            idx += diags_to_order * m
            A = A[idx1,:]
            for i in xrange(n):
                idx = idx1 + i * m
                A[:, i*m:m*(i + 1)] = A[:,idx]

    return V, A
