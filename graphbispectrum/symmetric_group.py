# encoding: utf-8
import itertools
import logging
import os
import numpy as np

from collections import defaultdict
from scipy import sparse
from sympy import BlockDiagMatrix, MatAdd, MatMul, Matrix, MatrixSymbol, solvers
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
from scipy.io import mmread, mmwrite

from csymmetric_group import Partition, StandardTableau, CSymmetricGroup
from simulortho import simultaneously_orthogonalize
from util import direct_sum, memoize_method, nullspace, sparse_nullspace


class SymmetricGroup(CSymmetricGroup):

    @memoize_method
    def sum_rho_edge_extensions(self, rho, i, j):
        return sum(rho(g) for g in self.generate_edge_extensions(i, j))

    def generate_edge_extensions(self, i, j):
        for perm in self.subgroup.subgroup.generate():
            elems = [i, j]
            for elem in perm.list():
                if elem < min(i, j):
                    elems.append(elem)
                elif elem < max(i, j) - 1:
                    elems.append(elem + 1)
                else:
                    elems.append(elem + 2)
            yield Permutation(elems)

    def Z(self, l, m, v):
        """ Returns multiplicity in clebsh-gordan matrix. """
        return int(round(1.0 / self.order() * sum((
            l.character(g) * m.character(g) * v.character(g)
            for g in self.generate()))))

    def multiplicity(self, l, m):
        p1, p2 = l.partition, m.partition
        if (p1, p2) in self.Z_cache:
            return self.Z_cache[(p1, p2)]
        if (p2, p1) in self.Z_cache:
            return self.Z_cache[(p2, p1)]
        if p1 == [self.n]:
            return [(p2, 1)]
        if p2 == [self.n]:
            return [(p1, 1)]
        return [(v.partition, self.Z(l, m, v)) for v in self.irreducibles]

    def direct_sum(self, l, m, g):
        Zv = self.multiplicity(l, m)
        return sparse.block_diag([
            self.irreducible(v, index=False)(g) for v, z in Zv for zi in xrange(z)])

    @memoize_method
    def clebsch_gordan(self, l, m):
        cg_filename = "C%d_%d_%d.mtx" % (self.n, l.index, m.index)
        cg_cache = os.path.join(os.path.dirname(__file__), "CG_cache", cg_filename)
        if os.path.exists(cg_cache):
            logging.info(u"Loading clebsh-gordan for %s, %s from cache" % (l, m))
            return mmread(cg_cache)

        logging.info(u"Computing clebsh-gordan for %s, %s" % (l, m))
        C = self.unitary_intertwiner(l, m)
        mmwrite(cg_cache, C)
        return C

    def n_cycles(self, n):
        if n == 0:
            yield Permutation([], size=self.n)
        elif n == 1:
            for i in xrange(self.n):
                for j in xrange(i + 1, self.n):
                    yield Permutation(i, j, size=self.n)
        elif n == 2:
            for i in xrange(self.n):
                for j in xrange(i + 1, self.n):
                    for k in xrange(j + 1, self.n):
                        yield Permutation(i, j, k, size=self.n)
                        yield Permutation(i, k, j, size=self.n)
        elif n == 3:
            yield Permutation(0, 1, 2, 3, size=self.n)
            yield Permutation(0, 1, 3, 2, size=self.n)
            yield Permutation(0, 2, 1, 3, size=self.n)
            yield Permutation(0, 2, 3, 1, size=self.n)
            yield Permutation(0, 3, 1, 2, size=self.n)
            yield Permutation(0, 3, 2, 1, size=self.n)
        elif n == 4:
            yield Permutation([[0, 1], [2, 3]], size=self.n)
            yield Permutation([[0, 2], [1, 3]], size=self.n)
            yield Permutation([[0, 3], [1, 2]], size=self.n)

    def hillartwiner(self, l, m, num_trials=2):
        # Make trivial tensors (n) x (*) = (*) have intertwiner = Identity
        if l.index == 0:
            return sparse.identity(m.degree)
        if m.index == 0:
            return sparse.identity(l.degree)

        A1 = self.direct_sum(l, m, self[0]).todense()
        A2 = self.direct_sum(l, m, self[1]).todense()
        B1 = np.kron(l(self[0]), m(self[0]))
        B2 = np.kron(l(self[1]), m(self[1]))
        C = simultaneously_orthogonalize(A1, A2, B1, B2)
        return sparse.csc_matrix(C)

    def symmetric_random_combination(self, A1, B1, A2, B2):
        t1 = np.random.randn()
        t2 = np.random.randn()
        S = t1 * A1 + t2 * A2
        S += S.T
        T = t1 * B1 + t2 * B2
        T += T.T
        return S, T

    def unitary_intertwiner(self, l, m):
        # Make trivial tensors (n) x (*) = (*) have intertwiner = Identity
        if l.index == 0:
            return sparse.identity(m.degree, dtype=np.complex128)
        if m.index == 0:
            return sparse.identity(l.degree, dtype=np.complex128)

        Zv = self.multiplicity(l, m)
        R = self.intertwiner(l, m)
        C = sparse.lil_matrix(R.shape, dtype=np.complex128)
        Q = R.dot(R.T)

        o = 0
        for p, zv in Zv:
            if zv > 0:
                v = self.irreducible(p, index=False)
                degree = v.degree
                M_tensor_I = Q[o:o + degree*zv, o:o + degree*zv]
                mdim = M_tensor_I.shape[0] / degree
                M = np.zeros((mdim, mdim), dtype=np.complex128)
                for i in xrange(mdim):
                    for j in xrange(mdim):
                        M[i][j] = M_tensor_I[i * degree, j * degree]

                Sv, d, _ = np.linalg.svd(M)
                Sv = sparse.csc_matrix(Sv)
                for i in xrange(len(d)):
                     if abs(d[i]) > 0:
                         d[i] **= -0.5

                D = sparse.diags(d, 0)
                Id = sparse.identity(degree)
                Rsub = R[o:o + degree * zv]
                C_ = sparse.kron(D.dot(Sv.T), Id).dot(Rsub).tolil()
                for i in xrange(C_.shape[0]):
                    for j in xrange(C_.shape[1]):
                        C[o + i, j] = C_[i, j]
                o += degree * zv
        return C

    def intertwiner(self, l, m):
        # Make trivial tensors (n) x (*) = (*) have intertwiner = Identity
        if l.index == 0:
            return sparse.identity(m.degree)
        if m.index == 0:
            return sparse.identity(l.degree)

        A1 = self.direct_sum(l, m, self[0])
        A2 = self.direct_sum(l, m, self[1])
        B1 = sparse.kron(l(self[0]), m(self[0]))
        B2 = sparse.kron(l(self[1]), m(self[1]))
        d = A1.shape[0]
        M1 = sparse.kronsum(A1, -1. * B1.T, "csc")
        M2 = sparse.kronsum(A2, -1. * B2.T, "csc")
        M = sparse.vstack([M1, M2], "csc")
        null_M = sparse_nullspace(M)

        ncols = len(null_M)
        randn_r = np.random.randn(null_M[0].shape[0])
        randn_c = np.random.randn(null_M[0].shape[0])
        randn = np.vectorize(complex)(randn_r, randn_c)
        R = sparse.lil_matrix((null_M[0].shape[0], 1), dtype=np.complex128)
        for i in xrange(ncols):
            # null_M[i] *= randn[i]
            R += null_M[i].astype(np.complex128) * randn[i]
        return R.reshape((d, d)).tocsc().T
