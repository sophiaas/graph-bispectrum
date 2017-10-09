# encoding: utf-8
# cython: profile=False
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

from simulortho import simultaneously_orthogonalize
from util import direct_sum, memoize_method, nullspace, sparse_nullspace

cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t


class Partition(list):

    def __init__(self, *args):
        super(Partition, self).__init__(*args)

    def __hash__(self):
        return hash(unicode(self))

    def restrictions(self):
        result = []
        for i in xrange(len(self)):
            if i != len(self) - 1 and self[i] <= self[i + 1]:
                continue

            p = Partition(self)
            p[i] -= 1
            if(p[i] == 0):
                p.pop()
            result.append(p)

        return result

    def __unicode__(self):
        return "(%s)" % ",".join((unicode(item) for item in self))


class StandardTableau(list):

    @classmethod
    def from_partition(cls, partition):
        obj = StandardTableau()
        c = 1
        for i in xrange(len(partition)):
            s = range(c, c + partition[i])
            c += partition[i]
            obj.append(s)
        return obj

    @classmethod
    def from_tableau(cls, T):
        obj = StandardTableau()
        obj.extend((list(row) for row in T))
        return obj

    def __init__(self, *args):
        super(StandardTableau, self).__init__()
        for i, y in enumerate(args):
            for j in xrange(len(self) + 1, y + 1):
                self.append([])
            self[y - 1].append(i + 1)

    def shape(self):
        result = Partition()
        for i in xrange(len(self)):
            self.append(len(self[i]))
        return result;

    def grow_to(self, partition):
      n = sum(partition)
      for i in xrange(len(self)):
          try:
             p = partition[i]
          except IndexError:
             p = 0

          if len(self[i]) + 1 == p:
              self[i].append(n)
              return self
      self.append([n])
      return self

    def apply_transposition(self, t):
        ai = aj = -1
        i = 0
        while ai < 0 and i < len(self):
            for j in xrange(len(self[i])):
                if self[i][j] == t:
                    ai = i
                    aj = j
                    break
            i += 1

        # Find t in tableau...
        bi = bj = -1
        i = 0
        while bi < 0 and i < len(self):
            for j in xrange(len(self[i])):
                if self[i][j] == t + 1:
                    bi = i
                    bj = j
                    break
            i += 1

        # a very weird signed distance returned here...
        distance = (bj - bi) - (aj - ai)

        # These are the only two ways that applying the transpisition
        # could yield a non-standard tableau.
        if (ai == bi and aj + 1 == bj) or (aj == bj and ai + 1 == bi):
            return False, distance

        self[bi][bj] = t
        self[ai][aj] = t + 1
        return True, distance

    def __unicode__(self):
        return os.linesep.join((" ".join((str(n) for n in it)) for it in self))


class SnIrreducible(object):

    tableaux_computed = False

    YOR_computed = False

    def __hash__(self):
        return hash(self.partition)

    def __init__(self, group, partition):
        self.group = group
        self.partition = Partition(partition)
        self.n = group.n
        self.tableaus = []
        self.eta_index = []
        self.eta = []

        if self.n == 1:
            self.tableaus.append(StandardTableau.from_partition(partition))
            tableaux_computed = True
            self.degree = 1
            return

        self.degree = 0
        subpartitions = partition.restrictions()
        for sp in subpartitions:
            subrepresentation_index = self.group.subgroup.irreducible(sp)
            subrepresentation = self.group.subgroup.irreducibles[subrepresentation_index]
            self.eta.append(subrepresentation)
            self.eta_index.append(subrepresentation_index)
            self.degree += subrepresentation.degree

    # TABLEAU STUFF
    def tableau(self, t):
        if self.tableaux_computed:
            return StandardTableau.from_tableau(self.tableaus[t])

        for etap in self.eta:
            t -= etap.degree;
            if t < 0:
                tableau = etap.tableau(t + etap.degree)
                return tableau.grow_to(self.partition)

        return StandardTableau.from_partition(self.partition)

    def compute_tableaux(self):
        if self.tableaux_computed:
            return

        for etap in self.eta:
            etap.compute_tableaux()
            for t in etap.tableaus:
                self.tableaus.append(
                    StandardTableau.from_tableau(t).grow_to(self.partition))

        self.tableaux_computed = True

    # YOUNG ORTHOGONAL STUFF

    def young_orthogonal_coefficients(self, tau, T):
        index = T * (self.n - 1) + tau - 1
        if self.YOR_computed:
            return self.tdash[index], self.coeff1[index], self.coeff2[index], index

        if self.tableaux_computed:
            tdash = StandardTableau.from_tableau(self.tableaus[T])
        else:
            tdash = self.tableau(T);

        transposition_result, distance = tdash.apply_transposition(tau)
        c1 = 1.0 / distance
        # Case 1: result is linear combination
        if transposition_result:
            # find the index of Tdash...

            for i in xrange(self.degree):
                if tdash == self.tableaus[i]:
                    return i, c1, (1 - pow(c1, 2)) ** 0.5, index
        # Case 2: result is just rescaling
        else:
            return -1, c1, None, index

    def compute_YOR(self):
        self.compute_tableaux()

        cdef tdash = np.zeros(self.degree * (self.n - 1), dtype=np.int)
        cdef coeff1 = np.zeros(self.degree * (self.n - 1), dtype=np.float)
        cdef coeff2 = np.zeros(self.degree * (self.n - 1), dtype=np.float)

        for t in xrange(self.degree):
            for j in xrange(1, self.n):
                td, c1, c2, index = self.young_orthogonal_coefficients(j, t)
                tdash[index] = td
                coeff1[index] = c1
                coeff2[index] = c2

        self.tdash = tdash
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.YOR_computed = True

    # MATRIX STUFF
    def apply_cycle_l(self, j, M, m=-1, inverse=False):
        cdef np.ndarray done = np.zeros(self.degree, dtype=np.int)

        if not self.YOR_computed:
            self.compute_YOR()

        if m == -1:
            m = self.n

        for p in xrange(m - 1, j - 1, -1):
            tau = p
            if inverse:
                tau = j + m - 1 - p

            for t in xrange(self.degree):
                done[t] = 0

            # tabloid[T] goes to ...
            for t in xrange(self.degree):
                if done[t]:
                    continue

                tdash, c1, c2, index = self.young_orthogonal_coefficients(tau, t)

                if tdash == -1:
                    M[t, :] *= c1
                else:
                    for i in xrange(self.degree):
                        temp = M[t, i]
                        M[t, i] = c1 * temp + c2 * M[tdash, i]
                        M[tdash, i] = -c1 * M[tdash, i] + c2 * temp
                    done[tdash] = 1

    def apply_cycle_r(self, j, M, m, inverse):
        cdef np.ndarray done = np.zeros(self.degree, dtype=np.int)

        if not self.YOR_computed:
            self.compute_YOR()

        if m == -1:
            m = self.n

        for p in xrange(m - 1, j - 1, -1):
            tau = p
            if inverse:
                tau = j + m - 1 - p

            for t in xrange(self.degree):
                done[t] = 0

            # tabloid[T] goes to ...
            for t in xrange(self.degree):
                if done[t]:
                    continue

                tdash, c1, c2, index = self.young_orthogonal_coefficients(tau, t)

                if tdash == -1:
                    M.resize(self.degree * self.degree)
                    for i in xrange(t, t + self.degree * (self.degree - 1), self.degree):
                        M[i] = c1 * M[i]
                    M.resize((self.degree, self.degree))
                else:
                    for i in xrange(self.degree):
                        temp = M[t, i]
                        M[t, i] = c1 * temp + c2 * M[tdash, i]
                        M[tdash, i] = -c1 * M[tdash, i] + c2 * temp
                    done[tdash] = 1

    def apply_transposition(self, j, M):
        cdef np.ndarray done = np.zeros(self.degree, dtype=np.int)

        taupre = j
        for taupre in xrange(j, 2 * self.n - 2 - j + 1):
            tau = taupre

            if tau > self.n - 1:
                tau = (self.n - 1) - (tau - (self.n - 1))

            for t in xrange(self.degree):
                done[t] = 0

            # tabloid[T] goes to ...
            for t in xrange(self.degree):
                if done[t]:
                    continue

                tdash, c1, c2, index = self.young_orthogonal_coefficients(tau, t)
                if tdash == -1:
                    for i in xrange(self.degree):
                        M[t, i] = c1 * M[t, i];
                else:
                    for i in xrange(self.degree):
                        temp = M[t, i]
                        M[t, i] = c1 * temp + c2 * M[tdash, i]
                        M[tdash, i] = -c1 * M[tdash, i] + c2 * temp
                    done[tdash] = 1

    def apply_transposition_r(self, j, M):
        cdef np.ndarray done = np.zeros(self.degree, dtype=np.int)

        taupre = j
        for taupre in xrange(j, 2 * self.n - 2 - j + 1):
            tau = taupre

            if tau > self.n - 1:
                tau = (self.n - 1) - (tau - (self.n - 1))

            for t in xrange(self.degree):
                done[t] = 0

            # tabloid[T] goes to ...
            for t in xrange(self.degree):
                if done[t]:
                    continue

                tdash, c1, c2, index = self.young_orthogonal_coefficients(tau, t)
                if tdash == -1:
                    for i in xrange(self.degree):
                        M[t, i] = c1 * M[t, i];
                        i += self.degree
                else:
                    for i in xrange(self.degree):
                        temp = M[t, i]
                        M[t, i] = c1 * temp + c2 * M[tdash, i]
                        M[tdash, i] = -c1 * M[tdash, i] + c2 * temp
                    done[tdash] = 1

    @memoize_method
    def __call__(self, permutation):
        """ Return the representation matrix correspoding to group element p.

        The representation is given in terms of Young's orthogonal basis.
        @deprecated
        """
        cdef np.ndarray v = np.zeros(self.n, dtype=np.int)
        cdef np.ndarray M = np.identity(self.degree, dtype=DTYPE)

        for i in xrange(self.n):
            v[i] = i + 1

        for m in xrange(self.n, 0, -1):
            j = permutation(m - 1)
            self.apply_cycle_l(v[j], M, m, True)
            for i in xrange(j + 2, self.n + 1):
                v[i - 1] -= 1

        return M

    @memoize_method
    def character(self, permutation):
        return self(permutation).trace()

    @memoize_method
    def character_risi(self, mu):
        cdef np.ndarray M = np.identity(self.degree, dtype=DTYPE)
        cdef int m = self.n
        for mu_k in mu:
            if mu_k == 1:
                break
            self.apply_cycle_l(m - mu_k + 1, M, m)
            m -= mu_k
        return M.trace()

    def __unicode__(self):
        return unicode(self.partition)


class SymmetricGroup(PermutationGroup):

    groups = {}

    def __new__(cls, n):
        if n in cls.groups:
            return cls.groups[n]

        if n == 1:
            G = super(SymmetricGroup, cls).__new__(cls, [Permutation([0])])
        elif n == 2:
            G = super(SymmetricGroup, cls).__new__(cls, [Permutation([1, 0])])
        else:
            a = range(1, n)
            a.append(0)
            gen1 = Permutation._af_new(a)
            a = range(n)
            a[0], a[1] = a[1], a[0]
            gen2 = Permutation._af_new(a)
            G = super(SymmetricGroup, cls).__new__(cls, [gen1, gen2])
        if n < 3:
            G._is_abelian = True
            G._is_nilpotent = True
        else:
            G._is_abelian = False
            G._is_nilpotent = False
        if n < 5:
            G._is_solvable = True
        else:
            G._is_solvable = False

        G.Z_cache = {
            (Partition([n]), Partition([n])): [(Partition([n]), 1)],
            (Partition([n]), Partition([n - 1, 1])): [(Partition([n - 1, 1]), 1)],
            (Partition([n]), Partition([n - 2, 2])): [(Partition([n - 2, 2]), 1)],
            (Partition([n]), Partition([n - 2, 1, 1])): [(Partition([n - 2, 1, 1]), 1)],
            (Partition([n - 1, 1]), Partition([n - 1, 1])): [
                (Partition([n]), 1),
                (Partition([n - 1, 1]), 1),
                (Partition([n - 2, 2]), 1 if n > 3 else 0),
                (Partition([n - 2, 1, 1]), 1 if n > 2 else 0)
            ],
            (Partition([n - 1, 1]), Partition([n - 2, 2])): [
                (Partition([n - 1, 1]), 1),
                (Partition([n - 2, 2]), 1 if n > 4 else 0),
                (Partition([n - 2, 1, 1]), 1 if n > 3 else 0),
                (Partition([n - 3, 3]), 1 if n > 5 else 0),
                (Partition([n - 3, 2, 1]), 1 if n > 4 else 0)
            ],
            (Partition([n - 1, 1]), Partition([n - 2, 1, 1])): [
                (Partition([n - 1, 1]), 1),
                (Partition([n - 2, 2]), 1 if n > 3 else 0),
                (Partition([n - 2, 1, 1]), 1 if n > 3 else 0),
                (Partition([n - 3, 3]), 1 if n > 8 else 0),
                (Partition([n - 3, 2, 1]), 1 if n > 4 else 0),
                (Partition([n - 3, 1, 1, 1]), 1 if n > 3 else 0),
            ],
            (Partition([n - 2, 2]), Partition([n - 2, 2])): [
                (Partition([n]), 1),
                (Partition([n - 1, 1]), 1 if n > 4 else 0),
                (Partition([n - 2, 2]), 2 if n > 5 else 1),
                (Partition([n - 2, 1, 1]), 1 if n > 4 else 0),
                (Partition([n - 3, 3]), 1 if n > 6 else 0),
                (Partition([n - 3, 2, 1]), 2 if n > 5 else (0 if n < 5 else 1)),
                (Partition([n - 3, 1, 1, 1]), 1 if n > 3 else 0),
                (Partition([n - 4, 4]), 1 if n > 7 else 0),
                (Partition([n - 4, 3, 1]), 1 if n > 6 else 0),
                (Partition([n - 4, 2, 2]), 1 if n > 5 else 0)
            ],
            (Partition([n - 2, 2]), Partition([n - 2, 1, 1])): [
                (Partition([n - 1, 1]), 1),
                (Partition([n - 2, 2]), 1 if n > 4 else 0),
                (Partition([n - 2, 1, 1]), 2 if n > 4 else (0 if n < 4 else 1)),
                (Partition([n - 3, 3]), 1 if n > 5 else 0),
                (Partition([n - 3, 2, 1]), 2 if n > 5 else (0 if n < 5 else 1)),
                (Partition([n - 3, 1, 1, 1]), 1 if n > 4 else 0),
                (Partition([n - 4, 3, 1]), 1 if n > 6 else 0),
                (Partition([n - 4, 2, 1, 1]), 1 if n > 5 else 0)
            ],
            (Partition([n - 2, 1, 1]), Partition([n - 2, 1, 1])): [
                (Partition([n]), 1),
                (Partition([n - 1, 1]), 1 if n > 3 else 0),
                (Partition([n - 2, 2]), 2 if n > 4 else (0 if n < 4 else 1)),
                (Partition([n - 2, 1, 1]), 1 if n > 3 else 0),
                (Partition([n - 3, 3]), 1 if n > 5 else 0),
                (Partition([n - 3, 2, 1]), 2 if n > 4 else 0),
                (Partition([n - 3, 1, 1, 1]), 1 if n > 4 else 0),
                (Partition([n - 4, 2, 2]), 1 if n > 5 else 0),
                (Partition([n - 4, 2, 1, 1]), 1 if n > 5 else 0),
                (Partition([n - 4, 1, 1, 1, 1]), 1 if n > 4 else 0)
            ],
        }
        G._degree = n
        G._is_transitive = True
        G._is_sym = True
        G.groups[n] = G

        G.irreducibles = []
        G.irreducibles_by_partition = {}
        G.n = n

        # bottom of recursive sequence ...
        if G.n == 1:
            lmbda = Partition()
            lmbda.append(1)
            G.irreducibles.append(SnIrreducible(G, lmbda))
            G._register_irreducibles()
            return G

        G.subgroup = SymmetricGroup(G.n - 1)
        for i in xrange(len(G.subgroup.irreducibles)):
            lmbda = Partition(G.subgroup.irreducibles[i].partition)
            if len(lmbda) == 1 or lmbda[len(lmbda) - 1] < lmbda[len(lmbda) - 2]:
                lmbda[len(lmbda) - 1] += 1
                G.irreducibles.append(SnIrreducible(G, lmbda))
                lmbda[len(lmbda) - 1] -= 1

            lmbda.append(1);
            G.irreducibles.append(SnIrreducible(G, lmbda))

        G._register_irreducibles()
        return G

    def _register_irreducibles(self):
        for index, rho in enumerate(self.irreducibles):
            rho.index = index
            self.irreducibles_by_partition[rho.partition] = (index, rho)

    def generate(self):
        for perm in xrange(self.order()):
            yield(self._generate(perm))

    @memoize_method
    def _generate(self, perm):
        cdef np.ndarray v = np.zeros(self.n, dtype=np.int)
        for i in xrange(self.n + 1):
            v[i - 1] = i
        cdef int p = perm
        cdef int res = 0
        cdef int j = 0
        cdef int t = 0
        for k in xrange(2, self.n + 1):
            res = p % k
            p = (p - res) // k
            j = k - res
            t = v[k - 1]
            i = k - 1
            while i >= j:
                v[i + 1 - 1] = v[i - 1]
                i -= 1
            v[j - 1] = t

        return Permutation(np.array(v) - 1)

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
            l.character(g) * m.character(g) * v.character(g)  \
            for g in self.generate()))))

    def multiplicity(self, l, m):
#        return [(v.partition, self.Z(l, m, v)) for v in self.irreducibles]
        p1, p2 = l.partition, m.partition
        if (p1, p2) in self.Z_cache:
            return self.Z_cache[(p1, p2)]
        if (p2, p1) in self.Z_cache:
            return self.Z_cache[(p2, p1)]
        if p1 == [self.n]:
            return [(p2, 1)]
        if p2 == [self.n]:
            return [(p1, 1)]

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
            return sparse.identity(m.degree)
        if m.index == 0:
            return sparse.identity(l.degree)

        Zv = self.multiplicity(l, m)
        R = self.intertwiner(l, m)
        C = sparse.lil_matrix(R.shape)
        Q = R.dot(R.T)

        o = 0
        for p, zv in Zv:
            if zv > 0:
                v = self.irreducible(p, index=False)
                degree = v.degree
                M_tensor_I = Q[o:o + degree*zv, o:o + degree*zv]
                mdim = M_tensor_I.shape[0] / degree
                M = np.zeros((mdim, mdim))
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
        randn = np.random.randn(null_M[0].shape[0])
        R = sparse.lil_matrix((null_M[0].shape[0], 1))
        for i in xrange(ncols):
            null_M[i] *= randn[i]
            R += null_M[i]
        return R.reshape((d, d)).tocsc().T

    def irreducible(self, partition, index=True):
        return self.irreducibles_by_partition.get(partition, (None, None))[0 if index else 1]
