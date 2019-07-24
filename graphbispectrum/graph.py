import itertools
import numpy as np

from scipy import sparse
from sympy.combinatorics.permutations import Permutation

from .function import Function, FourierTransform
from .symmetric_group import Partition, SymmetricGroup
from .util import direct_sum, memoize_method
from .util.distributions import corrupt_binary_data


class Graph(Function):

    @classmethod
    def erdos_renyi(cls, n, p=0.5, edges=None):
        e = (n * (n - 1)) / 2
        if edges is None:
            edges = corrupt_binary_data(np.zeros(e, dtype=np.float), avg_bits_corrupted=p * e)
        else:
            edges = corrupt_binary_data(np.zeros(e, dtype=np.float), bits_corrupted=edges)
        return cls.from_edges(edges)

    @classmethod
    def lex_index(cls, i, j, V):
        """ Index into (1D) edge array. """
        if i < j:
            return i * V - i * (i + 1) // 2 + j - i - 1
        return cls.lex_index(j, i, V)
    
    @classmethod
    def from_edges(cls, adjacency_matrix):
        """ Construct a graph from a list of edge weights. """
        return cls(adjacency_matrix)

#     @classmethod
#     def from_edges(cls, edges):
#         """ Construct a graph from a list of edge weights. """
#         edges = np.array(edges)
#         V = int((1.0 +  (1.0 + 8 * edges.shape[0]) ** 0.5) / 2.0)
#         adjacency_matrix = np.zeros((V, V), dtype=edges.dtype)
#         for i, j in itertools.product(range(V), range(V)):
#             if i != j:
#                 adjacency_matrix[i, j] = edges[cls.lex_index(i, j, V)]
#         return cls(adjacency_matrix)

    @property
    def edges(self):
        """ Constructs an edge vector from the adjacency matrix. """
        return self.adj_matrix[np.triu_indices(self.sn.n, 1)]

    @property
    def f(self):
        if not hasattr(self, "_f"):
            self._f = np.zeros(self.sn.order(), dtype=np.float)
            for g_index, g in enumerate(self.sn.generate()):
                self._f[g_index] = self(g)
        return self._f

    @property
    def V(self):
        """ Constructs an edge vector from the adjacency matrix. """
        return self.adj_matrix.shape[0]

    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.sn = SymmetricGroup(self.V)

    def __call__(self, g):
        glist = g.list()
        return self.adj_matrix[glist[self.sn.n - 2], glist[self.sn.n - 1]]

    def __hash__(self):
        return hash(unicode(self.edges))

    def apply_permutation(self, p):
        """ Return a new graph with the permutation applied. """
        return Graph(self.adj_matrix[list(p), :][:, list(p)])

    @memoize_method
    def fft_nonzero_partitions(self, as_dict=False):
        if as_dict:
            return dict((p, p) for p in self.fft_nonzero_partitions())

        return [
            Partition([self.sn.n]),
            Partition([self.sn.n - 1, 1]),
            Partition([self.sn.n - 2, 2]),
            Partition([self.sn.n - 2, 1, 1])
        ]

    @memoize_method
    def fft(self):
        if self.sn.n <= 3:
            fft = super(Graph, self).fft()
            fft.matrix = [sparse.csc_matrix(m) for m in fft.matrix]
            return fft
        return FourierTransform(self, [self.compute_fft(rho) for rho in self.sn.irreducibles])

    @memoize_method
    def sparse_fft(self):
        if self.sn.n <= 3:
            return self.fft()
        irreducibles = (self.sn.irreducible(p, index=False) for p in self.fft_nonzero_partitions())
        return FourierTransform(self, [self.compute_fft(rho) for rho in irreducibles])

    def compute_fft(self, rho):
        if rho.partition not in self.fft_nonzero_partitions(as_dict=True):
            return sparse.csc_matrix((rho.degree, rho.degree))

        result = 0
        for i, j in itertools.product(range(self.sn.n), range(self.sn.n)):
            if i == j:
                continue

            isn = list(range(self.sn.n))
            isn.remove(i)
            isn.remove(j)
            sigma = Permutation(isn + [i, j])
            restrictions = rho.partition.restrictions()
            if self(sigma) or type(result) == int:
                result += self(sigma) * rho(sigma)

        o = 0
        for tau_ in restrictions:
            for tau in tau_.restrictions():
                subgroup = self.sn.subgroup.subgroup
                if len(tau) == 1:
                    result[o, :] = result[o, :] * subgroup.order()
                    o += 1
                else:
                    deg = subgroup.irreducible(tau, index=False).degree
                    result[o:o + deg, :] *= 0.
                    o += deg

        return sparse.csc_matrix(result).T

    def direct_sum(self, l, m):
        fft = self.fft().matrix
        return sparse.block_diag([
            fft[self.sn.irreducible(p, index=False).index] \
            for p, z in self.sn.multiplicity(l, m) for zi in range(z)])

    def bispectrum(self, idx=None):
        if idx is None:
            idx = tuple((self.sn.irreducible(p) for p in self.fft_nonzero_partitions() \
                if self.sn.irreducible(p) is not None))
        fft = self.sparse_fft().matrix
        return super(Graph, self).bispectrum(idx)

    def bispectrum_element(self, l_index, m_index):
        fft = self.sparse_fft().matrix
        # return super(Graph, self).bispectrum_element(l_index, m_index, fft)
        l = self.sn.irreducibles[l_index]
        m = self.sn.irreducibles[m_index]
        return sparse.kron(fft[l_index], fft[m_index]).conjugate().T.dot(
            self.sn.clebsch_gordan(l, m).conjugate().T).dot(self.direct_sum(l, m))
