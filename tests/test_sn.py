import itertools
import logging
import os

import numpy as np
np.set_printoptions(linewidth=200)

from nose.tools import assert_equal, assert_not_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from sympy.combinatorics.permutations import Permutation

from graphbispectrum import Function, Partition, SymmetricGroup
from graphbispectrum.util import simultaneously_diagonalize

from utils import assert_permutation_equal, skip


class TestSymmetricGroup(object):

    def setUp(self):
        self.s3 =  SymmetricGroup(3)
        self.s4 =  SymmetricGroup(4)
        self.s5 =  SymmetricGroup(5)

    def tearDown(self):
        pass

    def test_homomorphism(self):
        for irr in self.s5.irreducibles:
            for g, h in zip(self.s5.generate(), self.s5.generate()):
                assert_array_almost_equal(irr(g * h), irr(g).dot(irr(h)), 1)

    def test_irreducibles(self):
        s5_irreducibles = [
            "(5)",
            "(4,1)",
            "(3,2)",
            "(3,1,1)",
            "(2,2,1)",
            "(2,1,1,1)",
            "(1,1,1,1,1)"]
        calc_s5_irreducibles = [unicode(irr) for irr in self.s5.irreducibles]
        assert_array_equal(calc_s5_irreducibles, s5_irreducibles)

    def test_subgroup(self):
        s3_irreducibles = [
            "(3)",
            "(2,1)",
            "(1,1,1)"]

        s3 = self.s5.subgroup.subgroup
        calc_s3_irreducibles = [unicode(irr) for irr in s3.irreducibles]
        assert_array_equal(calc_s3_irreducibles, s3_irreducibles)

    def ancestors(self, rho, indenter=""):
        result = [indenter, unicode(rho), os.linesep]
        for i in xrange(len(rho.eta)):
            result.append(self.ancestors(rho.eta[i], indenter + "  "))
        return "".join(result)

    def test_ancestors(self):
        expected = [
            "(5)",
            "  (4)",
            "    (3)",
            "      (2)",
            "        (1)",
            "(4,1)",
            "  (3,1)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "    (3)",
            "      (2)",
            "        (1)",
            "  (4)",
            "    (3)",
            "      (2)",
            "        (1)",
            "(3,2)",
            "  (2,2)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "  (3,1)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "    (3)",
            "      (2)",
            "        (1)",
            "(3,1,1)",
            "  (2,1,1)",
            "    (1,1,1)",
            "      (1,1)",
            "        (1)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "  (3,1)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "    (3)",
            "      (2)",
            "        (1)",
            "(2,2,1)",
            "  (2,1,1)",
            "    (1,1,1)",
            "      (1,1)",
            "        (1)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "  (2,2)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "(2,1,1,1)",
            "  (1,1,1,1)",
            "    (1,1,1)",
            "      (1,1)",
            "        (1)",
            "  (2,1,1)",
            "    (1,1,1)",
            "      (1,1)",
            "        (1)",
            "    (2,1)",
            "      (1,1)",
            "        (1)",
            "      (2)",
            "        (1)",
            "(1,1,1,1,1)",
            "  (1,1,1,1)",
            "    (1,1,1)",
            "      (1,1)",
            "        (1)",
            ""]

        calc = [self.ancestors(irr) for irr in self.s5.irreducibles]
        assert_equal(os.linesep.join(expected), "".join(calc))

    def test_tableux(self):
        rho = self.s5.irreducibles[2]
        lmbda = Partition([2, 2, 1])
        mu = Partition([3, 1, 1])
        sigma = Permutation([1, 2, 0, 3, 4])

        assert_equal(unicode(rho), "(3,2)")
        assert_equal(rho.degree, 5)

        expected_tableaux = [
            [
                [1, 3, 5],
                [2, 4]
            ],
            [
                [1, 2, 5],
                [3, 4]
            ],
            [
                [1, 3, 4],
                [2, 5]
            ],
            [
                [1, 2, 4],
                [3, 5]
            ],
            [
                [1, 2, 3],
                [4, 5]
            ]
        ]

        for i in xrange(rho.degree):
            assert_array_equal(expected_tableaux[i], rho.tableau(i))
        assert_equal(rho.character_risi(mu), -1)

        expected_rho_sigma = np.ndarray((5, 5))
        expected_rho_sigma[0][0:] = [-0.5, 0.8660254, 0, 0, 0]
        expected_rho_sigma[1][0:] = [-0.8660254, -0.5, 0, 0, 0]
        expected_rho_sigma[2][0:] = [0, 0, -0.5, 0.8660254, 0]
        expected_rho_sigma[3][0:] = [0, 0, -0.8660254, -0.5, 0]
        expected_rho_sigma[4][0:] = [0, 0, 0, 0, 1]
        assert_array_almost_equal(rho(sigma), expected_rho_sigma)

    def test_representation(self):
        rho = self.s5.irreducibles[2];
        sigma1 = Permutation([0, 1, 3, 2, 4])
        sigma2 = Permutation([1, 2, 0, 3, 4])

        expected_rho_sigma = np.ndarray((5, 5))
        expected_rho_sigma[0][0:] = [0.5, 0.8660254, 0, 0, 0]
        expected_rho_sigma[1][0:] = [0.8660254, -0.5, 0, 0, 0]
        expected_rho_sigma[2][0:] = [0, 0, -0.5, 0.28867513, 0.81649658]
        expected_rho_sigma[3][0:] = [0, 0, -0.8660254, -0.1666666667, -0.47140452]
        expected_rho_sigma[4][0:] = [0, 0, 0, 0.94280904, -0.33333333]

        x = rho(sigma1)
        y = rho(sigma2)

        assert_array_almost_equal(rho(sigma2 * sigma1), expected_rho_sigma)
        assert_array_almost_equal(rho(sigma2 * sigma1), rho(sigma2).dot(rho(sigma1)))

    def test_fft(self):
        f = Function(self.s5)
        f.randomize()
        assert_array_almost_equal(f.f, f.fft().iFFT().f)

    def test_edge_extensions(self):
        assert_equal(
            len(list(self.s5.generate_edge_extensions(0, 1))),
            self.s5.subgroup.subgroup.order())

    def test_Z(self):
        irr1 = self.s5.irreducibles[0]
        irr2 = self.s5.irreducibles[1]
        irr3 = self.s5.irreducibles[2]
        irr5 = self.s5.irreducibles[4]

        dim = sum((self.s5.Z(irr2, irr5, v) * v.degree for v in self.s5.irreducibles))
        assert_almost_equal(dim, irr2.degree * irr5.degree)

        assert_equal(self.s5.Z(irr1, irr2, irr3), self.s5.Z(irr1, irr3, irr2))
        assert_equal(self.s5.Z(irr3, irr2, irr1), self.s5.Z(irr3, irr2, irr1))
        assert_almost_equal(self.s5.Z(irr2, irr1, irr3), 0)
        assert_almost_equal(self.s5.Z(irr3, irr1, irr2), 0)
        assert_almost_equal(self.s5.Z(irr5, irr2, irr5), 1)

    def test_direct_sum(self):
        l = self.s5.irreducibles[1]
        m = self.s5.irreducibles[2]
        dim = l.degree * m.degree
        for sigma in self.s5.generate():
            ds = self.s5.direct_sum(l, m, sigma).todense()
            assert_array_almost_equal(ds.dot(ds.T), np.identity(dim))

    def test_intertwiner(self):
        s = SymmetricGroup(4)
        for l, m in itertools.product(s.irreducibles[:3], s.irreducibles[:3]):
            dim = l.degree * m.degree
            # print l.partition, m.partition
            R = s.intertwiner(l, m).todense()
            assert_equal(R.shape, (dim, dim))
            for sigma in s.generate():
                assert_not_equal(np.linalg.norm(R), 0)
                assert_array_almost_equal(
                    R.dot(np.kron(l(sigma), m(sigma))),
                    s.direct_sum(l, m, sigma).dot(R))

    def test_hillartwiner(self):
        s = SymmetricGroup(7)
        for l, m in itertools.product(s.irreducibles[:3], s.irreducibles[:3]):
            print unicode(l), unicode(m)
            dim = l.degree * m.degree
            R = s.hillartwiner(l, m).todense()
            assert_equal(R.shape, (dim, dim))

            g,h = s[0], s[1]  # generators of sn

            Y0 = np.kron(l(g), m(g))
            X0 = s.direct_sum(l, m, g)
            X0 = X0.todense()
            Y1 = np.kron(l(h), m(h))
            X1 = s.direct_sum(l, m, h)
            X1 = X1.todense()

            assert_array_almost_equal(
                R.dot(np.kron(l(g), m(g))),
                s.direct_sum(l, m, g).dot(R), 4)
            assert_array_almost_equal(
                R.dot(np.kron(l(h), m(h))),
                s.direct_sum(l, m, h).dot(R), 4)


    def test_clebsch_gordan(self):
        s = SymmetricGroup(6)
        for l, m in itertools.product(s.irreducibles[:3], s.irreducibles[:3]):
            # print unicode(l), unicode(m)
            # print [(unicode(v), s.Z(l, m, v)) for v in s.irreducibles]
            dim = l.degree * m.degree
            C = s.unitary_intertwiner(l, m).todense()
            sigma = s.random()
            assert_array_almost_equal(
                C.dot(np.kron(l(sigma), m(sigma))),
                s.direct_sum(l, m, sigma).dot(C))
            assert_array_almost_equal(C.dot(C.T), np.identity(dim))

    def test_simultaneously_diagonalize(self):
        I = np.random.randn(4, 4)
        I = I + I.T
        V, D = simultaneously_diagonalize(np.hstack([I, I.dot(I)]))
        assert_array_almost_equal(V.dot(V.T), np.identity(I.shape[0]))
        X = V.T.dot(I).dot(V)
        assert_array_almost_equal(X, np.diag(X.diagonal()))
        X = V.T.dot(I.dot(I)).dot(V)
        assert_array_almost_equal(X, np.diag(X.diagonal()))

        # complex version: eigs of matrices to diagonalize are complex
        I = np.random.randn(4, 4) + (0.+1.j) * np.random.randn(4, 4)
        I = I + I.conj().T
        V, D = simultaneously_diagonalize(np.hstack([I, I.dot(I)]), unitary=True)
        assert_array_almost_equal(V.dot(V.conj().T), np.identity(I.shape[0], dtype=np.complex128))
        X = V.conj().T.dot(I).dot(V)
        assert_array_almost_equal(X, np.diag(X.diagonal()))
        X = V.conj().T.dot(I.dot(I)).dot(V)
        assert_array_almost_equal(X, np.diag(X.diagonal()))

    @skip
    def test_failcase_clebsch_gordan(self):
        s = SymmetricGroup(8)  # n=6 break because of raise LinAlgError, 'SVD did not converge'
        l = s.irreducibles[2]
        m = s.irreducibles[3]
        # mult = [(unicode(v), s.Z(l, m, v)) for v in s.irreducibles[:8]]
        # print sum(mult)
        # print mult
        dim = l.degree * m.degree
        print l.degree, m.degree
        print dim
        print "---"
        for v in s.irreducibles[:10]:
            print unicode(v), v.degree

        C = s.unitary_intertwiner(l, m).todense()
        print l.degree, m.degree
        print C.shape
        # assert_array_almost_equal(C.dot(C.T), np.identity(dim))
        sigma = s.random()
        assert_array_almost_equal(
            C.dot(np.kron(l(sigma), m(sigma))),
            s.direct_sum(l, m, sigma).dot(C))
