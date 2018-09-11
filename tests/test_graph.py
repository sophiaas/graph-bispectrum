import logging
import os

import numpy as np

from nose.tools import assert_equal, assert_greater_equal, assert_less_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from sympy.combinatorics.permutations import Permutation

from graphbispectrum import Function, Graph, Partition, SymmetricGroup

from utils import assert_permutation_equal, kbits, skip


class TestGraph(object):

    def setUp(self):
        self.s3 = SymmetricGroup(3)
        self.s4 = SymmetricGroup(4)
        self.s5 = SymmetricGroup(5)
        self.f1 = Graph.from_edges([0, 1, 1])
        self.sigma = Permutation([2, 1, 0])

    def tearDown(self):
        pass

    def test_apply_permutation(self):
        f2 = self.f1.apply_permutation(self.sigma)
        assert_array_equal(f2.edges, [1, 1, 0])

    def calculate_invariant(self, invariant_name, n=4, sparse=False, **kwargs):
        iso_classes = {}
        tolerance=1.0
        e = int(round(n * (n - 1.0) / 2.0))
        for i in xrange(e + 1):
            logging.info("%d 1s in %d edges." % (i, e))
            for edges in kbits(e, i):
                f = Graph.from_edges(edges)
                invariant_value = getattr(f, invariant_name)(**kwargs)
                if sparse:
                    invariant_value = [inv.todense() for inv in invariant_value]

                found = False
                for key, (fdash, value) in iso_classes.iteritems():
                    diff = np.sum([
                        np.linalg.norm(v1 - v2) for v1, v2 in zip(invariant_value, value)
                    ])
                    if diff < tolerance:
                        found = key
                        break

                if not found:
                    key = unicode(invariant_value).encode("bz2")
                    iso_classes[key] = ([(f, invariant_value)], invariant_value)
                else:
                    fdash.append((f, invariant_value))

        return iso_classes

    def test_fft(self):
        f = Graph.from_edges([0, 1, 1])
        assert_array_almost_equal(f.f, [1, 1, 1, 1, 0, 0])
        expected = [[[4]], [[0, 0], [1.73205, 1]], [[0]]]
        for i, ex in enumerate(expected):
            assert_array_almost_equal(f.fft().matrix[i].todense(), np.array(ex))

    def test_graph_fft(self):
        n = 4
        e = int(round(n * (n - 1.0) / 2.0))
        for i in xrange(e + 1):
            for edges in kbits(e, i):
                logging.info("%d 1s in %d edges." % (i, e))
                f = Graph.from_edges(edges)
                fft = super(Graph, f).fft().matrix
                f = Graph.from_edges(edges)
                graph_fft = f.fft().matrix
                for fft_rho, gfft_rho in zip(fft, graph_fft):
                    assert_array_almost_equal(fft_rho, gfft_rho.todense())

    @skip
    def test_large_fft(self):
        f1 = Graph.erdos_renyi(n=15)
        # fft = super(Graph, f1).fft().matrix
        graph_fft = f1.sparse_fft().matrix

    def test_sparse_fft(self):
        n = 4
        e = int(round(n * (n - 1.0) / 2.0))
        for i in xrange(e + 1):
            for edges in kbits(e, i):
                logging.info("%d 1s in %d edges." % (i, e))
                f = Graph.from_edges(edges)
                fft = super(Graph, f).fft().matrix
                f = Graph.from_edges(edges)
                graph_fft = f.sparse_fft().matrix
                for fft_rho, gfft_rho in zip(fft, graph_fft):
                    assert_array_almost_equal(fft_rho, gfft_rho.todense())

    def test_power_spectrum(self):
        iso_classes = self.calculate_invariant("power_spectrum", sparse=True)
        assert_greater_equal(len(iso_classes), 10)
        assert_less_equal(len(iso_classes), 11)

    def test_power_spectrum_invariant(self):
        f1 = Graph.erdos_renyi(7)
        f2 = f1.apply_permutation(f1.sn.random())
        for fm1, fm2 in zip(f1.power_spectrum(), f2.power_spectrum()):
            assert_array_almost_equal(fm1.todense(), fm2.todense())

    @skip
    def test_power_spectrum_s5(self):
        iso_classes = self.calculate_invariant("power_spectrum", n=5)
        assert_greater_equal(len(iso_classes), 26)
        assert_less_equal(len(iso_classes), 34)

    @skip
    def test_bispectrum_invariance(self):
        n = 5
        e = int(round(n * (n - 1.0) / 2.0))
        for i in xrange(e + 1):
            for edges in kbits(e, i):
                logging.info("%d 1s in %d edges." % (i, e))
                G = Graph.from_edges(edges)
                bispectrum = G.bispectrum(idx=(0, 1))
                for g in G.sn.generate():
                    Gdash = G.apply_permutation(g)
                    bispectrum_ = Gdash.bispectrum(idx=(0, 1))
                    for b_i in xrange(len(bispectrum)):
                        assert_array_almost_equal(bispectrum[b_i], bispectrum_[b_i])

    def test_large_bispectrum(self):
        f1 = Graph.erdos_renyi(n=6)
        p = f1.sn.random()
        f2 = f1.apply_permutation(p)
        for m1, m2 in zip(f1.bispectrum(), f2.bispectrum()):
            assert_array_almost_equal(m1.todense(), m2.todense())

    @skip
    def test_bispectrum(self):
        iso_classes = self.calculate_invariant("bispectrum", sparse=True)
        assert_equal(len(iso_classes), 11)
        iso_classes = self.calculate_invariant("bispectrum", n=5, idx=(0, 1))
        assert_equal(len(iso_classes), 34)
        iso_classes = self.calculate_invariant("bispectrum", n=6, idx=(0, 1, 2))
        assert_equal(len(iso_classes), 156)