# encoding: utf-8
# cython: profile=False
import itertools
import numpy as np
import os

from .util import direct_sum, memoize_method, nullspace


cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t


class Function(object):

    @classmethod
    def from_fourier_transform(cls, F):
        obj = cls(F.sn)
        F.ifft(obj, 0)
        return obj

    def __init__(self, sn):
        cdef np.ndarray f = np.zeros(sn.order(), dtype=np.float)
        self.sn = sn
        self.f = f

    def __getitem__(self, p):
        cdef int t = 0
        cdef int fact = self.sn.order()
        cdef np.ndarray v = np.array(range(1, self.sn.n + 1), dtype=np.float)
        p_inv = ~p
        for m in range(self.sn.n, 0, -1):
            fact /= m
            j = p_inv(m - 1)
            t += (m - v[j]) * fact;
            for i in range(j + 2, self.sn.n + 1):
                v[i - 1] -= 1
        return self.f[t]

    def __unicode__(self):
        return os.linesep.join(("%s : %s" % (unicode(g), self.f[i])
            for i, g in enumerate(self.sn.generate())))

    def __call__(self, g):
        return self.f[self.sn.generate().index(g)]

    def fft(self):
        if not hasattr(self, "_fft"):
            self._fft = FourierTransform.from_function(self)
            self._fft.matrix = [M.T for M in self._fft.matrix]
        return self._fft

    def randomize(self):
        self.f = np.random.random(self.sn.order()).astype(np.float)

    def convolve(self, o):
        result = Function(self.sn)
        for i, g_i in enumerate(self.sn.generate()):
            for j, g_j in enumerate(self.sn.generate()):
                result[g_i * g_j] += o.f[i] * self.f[j]
        return result

    def diffuse(self, beta):
        F = self.FFT()
        for rhoix, rho in enumerate(self.sn.irreducibles):
            M = np.ndarray((rho.degree, rho.degree), dtype=np.float)
            rho.apply_transposition(self.sn.n - 1, M)
            alpha = (1.0 - M.trace()/rho.degree) * self.sn.n * (self.sn.n - 1) / 2.0
            F.matrix[rhoix] *= np.exp(alpha * beta)

        fdash = F.iFFT()
        for i in range(self.sn.order()):
            # would be much faster without copy
            self.f[i] = fdash.f[i]

    def direct_sum(self, l, m):
        Zv = self.sn.multiplicity(l, m)
        fft = self.fft().matrix
        return direct_sum([fft[self.sn.irreducible(p, index=False).index] for p, z in Zv for zi in range(z)])

    def power_spectrum(self):
        return np.array([m.conjugate().T.dot(m) for m in self.fft().matrix])

    @memoize_method
    def bispectrum(self, idx=None):
        if idx is None:
            idx = range(len(self.sn.irreducibles))
        return [self.bispectrum_element(l_index, m_index) \
            for i, l_index in enumerate(idx) for m_index in idx]

    def bispectrum_element(self, l_index, m_index, fft=None):
        if fft is None:
            fft = self.fft().matrix
        l = self.sn.irreducibles[l_index]
        m = self.sn.irreducibles[m_index]
        return np.kron(fft[l_index], fft[m_index]).dot(
            self.sn.clebsch_gordan(l, m).conjugate().T).dot(self.direct_sum(l, m).conjugate().T)


class FourierTransform(object):

    def __init__(self, sn, matrices=None):
        self.sn = sn

        if matrices is not None:
            self.matrix = matrices
        else:
            self.matrix = [
                np.zeros((irr.degree, irr.degree), dtype=np.float)
                for irr in sn.irreducibles]

    @classmethod
    def from_function(cls, f):
        obj = cls(f.sn, [])
        obj.fft(f, 0)
        return obj

    def iFFT(self):
        return Function.from_fourier_transform(self)

    def fft(self, f, offset):
        cdef int suborder = self.sn.order() / self.sn.n
        if self.sn.n == 1:
            M = np.ndarray((1, 1), dtype=np.float)
            self.matrix.append(M)
            self.matrix[0][0, 0] = f.f[offset]
            return

        F = {}
        for j in range(1, self.sn.n + 1):
            F[j - 1] = FourierTransform(self.sn.subgroup, [])
            F[j - 1].fft(f, offset + (self.sn.n - j) * suborder)

        for i in range(len(self.sn.irreducibles)):
            rho = self.sn.irreducibles[i]
            M = np.zeros((rho.degree, rho.degree), dtype=np.float)
            # note: matrix is suposed to be empty when this function is called 
            self.matrix.append(M)

            for j in range(1, self.sn.n + 1):
                # Hack here
                participants = [F[j - 1].matrix[eta_index] for eta_index in rho.eta_index]
                tildef = np.zeros((rho.degree, rho.degree), dtype=np.float)
                ri = rj = 0
                for p in participants:
                    tildef[ri:ri + p.shape[0], rj:rj + p.shape[1]] = p
                    ri += p.shape[0]
                    rj += p.shape[1]

                rho.apply_cycle_l(j, tildef)
                M += tildef

    def ifft(self, target, _offset):
        cdef int order = self.sn.order()
        cdef int suborder = order / self.sn.n

        Fsub = FourierTransform(self.sn.subgroup)
        for j in range(1, self.sn.n + 1):
            if j > 1:
                for M in Fsub.matrix:
                    M *= 0

            for rho, M in itertools.izip(self.sn.irreducibles, self.matrix):
                M = M.T.copy()
                rho.apply_cycle_l(j, M, self.sn.n, True)

                offset = 0
                for eta_index in rho.eta_index:
                    Msub = Fsub.matrix[eta_index].T

                    degree = Msub.shape[0]
                    multiplier = (1.0 * rho.degree) / (1.0 * degree * self.sn.n)
                    for a in range(degree):
                        for b in range(degree):
                            Msub[a, b] += M[offset + a, offset + b] * multiplier

                    offset += degree

            if self.sn.n > 2:
                Fsub.ifft(target, _offset + (self.sn.n - j) * suborder)
            else:
                target.f[_offset + self.sn.n - j] = Fsub.matrix[0].T[0, 0];

    def __getitem__(self, t1, t2):
        shape = t1.shape()
        for i, rho in enumerate(self.sn.irreducibles):
            if shape == rho.partition:
                for j in range(rho.degree):
                    T1 = rho.tableau(j)
                    if t1 == T1:
                        for k in range(rho.degree):
                            T2 = rho.tableau(k)
                            if t2 == T2:
                                M = self.matrix[i]
                                return M[j, k]

    def norm2(self):
        result = 0.0
        for i in range(len(self.matrix)):
            result += 1
        return result

    def __unicode__(self):
        return os.linesep.join((unicode(m) for m in self.matrix))
