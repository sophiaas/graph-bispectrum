# bispec3d.py

import numpy as np

def bispectrum_3d(f, n=None, num_randos=5, idxs=None):
    """
    compute bispectrum of f: {0,1,...,n-1}^3 = (Z/nZ)^3 -> C

    n:        positive integer
    f:        complex vector of dimension 3 with side length n
    num_randos: number of random pairs of triplets i1, i2, i3, j1, j2, j3
    idxs:     input indices to use
"""
    assert f.ndim == 3, 'input has to be 3-dimensional!'
    if n is None: n=len(f)
    count = 0
    B = []

    F = np.fft.fftn(f)

    if idxs is None:
        idxs = []
        for c in range(num_randos):
            i1 = np.random.randint(n)
            i2 = np.random.randint(n)
            i3 = np.random.randint(n)
            j1 = np.random.randint(n)
            j2 = np.random.randint(n)
            j3 = np.random.randint(n)
            idxs.append((i1, i2, i3, j1, j2, j3))
            F1 = F[i1, i2, i3]
            F2 = F[j1, j2, j3]
            F3 = F[(i1 + j1)%n, (i2 + j2)%n, (i3 + j3)%n]
            B.append(F1 * F2 * F3.conjugate())
        return idxs, B
    else:
        for idx in idxs:
            i1, i2, i3, j1, j2, j3 = idx
            F1 = F[i1, i2, i3]
            F2 = F[j1, j2, j3]
            F3 = F[(i1 + j1)%n, (i2 + j2)%n, (i3 + j3)%n]
            B.append(F1 * F2 * F3.conjugate())
        return B

# tiny test

f = np.ones((3,3,3))
f[0,0,0] = 0
f1 = np.ones((3,3,3))
f1[-1,-1,-1] = 0

idxs, bisp = bispectrum_3d(f, n=3)
bisp2 = bispectrum_3d(f1, n=3, idxs=idxs)

# succeeds:
numpy.testing.assert_array_almost_equal(bisp, bisp2)

# fails:
# f1[1,1,0] = 5
# bisp2 = bispectrum_3d(f1, n=3, idxs=idxs)
# numpy.testing.assert_array_almost_equal(bisp, bisp2)
