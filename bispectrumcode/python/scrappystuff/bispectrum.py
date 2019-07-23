"""
one and two dimensional bispectrum
"""
import numpy as np

def bispectrum_1d(f,n=None):
    """
    compute bispectrum of f: {0,1,...,n-1} = Z/nZ -> C

    n:        positive integer
    f:        complex vector of length n
     
    Computes a 1x(n+1) vector B = [B(0,0), B(0,1), B(1,1), ..., B(n-1,1)]
    in which B(k1,k2) = f^(k1)* f^(k2)* f^(k1+k2)

    Here f^ is the Fourier Transform of f and * means complex conjugation
    (starting at 0 not 1 like matlab)

    Christopher Hillar 2008
    """
    
    assert f.ndim == 1, 'input has to be one-dimensional!'
    if n is None: n=len(f)

    B = np.zeros(n+1,dtype=np.complex)
    F = np.fft.fft(f,n)

    B[0] = np.conj(F[0]*F[0])*F[0]
    B[1:-1] = np.conj(F[:-1]*F[1])*F[1:]
    B[n] = np.conj(F[n-1]*F[1])*F[0]

    return B


def ibispectrum_1d_fft(B):
    """
    compute FFT inverse of bispectrum of f: Z/nZ -> C
    
    B: complex vector of length n+1
    
    Computes a 1x(n) vector f^ whose fourier inverse, f, has truncated bispectrum B
    
    Christopher Hillar 2008
    """

    assert B.ndim == 1, 'input has to be one-dimensional!'
    n=len(B)-1

    # recover Fourier transform of f
    F = np.zeros(n,dtype=np.complex)
    F[0] = np.abs(B[0])**(1.0/3)*np.exp(-1j*np.angle(B[0]))

    # first need to recover F[1] - up to translation
    if np.mod(n,2):
        Prod = np.prod(B[2:n+1:2]/np.conj(B[3:n+1:2]))
    else:
        Prod = B[n]*np.conj(F[0])*np.prod(B[2:n:2]/np.conj(B[3:n:2]))

    F1norm = np.sqrt(B[1]/np.conj(F[0]))
    F1arg = -(1.0/n)*np.angle(Prod)
    F[1] = F1norm*np.exp(1j*F1arg) #for different shifts: *exp(2*pi*i*k/n);

    # recover the rest of the Fourier transform of f
    for i in xrange(2,n):
        F[i] = B[i]/np.conj(F[1]*F[i-1])

    return F


def ibispectrum_1d(B):
    """
    invOneDBispectrum.m - compute inverse of bispectrum of f: Z/nZ -> C

    B:        complex vector of length n+1

    Computes a 1x(n) vector f whose bispectrum is B

    Christopher Hillar 2008
    """

    F = ibispectrum_1d_fft(B)
    f = np.fft.fftshift(np.fft.ifft(F))

    return f


def test_bispectrum_1d_consistency():
    for n in [64,65]:
        f = np.random.rand(n)
        B = bispectrum_1d(f)
        g = ibispectrum_1d(B)
        yield np.testing.assert_almost_equal, np.roll(f,-f.argmin()), np.roll(g,-g.argmin())


def plot_test_bispectrum_1d(display=False):
    f = np.zeros(64)
    f[10:50] = -np.arange(40)*np.arange(-39,1)+100*np.sin(np.arange(11,51))

    B = bispectrum_1d(f)
    g = ibispectrum_1d(B)

    np.testing.assert_almost_equal(np.roll(f,-f.argmin()), np.roll(g,-g.argmin()))

    # the first plot is the original function
    # the second is the reconstruction
    # recall: reconstruction is only possible up to translation
    if display:
        import matplotlib.pyplot as plt
        x = np.arange(len(f))
        plt.clf()
        plt.plot(x,np.roll(f,-f.argmin()))
        plt.plot(x,np.roll(g,-g.argmin()))
        plt.show()


def bispectrum_2d(f,s=None):
    """
    compute a truncated 2D bispectrum of f: {0..m-1} x {0..n-1} = Z/mZ x Z/nZ -> C
    
    s = (m,n):  positive integers
    f:          complex matrix of size m x n
    
    Computes an mn+2 vector B = [B(0,0,0,0),B(i,0,1,0),B(0,j,0,1),B(0,j,i,0)]
                                          i = 0..m-1  j = 0..n-1  j,i = 1..n-1,m-1
    in which B(k1,k2,k3,k4) = f^(k1,k2)* f^(k3,k4)* f^(k1+k3,k2+k4)
    
    Here f^ is the 2D Fourier Transform of f and * means complex conjugation
    (starting at 0 not 1 like matlab)
    
    Christopher Hillar 2008
    """

    assert f.ndim == 2, 'input has to be two-dimensional!'
    if s is None: s=f.shape

    m,n = s
    B = np.zeros(m*n+2,np.complex)
    F = np.fft.fft2(f,s)
    B[0] = np.conj(F[0,0]*F[0,0])*F[0,0]
    B[1:m] = np.conj(F[:-1,0]*F[1,0])*F[1:,0]
    B[m] = np.conj(F[m-1,0]*F[1,0])*F[0,0]
    B[m+1:m+n] = np.conj(F[0,:-1]*F[0,1])*F[0,1:]
    B[m+n] = np.conj(F[0,n-1]*F[0,1])*F[0,0]

    # itercount = 1
    # for j in xrange(n-1):
    #     for i in xrange(m-1):
    #         B[m+n+itercount] = np.conj(F[0,j+1]*F[i+1,0])*F[i+1,j+1]
    #         itercount += 1
    B[m+n+1:] = (np.conj(F[0:1,1:]*F[1:,0:1]).T*F[1:,1:].T).flatten()

    return B
 

def ibispectrum_2d(B,s=None):
    """
    compute inverse of bispectrum of f: Z/mZ x Z/nZ -> C
    
    s = (m,n): positive integers
    B:         complex vector of length mn+2
    
    Computes a m x n matrix f whose truncated bispectrum is B
    The inversion is only up to a 2D Translation
    
    Christopher Hillar 2008
    """

    assert B.ndim == 1, 'input has to be one-dimensional!'
    if s is None: s=(np.int(np.sqrt(len(B)-2)),)*2

    m,n = s

    # first find Fourier transform of f
    F = np.zeros((m,n),dtype=np.complex)

    # Create 1D vectors to recover f^(i,0)  i = 1..m-1
    #                              f^(0,j)  j = 1..n-1     
    B0dcol = B[:m+1]
    F0dcol = ibispectrum_1d_fft(B0dcol)
    B0drow = B[[0,]+range(m+1,m+n+1)]
    F0drow = ibispectrum_1d_fft(B0drow)

    F[0,:n] = F0drow
    F[:m,0] = F0dcol

    # fill in the rest of the matrix for f
    # icount = 0
    # for j in xrange(n-1):
    #     for i in xrange(m-1):
    #         F[i+1,j+1] = B[m+n+1+icount]/np.conj(F[0,j+1]*F[i+1,0])
    #         icount += 1    
    F[1:,1:].T.flat = B[m+n+1:]/np.conj(F[0:1,1:]*F[1:,0:1]).T.flatten()

    # invert Fourier transform F to get back f
    f = np.fft.ifft2(F)

    return f


def test_bispectrum_2d_consistency():
    for n in [64,65]:
        f = np.random.rand(n,n)
        f = np.roll(np.roll(f,-f.min(1).argmin(),0),-f.min(0).argmin(),1)
        B = bispectrum_2d(f)
        g = ibispectrum_2d(B)
        g = np.roll(np.roll(g,-g.min(1).argmin(),0),-g.min(0).argmin(),1)

        yield np.testing.assert_almost_equal, f, g


def plot_test_bispectrum_2d(display=False):
    import scipy as sp

    im = sp.misc.imread('matlab/TwoDBispectrum/line.bmp').astype(np.double)
    bsp = bispectrum_2d(im)
    im2 = ibispectrum_2d(bsp).real

    np.testing.assert_almost_equal(im,np.roll(np.roll(im2,6,0),7,1))

    if display:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        plt.imshow(im,cmap=plt.cm.gray,interpolation='nearest')

        plt.subplot(122)
        plt.imshow(np.roll(np.roll(im2,6,0),7,1),cmap=plt.cm.gray,interpolation='nearest')
        plt.show()

if __name__ == '__main__':
    import nose
    # nose.runmodule(exit=False,argv=['-s','--pdb-failures','-v'])
    nose.runmodule(exit=False,argv=['-s'])

    # to plot the test example, use the following line:
    # plot_test_bispectrum_1d(True)

    # to plot the 2d examples of bispectrum, use the following line:
    # plot_test_bispectrum_2d(True)



