# Bispectrum code
#
# Uses recursive algorithm with memoization
#
# See: Sadler, Giannakis, J.Opt.Soc.Am.A, 1992
#
# Chris Hillar, May 2013
#

import numpy as np


def bispectrum(F, truncated=False, flatten=False):
    """
        Computes the 2D bispectrum for an M x N fourier transform

        F: numpy (M x N) or (M x N x T) numpy array
        (2nd case: a list of T Fourier transforms to compute Bispectrum of)

        OUTPUT: M x N x M x N or (M x N x M x N x T) numpy array

        if truncated is True then

        OUTPUT: (MN+2) numpy array

        F(i,j) = F*(M-i,N-j)

    """
    if len(F.shape) < 2:
        print("Need a 2D fourier transform")
        return None
    F = np.atleast_3d(F)
    M,N,T = F.shape

    if truncated is True: # compute smaller but sufficient num bispect coeffs
        B = np.zeros((M*N+2,T),np.complex)
        B[0,:] = np.conj(F[0,0,:]*F[0,0,:])*F[0,0,:]
        B[1:M,:] = np.conj(F[:-1,0,:]*F[1,0,:])*F[1:,0,:]
        B[M,:] = np.conj(F[M-1,0,:]*F[1,0,:])*F[0,0,:]
        B[M+1:M+N,:] = np.conj(F[0,:-1,:]*F[0,1,:])*F[0,1:,:]
        B[M+N,:] = np.conj(F[0,N-1,:]*F[0,1,:])*F[0,0,:]
        B[M+N+1:,:] = (np.conj(F[0:1,1:,:]*F[1:,0:1,:]).T*F[1:,1:,:].T).flatten()[:,None]
        return B

    B = np.zeros((F.shape[0],F.shape[1],F.shape[0],F.shape[1],F.shape[2]),dtype=np.complex128)
    for i in range(M):     # TODO: loop could be made faster with numpy tricks?
        for j in range(N):
            for k in range(M):
                for l in range(N):
                    B[i,j,k,l,:] = F[i,j,:]*F[k,l,:]*F[(i+k)%M,(j+l)%N,:].conj()
    if T == 1: return B[:,:,:,:,0]
    if flatten is True:
        return B.reshape(M*N*M*N,T)
    return B

def inv_bispectrum(B, denoise=False, shape=None, the_mean=0.,freq_tol=0.):
    """
        Computes the inverse of the 2D M x N (x T) bispectrum
        See: Sadler, Giannakis, J.Opt.Soc.Am.A, 1992

        TODO: use redundancy

        B: bispectrum of a 2D fourier transform F
           if B has shape (D,T) then need to unflatten B with shape=(M,N)
           otherwise, B should have shape M,N,M,N or (M,N,M,N,T)

        shape: (M,N) if B is flattened

        OUTPUT: M x N (x T) matrix F  (if denoise is False)

        if denoise is True: then F(0,0) = 0 is assumed and

        OUTPUT: M x N matrix F
    """
    if shape is not None:  # B is flattend bispectrum
        M,N = shape
        B = np.atleast_2d(B)
        T = B.shape[1]
        B = B.reshape(M,N,M,N,T)
    if len(B.shape) == 4:
        M,N = B.shape[0],B.shape[1]
        B = B.reshape(M,N,M,N,1)
    M,N,T = B.shape[0],B.shape[1],B.shape[4]
    if denoise is True:
        F = np.zeros((M,N),dtype=np.complex128)
        B = B.mean(4)   # average T bispects together
    else:
        F = np.zeros((M,N,T),dtype=np.complex128)

    F_hash = dict()  # for memoization
    F[0,0] = the_mean

    for i in range(M):
        for j in range(N):
            if denoise is True:
                F[i,j] = F_recursive(i,j,F_hash,B,denoise=denoise,freq_tol=freq_tol)
            else:
                F[i,j,:] = F_recursive(i,j,F_hash,B)
    if denoise is True:     # adjust back to zero mean & phase correction

        # TODO: make faster
        for k in range(M):
            for l in range(N):
                F[k,l] = F[k,l]*(np.cos(2.*(k/M)*np.angle(F[M//2,0])+2.*(l/N)*np.angle(F[0,N//2]))-(0.+1.j)*np.sin(2.*(k/M)*np.angle(F[M//2,0])+2.*(l/N)*np.angle(F[0,N//2])))

        # should try and recover other F(k1,k2) = F*(M-k1, N-k2)
        for k in range(M//2,M):
            for l in range(N//2,N):
                F[k,l] = (F[k,l]+F[M-k,N-l].conjugate())/2

        F[0,0] = the_mean   # put back mean



        # G_hash = dict()
        # G_hash[1,0] = F[23,0].conjugate()
        # G_hash[0,1] = F[0,23].conjugate()
        #
        # # one more run through
        # for i in range(M):
        #     for j in range(N):
        #         F[i,j] = F_recursive(i,j,G_hash,B,denoise=denoise,freq_tol=freq_tol)
        #
        #
        # for k in range(M):
        #     for l in range(N):
        #         F[k,l] = F[k,l]*(np.cos(2.*(k/M)*np.angle(F[M/2,0])+2.*(l/N)*np.angle(F[0,N/2]))
        #                     -(0.+1.j)*np.sin(2.*(k/M)*np.angle(F[M/2,0])+2.*(l/N)*np.angle(F[0,N/2])))
        #

        return F
    if T==1: return F[:,:,0]
    return F

def F_recursive(i, j, F_hash, B, denoise=False, freq_tol=0.):
    """
        Recursive helper function for inverting bispectrum

        freq_tol: for denoising when to invert certain fourier components (when > freq_tol)
        TODO: cleanup [denoise is True] case clutter
    """
    if F_hash.get((i,j),0) is not 0:  # if F(i,j) memoized
        return F_hash.get((i,j))
    if (i,j)==(0,0): # base case
        if denoise is True:  # if original images zero meaned and want to denoise translations
            Fij_conj = 0.  # (1.+0.j) # np.ones(T,dtype=np.complex128)
#            Fij_conj = (1.+0.j) # np.ones(T,dtype=np.complex128)
        else:
            T = B.shape[4]
            Fij_conj = np.zeros(T,dtype=np.complex128)
            F0_real = np.abs(B[0,0,0,0,:])**(1./3)
            F0_angle = np.angle(B[0,0,0,0,:])
            Fij_conj.real = F0_real*np.cos(F0_angle)
            Fij_conj.imag = F0_real*np.sin(-F0_angle)
    elif (i,j)==(0,1): # base case
        if denoise is True:  # if original images zero meaned and want to denoise translations
            F0_real = (np.abs(B[0,0,0,1]/1.))**(1./2)
            Fij_conj = F0_real
        else:
            F0_real = (np.abs(B[0,0,0,1,:]/F_hash[0,0]))**(1./2)
            Fij_conj = F0_real
    elif (i,j)==(1,0): # base case
        if denoise is True:  # if original images zero meaned and want to denoise translations
            F0_real = (np.abs(B[0,0,1,0]/1.))**(1./2)
            Fij_conj = F0_real
        else:
            F0_real = (np.abs(B[0,0,1,0,:]/F_hash[0,0]))**(1./2)
            Fij_conj = F0_real
    elif j > 1:
        if denoise is True:  # if original images zero meaned and want to denoise translations
            Fij_conj_sum = 0.
            count = 0.
            for k in range(i+1):
                for l in range(j+1):
                    if (k,l) != (0,0) and (k,l) != (i,j):  # only average over those freq which are nonzero
                        temp = F_recursive(k,l,F_hash,B,denoise,freq_tol)*F_recursive(i-k,j-l,F_hash,B,denoise,freq_tol)
                        if np.abs(temp) > freq_tol:
                            Fij_conj_sum += B[i-k,j-l,k,l]/temp
#                            print B[i-k,j-l,k,l]/temp
                            count += 1
                        else:
                            pass #print "(k,l) = (%d,%d) has too 0 F(k,l) %1.3f" % (k,l,temp)
            Fij_conj = Fij_conj_sum/count
        else:
            Fij_conj = B[i,j-1,0,1,:]/F_recursive(i,j-1,F_hash,B)/F_recursive(0,1,F_hash,B)
    else:
        if denoise is True:  # if original images zero meaned and want to denoise translations
            Fij_conj_sum = 0.
            count = 0.
            for k in range(i+1):
                for l in range(j+1):
                    if (k,l) != (0,0) and (k,l) != (i,j):  # only average over those freq which are nonzero
                        temp = F_recursive(k,l,F_hash,B,denoise,freq_tol)*F_recursive(i-k,j-l,F_hash,B,denoise,freq_tol)
                        if np.abs(temp) > freq_tol:
                            Fij_conj_sum += B[i-k,j-l,k,l]/temp
#                            print B[i-k,j-l,k,l]/temp
                            count += 1
                        else:
                             pass # print "(k,l) = (%d,%d) has too 0 F(k,l) %1.3f" % (k,l,temp)
            Fij_conj = Fij_conj_sum/count
        else:
            Fij_conj = B[i-1,j,0,1,:]/F_recursive(i-1,j,F_hash,B,denoise)/F_recursive(1,0,F_hash,B,denoise)
    #import pdb; pdb.set_trace()
    F_hash[i,j] = np.conj(Fij_conj)
    return F_hash[i,j]


#####################
# examples

# bispectrum
F = np.array([[1,2],[3,4]],dtype=np.complex128)
# print bispectrum(F,truncated=True)
G = np.zeros((2,2,2),dtype=np.complex128)
G[:,:,0] = F[:,:]
G[:,:,1] = 2*F[:,:]
# print bispectrum(G,truncated=True)

B = bispectrum(G,truncated=False,flatten=True)
# print B

B = bispectrum(G,truncated=False)
# print B

# inverse
F = np.array([[1,2],[3,4]],dtype=np.complex128)
#print F
B = bispectrum(F)
#print B
iB = inv_bispectrum(B)
#print iB  # should equal F up to shift
#print bispectrum(iB)


F = np.array([[(1+1.j),2],[3,(4-1.j)]],dtype=np.complex128)
#print F
B = bispectrum(F)
#print B
iB = inv_bispectrum(B)
#print iB  # should equal F up to shift
#print bispectrum(iB)


F = np.array([[1,2,3],[3,4,(1+.1j)]],dtype=np.complex128)
#print F
B = bispectrum(F)
#print B
iB = inv_bispectrum(B)
#print iB  # should equal F up to shift
#print bispectrum(iB)

F = np.array([[1,2],[3,4]],dtype=np.complex128)
G = np.array([[4,1],[2,2]],dtype=np.complex128)



def translate_img(img,x=0,y=0):
    """
    given an image, and an offset x,y returns a
    translated image up in y and right in x
    """
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            oldi = (i-x)%img.shape[0]
            oldj = (j-y)%img.shape[1]
            new_img[i,j] = img[oldi,oldj]
    return new_img
