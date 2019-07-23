# Charles Garfinkle, Nov 30, 2018 
#
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import minimize

def TwoDBispectrumAllCoeffs(f,m,n):

    B = np.zeros((m,n,m,n))
    F = np.fft.fft2(f)

    for i in range(1, m+1):
        for j in range(1, n+1):
            for k in range(1, m+1):
                for l in range(1, n+1):
                    sumik = np.mod(i + k - 2, m)+1
                    sumjl = np.mod(j + l - 2, n)+1
                    B[i-1,j-1,k-1,l-1] = np.conj(F[i-1, j-1] * F[k-1,l-1]) * F[sumik-1, sumjl-1]

    return B

def sub2ind(array_shape, rows, cols):
    return np.atleast_2d(cols * array_shape[0] + rows).T

def ind2sub(array_shape, idx):
    return (idx % array_shape[0], idx // array_shape[1])

def bsp_getFid(N):
    # bsp_getFid.m
    
    y, x = np.meshgrid(0, range(N[0]//2+1)) # [y,x] = meshgrid(0,0:(N(1)/2));
    ids = np.hstack((x, y))
    for i in range(1, N[1]//2):
        y, x = np.meshgrid(i, range(N[0]))
        ids = np.vstack( (ids, np.hstack((x, y))) )
    y, x = np.meshgrid(N[1]//2, range(N[0]//2+1))
    ids = np.vstack( (ids, np.hstack((x, y))) )
    ids = sub2ind(N, ids[:,0], ids[:,1])
    return ids.T

def bsp_getBid(ids, N):
    k11 = np.arange(0, N[0]) # k11 = 0:(N(2)-1);
    k12 = np.arange(0, N[1]) # k12 = 0:(N(1)-1);

    [y, x] = np.meshgrid(k11,k12)
    K1v = np.array([x.ravel(order='F'),y.ravel(order='F')]).T
    K1id = sub2ind(N, K1v[:,0], K1v[:,1])

    Am = np.zeros((max(N)**3, N[0]*N[1]//2+2)) # TODO: make sparse
    Ap = np.zeros((max(N)**3, N[0]*N[1]//2+2)) # TODO: make sparse
    bid = np.zeros((max(N)**3,3)).astype(int)
    c=0
    for i in range(K1v.shape[0]):
        if np.any(ids == K1id[i]) and (K1id[i] > 0): # if any(id==K1id(i)) && (K1id(i)>1)
            y, x = np.meshgrid( range(K1v[i,0], N[1]-K1v[i,0]), range(N[0]-K1v[i,1]) )
            K2id = sub2ind(N, x.ravel(order='F'), y.ravel(order='F'))
            k1pk2 = K1v[i,:] + np.array([x.ravel(order='F'),y.ravel(order='F')]).T

            # ignore k2=0
            valid_id = K2id > 0
            k1pk2 = k1pk2[valid_id.ravel(), :]
            K2id  = K2id[valid_id.ravel(), :]

            # Ignore k1+k2 outside of range...
            valid_id = (k1pk2[:,0] < N[0]) & (k1pk2[:,1] < N[1])
            k1pk2 = k1pk2[valid_id,:]
            K2id  = K2id[valid_id]
            K1K2id = sub2ind(N, k1pk2[:,0], k1pk2[:,1])

            if K2id.size != 0: 
                tmp1 = K2id - ids == 0 # TODO : make sparse
                tmp2 = K1K2id - ids == 0 # TODO : make sparse
                valid_id = np.sum(tmp1, 1) * np.sum(tmp2, 1) > 0 # use np.multiply if sparse matrices

                cid = np.arange(c, c + sum(valid_id))
                Am[cid, (ids==K1id[i]).ravel()] = 1
                Am[cid,:] = Am[cid,:] + tmp1[valid_id,:]
                Am[cid,:] = Am[cid,:] + tmp2[valid_id,:]

                Ap[cid, (ids==K1id[i]).ravel()] = 1
                Ap[cid,:] = Ap[cid,:] + tmp1[valid_id,:]
                Ap[cid,:] = Ap[cid,:] - tmp2[valid_id,:]
                
                bid[cid,0] = K1id[i]
                bid[cid, 1] = K2id[valid_id].ravel() 
                bid[cid,2] = K1K2id[valid_id].ravel()

                c += cid.size

    Am  = Am[range(c), :]
    Ap  = Ap[range(c), :]
    bid = bid[range(c), :]

    return (bid, Am, Ap)

def objBspPhaseLoss(fphi, bphi, A, w):
    f  = np.sum( ( ( (bphi - np.dot(A, fphi) + np.pi) % (2 * np.pi) - np.pi) / w) ** 2) / 2 # f  = sum(((mod(bphi-A*fphi+pi,2*pi)-pi)./w).^2)/2;
    return f

def DobjBspPhaseLoss(fphi, bphi, A, w):
    df = - np.dot(A.T, ((( (bphi - np.dot(A, fphi) + np.pi) % (2 * np.pi) - np.pi) / w) / w)) # df = -A'*(((mod(bphi-A*fphi+pi,2*pi)-pi)./w)./w);
    return df

def fconjsym(f):
    nx, ny = f.shape
    r , c = np.nonzero(np.logical_not(np.isfinite(f)))
    for i in range(r.size):
        f[r[i],c[i]] = np.conj( f[((nx-r[i]) % nx), ((ny-c[i]) % ny)] )
    return f

def bsp_leastsquares_recon_fromFullBSP(B):

    nx, ny, tmp, tmp = B.shape
    nsamp = 1

    print('Setting up systems of equations...')
    # Non-redundant set of Fourier coefficients to use...
    ids = bsp_getFid([nx, ny])
    # Non-redundant set of Bispectrum coefficients (and linear systems Am and Ap for the Fourier magnitude and phase)...
    bid, Am, Ap = bsp_getBid(ids, [nx, ny]) # HERE!! TESTED UP TO HERE
    k1x, k1y = np.unravel_index(bid[:,0], [nx, ny], order='F') # [k1x, k1y] = ind2sub([nx, ny], bid[:,0])
    k2x, k2y = np.unravel_index(bid[:,1], [nx, ny], order='F') # [k2x, k2y] = ind2sub([nx, ny], bid[:,1])
    #bbid = np.ravel_multi_index([k1x, k1y, k2x, k2y], [nx, ny, nx, ny], order='F')
    #b = B[bbid]
    b = np.atleast_2d(B[k1x, k1y, k2x, k2y]).T

    # Least-squares for the amplitudes...
    print('Least-squares for amplitudes...')
    midx = np.arange(1, Am.shape[1]);  # exclude F(0,0)

    # least-squares on the mean bispectrum...
    if nsamp > 2:
        bmu = np.sum(b, axis=1) / nsamp
    else:
        bmu = b
    xhat = np.linalg.lstsq(Am[:,midx], np.log(np.abs(bmu)))[0]

    mhat = np.ones(nx * ny) * np.nan
    mhat[ids.ravel()[midx]] = np.exp(xhat).ravel() # estimated magnitude
    mhat[0] = 0
    mhat = mhat.reshape((nx, ny), order='F')

    # Least squares only defines phase up to a multiple of 2*pi...
    # I'm using phase unwrapping optimization instead...
    print('Phase unwrapping optimization...')
    pidx_rem = np.ravel_multi_index( ([0, 0, nx//2], [0, ny//2, 0]), [nx, ny], order='F'); # exclude F(0,0), F(0,N/2), F(N/2,0)
    pidx = np.ones_like(ids).astype(bool)
    for j in range(pidx_rem.size):
        pidx[ids == pidx_rem[j]] = False
    ab = np.angle(np.sum(b, axis=1))  # circular mean
    r = np.abs(np.sum(np.exp(1j*np.angle(b)), axis=1)) / nsamp
    w = np.sqrt(2*(1-r))                        # angular deviation

    if nsamp==1: w = w*0 + 1

    phat0 = np.linalg.lstsq(Ap[:,pidx.ravel()], ab)[0]

    # Random resarts
    fphats = []
    phats = []
    rrnum = 100
    rriter = 50
    print('Random restarts...')
    pidx = pidx.ravel()
    f = lambda x: objBspPhaseLoss(x, ab, Ap[:,pidx], w)
    df = lambda x: DobjBspPhaseLoss(x, ab, Ap[:,pidx], w)
    for i in range(rrnum): 
        print("%03d/%03d...\n" % (i, rrnum) )
        phat0 = np.random.rand(np.sum(pidx), 1) * 6 * np.pi - 3 * np.pi
        res = minimize(f, phat0, jac = df, options = {'maxiter': rriter}) #, options = {'maxiter': rriter}) # phat, fphat = minimize_verbose1(phat0, 'objBspPhaseLoss', rriter, ab, Ap(:,pidx), w);
        phats.append(res.x)
        fphats.append(f(res.x))

    tmp = np.min(fphats)
    i = fphats.index(tmp) 
    phat = phats[i]
    res = minimize(f, phat, jac = df, options = {'maxiter': 1000})  # make sure it converges
    phat = res.x
    
    ahat = np.zeros(nx * ny) * np.nan
    ahat[ids.ravel()[pidx]] = phat
    ahat[pidx_rem] = 0
    ahat = ahat.reshape((nx,ny), order='F')

    Fhat = mhat * (np.cos(ahat)+1j * np.sin(ahat))
    Fhat = fconjsym(Fhat); # Conjugate symmetry to fill in the rest of the fft2...

    imhat = np.real(np.fft.ifft2(Fhat))

    return imhat, Fhat, ids, bid, Am, Ap, b, fphats


if __name__ == '__main__':
    # bsp_recon_test_v2.m
    import scipy.io
    import matplotlib.pyplot as pp
    
    # load image
    f = scipy.io.loadmat('./phantom.mat')['f']
    B = TwoDBispectrumAllCoeffs(f, f.shape[0], f.shape[1]) # take bispectrum
    imhat, Fhat, ids, bid, Am, Ap, b, fpopt = bsp_leastsquares_recon_fromFullBSP(B) # invert bispectrum

    # plot original and reconstruction
    fig, axes = pp.subplots(1,3)
    axes[0].imshow(f) # orig
    axes[0].set_xlabel('orig')
    axes[1].imshow(imhat) # recon
    axes[1].set_xlabel('recon')
    axes[2].imshow(np.fft.fftshift(np.abs(Fhat))) # FFT
    axes[2].set_xlabel('FFT')
    pp.show()