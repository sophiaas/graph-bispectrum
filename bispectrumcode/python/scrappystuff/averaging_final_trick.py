"""
Superpixel Bispectrum Example
Translated noisy images of invader average in
bispectrum space to invert back to clean invader
Chris Hillar, Feb 2013
"""

import numpy as np
import scipy
import bispectrum_tricky
from bispectrum_tricky import *
from PIL import Image
import matplotlib.pyplot as plt
import h5py

h5 = h5py.File('spaceinvader.h5', 'r')
X = h5['/x'][:]
X = -h5['/y'][:]
h5.close()
X = (X-X.mean()) / X.std();
invader = -X.T

M = 500
sigma = .5
origs = np.zeros((invader.shape[0],invader.shape[1],M),dtype=np.complex128)
count = 0
print("Creating noisy images")
for i in range(M):
        origs[:,:,count] = invader.copy() + sigma*np.random.randn(invader.shape[0],invader.shape[1])
        count += 1
orig_mean = origs.mean(2)

count = 0
print("Creating random translations")
for i in range(M):
        new_im = translate_img(invader,x=np.random.random_integers(invader.shape[0]),y=np.random.random_integers(invader.shape[1]))
        origs[:,:,count] = new_im.copy() + sigma*np.random.randn(invader.shape[0],invader.shape[1])
        count += 1

orig_fft = np.zeros((invader.shape[0],invader.shape[1],M),dtype=np.complex128)
count = 0
for i in range(M):
    orig_fft[:,:,count] = np.fft.fft2(origs[:,:,count])
    count += 1

print("Computing bispectra")
B = bispectrum(orig_fft)


import scipy.ndimage

Bm = B.mean(4)
BmSm = np.zeros(Bm.shape,dtype=np.complex128)
# BmSm.real = scipy.ndimage.filters.gaussian_laplace(Bm.real,.001)
# BmSm.imag = scipy.ndimage.filters.gaussian_laplace(Bm.imag,.001)
BmSm = Bm


# BmSm.real = scipy.ndimage.filters.median_filter(Bm.real,size=1)
# BmSm.imag = scipy.ndimage.filters.median_filter(Bm.imag,size=1)
#


# BmSm.real = scipy.ndimage.filters.gaussian_filter(Bm.real,.8,mode='wrap')
# BmSm.imag = scipy.ndimage.filters.gaussian_filter(Bm.imag,.8,mode='wrap')
# BmSm = Bm

a,b = invader.shape[0],invader.shape[1]

plt.figure(1)
plt.title('Smoothed BiSpectrum directly')
plt.imshow(BmSm.real.reshape(a*b,a*b),interpolation='nearest',cmap=plt.cm.gray)


plt.figure(2)
plt.title('BiSpectrum directly')
plt.imshow(B.mean(4).reshape(a*b,a*b).real,interpolation='nearest',cmap=plt.cm.gray)



# from scipy import cluster
# a,b = invader.shape[0],invader.shape[1]
# data = B.reshape((a*b*a*b,M)).T
# dataW = scipy.cluster.vq.whiten(data)
# BW = dataW.reshape((a,b,a,b,M))

#print "Inverting whitened Bis with denoise=True"
#iB = inv_bispectrum(BW,denoise=True)  #,freq_tol=10**-1)

print("Inverting smoothed Bis with denoise=True")
iB = inv_bispectrum(BmSm,denoise=True)

# plt.figure(3)
# plt.imshow(np.abs(np.fft.ifft(iB)),interpolation='nearest',cmap=plt.cm.gray)

# smoothing
#

# np.smooth(x,window_len=1,window='hamming')



# print "Inverting with denoise=True"
# iB = inv_bispectrum(B,denoise=True)  #,freq_tol=10**-1)

orig_fft[:,:,0][1,0]
orig_fft[:,:,0][0,1]
iB[0,23]
iB[23,0]




# a,b = invader.shape[0],invader.shape[1]
# Bm = B.mean(4)
# plt.figure(1)
# plt.title('inverse BiSpectrum directly')
# plt.imshow(inv_bispectrum(Bm,denoise=True).real,interpolation='nearest',cmap=plt.cm.gray)




plt.figure(0)
plt.clf()

plt.subplot(2,3,1)
plt.imshow(origs[:,:,0].real,interpolation='nearest',cmap=plt.cm.gray)
plt.title('Noisy Original (one example)')
plt.subplot(2,3,4)
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(origs[:,:,0]))),interpolation='nearest',cmap=plt.cm.gray)

plt.subplot(2,3,2)
plt.imshow(orig_mean.real,interpolation='nearest',cmap=plt.cm.gray)
plt.title('Pixel averaging (no translation)')
plt.subplot(2,3,5)
#plt.imshow(np.angle(np.fft.fft2(orig_mean)),interpolation='nearest',cmap=plt.cm.gray)
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(orig_mean))),interpolation='nearest',cmap=plt.cm.gray)

img_recon1 = np.fft.ifft2(iB)
plt.subplot(2,3,3)
plt.imshow(img_recon1.real,interpolation='nearest',cmap=plt.cm.gray)
plt.title('BiSpectrum recursive reconstruction')
plt.subplot(2,3,6)
# plt.imshow(np.angle(iB),interpolation='nearest',cmap=plt.cm.gray)
#plt.imshow(img_recon1.real,interpolation='nearest',cmap=plt.cm.gray)
plt.imshow(np.abs(np.fft.fftshift(iB)),interpolation='nearest',cmap=plt.cm.gray)

# plt.savefig('chris_averaging_M%d_sigma%1.2f.pdf' % (M,sigma))


# plt.figure(1)
# plt.title('BiSpectrum average')
# a,b = invader.shape[0],invader.shape[1]
# plt.imshow(B.mean(4).reshape(a*b,a*b).real,interpolation='nearest',cmap=plt.cm.gray)


#stimulus = stadict[0][1]
#response = stadict[1][1]
#stadata = np.load('stadata.npz')
