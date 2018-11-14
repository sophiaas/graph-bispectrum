"""
Translated noisy images of invader average in
bispectrum space to invert back to clean invader
Chris Hillar, Feb 2018
"""

import numpy as np
import scipy
import scipy.ndimage
from bispectrum import translate_img, bispectrum_2d, inv_2d_bispectrum
from PIL import Image
import matplotlib.pyplot as plt
import h5py

h5 = h5py.File('spaceinvader.h5', 'r')
X = h5['/x'][:]
X = -h5['/y'][:]
h5.close()
X = (X-X.mean()) / X.std();
invader = -X.T

T = 50
sigma = .1
m, n = 24, 24

origs = np.zeros((T, m, n),dtype=np.complex128)

count = 0
print("Creating random translations")
for i in range(T):
        new_im = translate_img(invader,x=np.random.random_integers(m),y=np.random.random_integers(n))
        origs[i,:,:] = new_im.copy() + sigma * np.random.randn(m,n) # np.random.random((invader.shape[0],invader.shape[1]))

orig_fft = np.zeros((T, m, n),dtype=np.complex128)
count = 0
for i in range(T):
    orig_fft[i, :,:] = np.fft.fft2(origs[i,:,:])

print("Computing bispectra")
B = bispectrum_2d(orig_fft, m, n)

print("Computing inverse")
img = inv_2d_bispectrum(B, orig_fft, m, n)
plt.matshow(img, cmap='gray')
plt.title('recovery functional (least squares)')

