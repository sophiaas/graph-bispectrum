"""
Calvin and Hobbes power spectrum example:
Swap phase of Calvin (keep his FFT amplitudes) with phase of Hobbes (you should see a "Hobbes")
to show power spectrum is super not a complete set of invariants
Chris Hillar, Feb 2013
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


im_size = (90,100)
calvin = 1.0*np.asarray(Image.open('calvin_sml.png'))[:,:,0]
hobbes = 1.0*np.asarray(Image.open('hobbes_sml.png'))[:,:,0]

C = np.fft.fft2(calvin,calvin.shape)
Cimag = np.imag(C)
Creal = np.real(C)
Cabs = np.abs(C)
Cphase = C/Cabs

H = np.fft.fft2(hobbes,hobbes.shape)
Himag = np.imag(H)
Hreal = np.real(H)
Habs = np.abs(H)
Hphase = H/Habs

CAmp_HPhs = Cabs*Hphase 
camp_hphs = np.fft.ifft2(Cabs*Hphase)

plt.figure(1)
plt.imshow(np.abs(camp_hphs),interpolation='nearest',cmap=plt.cm.gray)
plt.title('Calvin amplitudes, Hobbes phase')

CAmp_CPhs = Habs*Cphase 
camp_cphs = np.fft.ifft2(Cabs*Cphase)
plt.figure(2)
plt.imshow(np.abs(camp_cphs),interpolation='nearest',cmap=plt.cm.gray)
plt.title('Calvin amplitudes, Calvin phase')

plt.figure(4)
plt.imshow(hobbes,interpolation='nearest',cmap=plt.cm.gray)
plt.title('Hobbes image')

power_diff = np.abs(np.fft.fft2(camp_hphs,hobbes.shape))**2 - np.abs(np.fft.fft2(camp_cphs))**2
plt.figure(3)
plt.imshow(np.abs(power_diff),interpolation='nearest',cmap=plt.cm.gray)
plt.title('Power spectrum diff btwn images')