from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from bispectrumcode.python.scrappystuff.bispectrum_tricky import *
import sklearn.decomposition
import cmath
import os
import imageio
import seaborn as sns
import scipy
import hypertools

"""
DATA WRANGLING
"""

def load_gallant(data_dir, num_frames=None, num_neurons=None, ds=True):
    files = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f)) and f[-3:]=='mat']
    spike_count = []
    stim = {}
    neuron = []
    for i, f in enumerate(files[:num_neurons]):
        loaded = h5py.File(data_dir+f,'r')
        num = f[3:6]
        spikes = np.array(loaded['psths']).tolist()[0][:num_frames]
        spike_count += [x for x in spikes]
        vid = np.array(loaded['rawStims'], dtype=np.float32)[:num_frames]
        vid = np.transpose(vid, axes=[0,2,1])
        vid = (vid.transpose() - np.mean(vid, axis=(1,2))).transpose()
        if ds:
            vid = downsample(vid)
        stim[num] = vid
        neuron += [num] * len(vid)
    data = pd.DataFrame({'neuron': neuron, 'spike_count': spike_count})
    return data, stim

def load_park_run(params):
    files = os.listdir(params['data_dir'])
    files.sort()
    vid = np.zeros(shape=[len(files), params['img_dim'][0], params['img_dim'][1]])
    for i, f in enumerate(files):
        img = imageio.imread(params['data_dir']+f)
        vid[i] = np.mean(img, axis=2)
    vid = vid[:, :, 15:]
    return vid

def downsample(data, ds_factor=2):
    ds = np.zeros((len(data), int(data.shape[1]/ds_factor), int(data.shape[2]/ds_factor)))
    for idx, img in enumerate(data):
        new_img = np.zeros((img.shape[0] / ds_factor, img.shape[1] / ds_factor))
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                patch = img[i*ds_factor:(i+1)*ds_factor, j*ds_factor:(j+1)*ds_factor]
                new_img[i, j] = np.mean(patch)
            ds[idx] = new_img
    return ds

def ravel_data(data):
    return np.reshape(data, (data.shape[0], np.product(data.shape[1:])))


"""
PLOTTING
"""

def make_movie(vid):
    def init():
        return (im,)
    
    def animate(frame):
        im.set_data(frame)
        return (im,)
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros(vid[0].shape), vmin=np.min(vid), vmax=np.max(vid), cmap='Greys_r');
    plt.axis('off')
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=vid, interval=25, blit=True)
    plt.close()
    return HTML(anim.to_html5_video())    

def show_stim(neuron, df, n_frames=100):
    vid = df[neuron][:n_frames]
    html_vid = make_movie(vid)
    return html_vid

def plot_spike_counts(neuron, data, num_frames=100):
    plt.bar(x=range(num_frames), height=data[data['neuron']=='058'].spike_count[:num_frames])
    
def raster_plot(data, num_neurons, num_frames):
    raster = np.zeros((num_neurons, num_frames))
    for i, n in enumerate(data.neuron.unique()[:num_neurons]):
        raster[i] = data[data.neuron==n].spike_count[:num_frames]
    return raster

def anim2d(spectrum, interval=25, line=True):
    plt.rcParams["figure.figsize"] = (10,10)
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    if line:
        ln, = plt.plot([], [], lw=3, animated=True)
    else:
        ln, = plt.plot([], [], 'ro', animated=True)

    def init():
        ax.set_xlim(np.min(spectrum[:,0]), np.max(spectrum[:,0]))
        ax.set_ylim(np.min(spectrum[:,1]), np.max(spectrum[:,1]))
        return ln,

    def update(frame):
        xdata.append(frame[0])
        ydata.append(frame[1])
        ln.set_data(xdata, ydata)
        return ln,

    ani = animation.FuncAnimation(fig, update, frames=spectrum,
                        init_func=init, interval=interval, blit=True)
    plt.close()
    return HTML(ani.to_html5_video())

def anim3d(spectrum, reduction=None):
    plt.rcParams["figure.figsize"] = (10,10)
    h = hypertools.plot(spectrum, animate=True, frame_rate=10, reduce=reduction, ndims=3);
    plt.close()
    return HTML(h.line_ani.to_html5_video()) 

"""
ANALYSIS
"""

def PCA_reduction(data, n_components=100):
    means = data.mean(axis=(1))[:,None]
    data -= means
    C = np.cov(data.T)
    U, S, V = np.linalg.svd(C)
    S = 1/np.sqrt(S + 1e-6)
    S = np.diag(S)
    U = U[:, :n_components]
    reduced = np.matmul(data, U)
    return reduced

def compute_spectra_batch(stim, dim_reduc=True, n_components=100, whiten=True, truncate_bs=True):
    spectra_df = pd.DataFrame()
    for neuron in stim.keys():
        spectra = compute_spectra(stim[neuron], dim_reduc, n_components, whiten, truncate_bs)
        spectra_df = spectra_df.append(spetra, ignore_index=True)
    spectra_df = spectra_df.set_index('neuron', drop=True)
    return spectra_df

def compute_spectra(vid, dim_reduc=True, n_components=100, whiten=True, truncate_bs=True):
    vid_power_spectrum = []
    vid_bispectrum = []
    vid_pca = []
    vid_ft = []
    for img in vid:
        new_img = img - img.mean()
        imgFT = np.fft.fftshift(np.fft.fft2(new_img))
        if whiten:
            nyq = np.int32(np.floor(img.shape[0]/2))
            freqs = np.linspace(-nyq, nyq-1, num=img.shape[0])
            fspace = np.meshgrid(freqs, freqs)
            rho = np.sqrt(np.square(fspace[0]) + np.square(fspace[1]))
            lpf = np.exp(-0.5 * np.square(rho / (0.7 * nyq)))
            w_filter = np.multiply(rho, lpf) # filter is in the frequency domain
            imgFT = np.squeeze(np.multiply(imgFT, w_filter[None, :]))
            new_img = np.squeeze(np.real(np.fft.ifft2(np.fft.ifftshift(imgFT))))
        img_power_spectrum = imgFT * np.conj(imgFT)
        img_pca = PCA_reduction(new_img, n_components)
        vid_power_spectrum.append(img_power_spectrum)
        vid_bispectrum.append(bispectrum(imgFT, truncated=truncate_bs))
        vid_pca.append(img_pca)
        vid_ft.append(imgFT)
    vid_ft = ravel_data(np.asarray(vid_ft))
    vid_power_spectrum = ravel_data(np.asarray(vid_power_spectrum))
    vid_bispectrum = ravel_data(np.asarray(vid_bispectrum))
    vid_pca = ravel_data(np.asarray(vid_pca))
    if dim_reduc:
        vid_ft = PCA_reduction(vid_ft, n_components)
        vid_power_spectrum = PCA_reduction(vid_power_spectrum, n_components)
        vid_bispectrum = PCA_reduction(vid_bispectrum, n_components)
    return {'fourier': vid_ft, 'power_spectrum': vid_power_spectrum, 'bispectrum': vid_bispectrum, 'pca': vid_pca}

