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
# from bispectrumcode.python.scrappystuff.bispectrum_tricky import *
from analysis.utils.visualization import *
from bispectrumcode.python.cjh_tests.bispectrum import *
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

def clean_neurons(neurons):
    """Tang data"""
    cleaned = []
    completed = []
    for n in neurons:
#         if np.max(n) >= 0.5:
        completed_trials = np.zeros(n.shape[1])
        n_completed = 0
        for trial in n:
            if np.mean(trial) > 1e-4 and np.sum(np.isnan(trial)) == 0:
                completed_trials += trial
                n_completed += 1
        if n_completed > 0:
            completed_trials /= n_completed
            cleaned.append(completed_trials)
            completed.append(n_completed)
    return np.array(cleaned)

def load_gallant(data_dir, num_frames=None, num_neurons=None, ds=True):
    """Gallant data"""
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

def make_movie(vid, interval=25, figsize=(5,5)):
    def init():
        return (im,)
    
    def animate(frame):
        im.set_data(frame)
        return (im,)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.zeros(vid[0].shape), vmin=np.min(vid), vmax=np.max(vid), cmap='Greys_r');
    plt.axis('off')
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=vid, interval=interval, blit=True)
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
    h = hypertools.plot(spectrum, animate=True, frame_rate=10, reduce=reduction, ndims=3, show=False);
    return HTML(h.line_ani.to_html5_video()) 

def plot_neuron(neurons, stim, neuron_idx, stim_means=None, n_stds=3, img_cols=20, img_scale=1, white=True, crop=True):
    mean_across_reps = neurons[neuron_idx]
#     threshold = np.mean(mean_across_reps) + n_stds * np.std(mean_across_reps)
    threshold = .5 * np.max(mean_across_reps)
    print('threshold: {}'.format(threshold))
    preferred_images = get_preferred_images(mean_across_reps, stim, threshold)
    mean_bs_img, ffts = rev_corr(mean_across_reps, stim, stim_means, method='bispectrum', threshold=threshold, white=white)
    mean_pixel_img = rev_corr(mean_across_reps, stim, stim_means, method='pixel', threshold=threshold, white=white)
    r = plt.figure(figsize=(20, 1))
    sns.heatmap(mean_across_reps[None, :], cmap="Greys")
    h = plt.figure(figsize=(20, 4))
    plt.hist(mean_across_reps, bins=200)
    plt.axvline(x=threshold, color='red')
    if crop:
        plot_grid(preferred_images[:, 70:90, 70:90], scale=img_scale, cols=img_cols)
    else:
        plot_grid(preferred_images, scale=img_scale, cols=img_cols)
    mean_bs_img[mean_bs_img < -40] = -10
    mean_bs_img[mean_bs_img > 40] = -10
    plot_grid(np.array([mean_bs_img, mean_pixel_img]), cols=2, scale=8)
    return mean_bs_img, ffts
    
def plot_shapes_and_ns(shapes_neurons, shapes, ns_neurons, ns, neuron_idx, n_stds=3, crop_ns=False):
    plot_neuron(shapes_neurons, shapes, neuron_idx, n_stds=n_stds, img_cols=20, img_scale=1, white=False, crop=True)
    plot_neuron(ns_neurons, ns, neuron_idx, n_stds=n_stds, img_cols=8, img_scale=3, white=False, crop=crop_ns)
    
def load_imagenet(data_dir):
    f = h5py.File(data_dir+'PicStimi2.mat','r')
    stim = np.transpose(np.array(f['PicStimi2'], dtype=int), (0, 3, 2, 1))
    crop_stim = np.zeros((stim.shape[0], 200, 200, stim.shape[3]))
    stripe = np.zeros(stim.shape[1], dtype=int)
    img = stim[500]
    for x, img in enumerate(stim):
        row_idx = []
        col_idx = []
        for i, row in enumerate(np.sum(img, axis=2)):
            if (row == stripe).all():
                row_idx.append(i)
        for j, col in enumerate(np.sum(img, axis=2).T):
            if (col == stripe).all():
                col_idx.append(j)
        c1 = np.delete(img, row_idx, axis=0)
        c2 = np.delete(c1, col_idx, axis=1)
        # Catching images that are (199, 199, 3)
        if c2.shape != (200,200,3):
            print(x)
            c2 = np.append(c2, c2[-1][None, :, :], axis=0)
        crop_stim[x] = c2
    crop_stim = np.array(crop_stim, dtype=int)
    return crop_stim

def get_bispectrum(stim):
    fts = []
    bs = []
    for img in stim:
        f = np.fft.fftshift(np.fft.fft2(img))[None, :, :]
        fts.append(f)
        bs.append(bispectrum_2d(f, stim.shape[1], stim.shape[2]))
    bs = np.array(bs)
    return bs

"""
ANALYSIS
"""

def whiten(data):
    data_mean = np.mean(data, axis=(1,2))
    data -= data_mean[:, None, None]
    dataFT = np.fft.fftshift(np.fft.fft2(data))
    nyq = np.int32(np.floor(data.shape[1]/2))
    freqs = np.linspace(-nyq, nyq-1, num=data.shape[1])
    fspace = np.meshgrid(freqs, freqs)
    rho = np.sqrt(np.square(fspace[0]) + np.square(fspace[1]))
    lpf = np.exp(-0.5 * np.square(rho / (0.7 * nyq)))
    w_filter = np.multiply(rho, lpf) # filter is in the frequency domain
    dataFT = np.squeeze(np.multiply(dataFT, w_filter[None, :]))
    data = np.squeeze(np.real(np.fft.ifft2(np.fft.ifftshift(dataFT))))
    return data, data_mean

def rgb_yuv(img, convert_to="yuv"):
    shape = img.shape
    converted_img = np.zeros((shape[2], shape[0] * shape[1]))
    if convert_to == "yuv":
        weights = np.array([[0.299, 0.587, 0.114],
                            [-0.14713, -0.28886, 0.436],
                            [0.615, -0.51488, -0.10001]])
    elif convert_to == "rgb":
        weights = np.array([[1, 0, 1.13983],
                            [1, -0.39465, -0.58060],
                            [1, 2.03211, 0]])
    for i, x in enumerate(img.T):
        converted_img[i] = x.ravel()
    converted_img = np.matmul(weights, converted_img)
    converted_img = np.reshape(converted_img.T, (shape[0], shape[1], shape[2]))
    return converted_img

def rev_corr(neuron, stim, stim_means, method='bispectrum', threshold=.5, white=False):
    revcorr = []
    fts = []
    responses = []
    for i, response in enumerate(neuron):
        if response > threshold:
            if method == 'bispectrum' or method == 'power_spectrum':
                imgFT = np.fft.fftshift(np.fft.fft2(stim[i]))[None, :, :]
            if method == 'bispectrum':
                fts.append(imgFT)
                img = bispectrum_2d(imgFT, stim.shape[1], stim.shape[2])
            elif method == 'power_spectrum':
                img = imgFT * np.conj(imgFT)
            else:
                img = stim[i]
            responses.append(response)
            revcorr.append(img * response)
    revcorr = np.sum(np.array(revcorr), axis=0) / np.sum(responses)
    if method == 'bispectrum':
        revcorr, ffts = inv_2d_bispectrum(revcorr, np.array(fts), stim.shape[1], stim.shape[2])
        return revcorr, ffts
    else:
#     if white:
#         revcorr = unwhiten(revcorr, stim_means)
#     if len(revcorr.shape) >= 3:
#         revcorr /= np.max(revcorr)
        return revcorr

def unwhiten(data, data_mean=None):
    nyq = np.int32(np.floor(data.shape[0]/2))
    freqs = np.linspace(-nyq, nyq-1, num=data.shape[0])
    fspace = np.meshgrid(freqs, freqs)
    rho = np.sqrt(np.square(fspace[0]) + np.square(fspace[1]))
    invw_filter = np.zeros(rho.shape)
    if len(data.shape) < 3:
        data = data[None, :, :]
    for i, a in enumerate(rho):
        for j, b in enumerate(a):
            invw_filter[i, j] = b ** -1 if b != 0 else 0
    for i, img in enumerate(data):
        unwhite = np.fft.fftshift(np.fft.fft2(img))
        unwhite = np.multiply(unwhite, invw_filter)
        unwhite = np.real(np.fft.ifft2(np.fft.ifftshift(unwhite)))
        if data_mean is not None:
            unwhite += data_mean[i].squeeze()
        data[i] = unwhite.squeeze()
    return data.squeeze()
    
def get_preferred_images(neuron, stim, threshold=.5):
    idxs = []
    responses = []
    for i, response in enumerate(neuron):
        if response > threshold:
            idxs.append(i)
            responses.append(response)
    idxs = np.array(idxs)[np.argsort(responses)][::-1]
    responses = np.array(responses)[np.argsort(responses)][::-1]
    print('max_stim: {}   response: {}'.format(idxs[0], responses[0]))
    return stim[idxs]


    
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

