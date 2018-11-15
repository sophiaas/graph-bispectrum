import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
import math
import numpy as np

def plot_bispectrum(orbits, orbit_index):
    for idx, m in enumerate(orbits[orbit_index]):
        bispectrum_ = compute_bispectrum(m)
        plot_grid(bispectrum_, "orbit " + str(orbit_index) + "-" + str(idx))
        
def plot_grid(batch, title=None, scale=3, img=True, show_colorbar=False, cmap="Greys", cols=6):
    N = len(batch)
    cols = cols
    rows = int(math.ceil(np.float(N) / cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(cols*scale, rows*scale))
    for n, idx in enumerate(batch):
        ax = fig.add_subplot(gs[n])
        if img == True:
            plot = ax.imshow(idx, cmap=cmap)
        else:
            plot = ax.matshow(idx, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_colorbar:
            plt.colorbar(plot, ax=ax)  
    if title is not None:
        fig.suptitle(title, fontsize=30)