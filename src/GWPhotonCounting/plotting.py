import jax.numpy as jnp

# plotting imports
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from copy import copy
params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'axes.titlesize':10,
   'text.usetex': True,
   'font.family':'serif',
   'font.serif':'Computer Modern'
   }
matplotlib.rcParams.update(params)
matplotlib.rcParams["font.serif"] = "Computer Modern Roman"
matplotlib.rcParams["font.family"] = "Serif"
matplotlib.rcParams['text.latex.preamble'] = r'\renewcommand{\mathdefault}[1][]{}'

from matplotlib.markers import MarkerStyle
import matplotlib as mpl


def generate_count_plot(data, detector):

    # TODO customize labels
    # TODO allow for discrete or continuous colorbar
    # TODO add second plot for the strain signal with the color-coded filters overlaid on top (maybe a second function)

    points = jnp.array([(float(f0), float(t0)) for f0 in detector.f0_values for t0 in detector.t0_values])

    color_values = [[0,0] for _ in range(int(detector.N_total_filters/2))]

    for i in range(detector.N_total_filters):
        
        label = detector.filter_labels[i]
        
        point_idx = int(jnp.argwhere(jnp.sum(jnp.array(points) - jnp.array(label[0:2]),axis=1) == 0)[0][0])
        
        if label[2] > 0:
            color_values[point_idx][1] = (data[i]/jnp.max(data))
            
        else:
            color_values[point_idx][0] = (data[i]/jnp.max(data))
            
    color_values = jnp.array(color_values)

    cmap = plt.cm.Blues
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = jnp.linspace(0, jnp.max(data), 100)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(3.375, 4))

    ax = plt.gca()
    plt.scatter(points.T[0], points.T[1], c=cmap(color_values.T[0]), edgecolor="k", marker=MarkerStyle("o", fillstyle="right"), s=70)
    plt.scatter(points.T[0], points.T[1], c=cmap(color_values.T[1]), edgecolor="k", marker=MarkerStyle("o", fillstyle="left"), s=70)

    ax2 = fig.add_axes([0.125, 0.9, 0.7775, 0.03])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, label=r'$\bar{N}_{k}$ [quanta]', norm=norm,
                                boundaries=bounds, format='%.2f', orientation='horizontal')


    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')

    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(r'Time offset [s]')

    return fig


def generate_count_plot_with_strain(data, detector, frequencies, strain):

    # TODO customize labels
    # TODO allow for discrete or continuous colorbar
    # TODO add second plot for the strain signal with the color-coded filters overlaid on top (maybe a second function)

    points = jnp.array([(float(f0), float(t0)) for f0 in detector.f0_values for t0 in detector.t0_values])

    color_values = [[0,0] for _ in range(int(detector.N_total_filters/2))]
    color_values_filter = []

    for i in range(detector.N_total_filters):
        
        label = detector.filter_labels[i]
        
        point_idx = int(jnp.argwhere(jnp.sum(jnp.array(points) - jnp.array(label[0:2]),axis=1) == 0)[0][0])
        
        if label[2] > 0:
            color_values[point_idx][1] = (data[i]/jnp.max(data))
            
        else:
            color_values[point_idx][0] = (data[i]/jnp.max(data))

        color_values_filter.append(data[i]/jnp.max(data))
            
    color_values = jnp.array(color_values)
    color_values_filter = jnp.array(color_values_filter)

    cmap = plt.cm.Blues
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = jnp.linspace(0, jnp.max(data), 100)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(2, 1, figsize=(3.375, 5), sharex=True, gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05})
    ax = axs[1]
    ax1 = axs[0]

    ax.scatter(points.T[0], points.T[1], c=cmap(color_values.T[0]), edgecolor="k", marker=MarkerStyle("o", fillstyle="right"), s=70)
    ax.scatter(points.T[0], points.T[1], c=cmap(color_values.T[1]), edgecolor="k", marker=MarkerStyle("o", fillstyle="left"), s=70)

    ax1a = ax1.twinx()

    for i in range(detector.N_total_filters):

        #if data[i]/jnp.max(data) > 0.25:
        ax1a.plot(frequencies, jnp.abs(detector.filter_functions[i]), color=cmap(color_values_filter[i]), lw=1, zorder=2+data[i])

    ax1.plot(frequencies, jnp.abs(strain), color="#605B56", label='Post-merger',zorder=2.999, lw=2)
    ax1.set_zorder(2.999)
    ax1.set_facecolor("none")
    ax1.legend(loc='upper left', frameon=False, handlelength=1)
    ax1.set_yscale("log")
    ax1a.set_yscale("log")
    ax1.set_xlim(1.4e3, 4.1e3)
    ax1.set_ylim(1e-27,1e-25)
    ax1a.set_ylim(1e-3,2e-1)


    ax2 = fig.add_axes([0.125, 0.9, 0.7775, 0.03])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, label=r'$\bar{N}_{\textrm{sig},k}$ [quanta]', norm=norm,
                                boundaries=bounds, format='%.2f', orientation='horizontal')

    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    ax2.set_xticks([0, 0.01, 0.02,0.03])
    ax2.set_xticklabels([0, r'$1\times10^{-2}$', r'$2\times10^{-2}$', r'$3\times10^{-2}$'])

    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(r'Time offset [s]')
    ax1.set_ylabel(r'Strain [1/$\sqrt{\mathrm{Hz}}$]')
    ax1a.set_ylabel(r'$|d_k(f)|$ [1/$\sqrt{\mathrm{Hz}}$]')
    plt.minorticks_off()

    return fig
