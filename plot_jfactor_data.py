#!/usr/bin/env python3

# Place import files below
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, MaxNLocator


def main():
    # Plot settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    l_hermeian_style = {
        'marker': ['d', '*', 'X', 'p'],
        's': 80,
        'linewidth': 0.5,
        'edgecolors': 'k'
    }
    filtered_l_hermeian_style = {
        x: l_hermeian_style[x]
        for x in l_hermeian_style if x not in ['marker']
    }
    colormap_1 = 'viridis_r'
    colormap_2 = 'plasma_r'
    psf_color = 'grey'
    scatter_dict = {
        's': 40,
        'marker': '^',
        'linewidth': 0.5,
        'edgecolors': 'k'
    }

    # File locations
    data_file = os.path.join('data', '17_11_z0_data.hdf5')
    jfactor_plot = 'hermeian_j_factors.pdf'

    # Read data
    with h5py.File(data_file, 'r') as data:
        hermeian_data = data['Hermeian'][...]

    select_l_herm = hermeian_data[:, 9].astype(bool)

    ####################################################################
    # Generate data
    ####################################################################
    log_m200 = np.log10(hermeian_data[:, 8])
    c200 = hermeian_data[:, 3]
    d_mw = hermeian_data[:, 7]  # kpc
    log_j = hermeian_data[:, 10]
    alpha = hermeian_data[:, 11]

    clean_haloes = (log_j > 0.)
    select_dm_haloes = clean_haloes * (~select_l_herm)
    select_l_haloes = clean_haloes * (select_l_herm)

    m200_vmin = np.nanmin(log_m200[clean_haloes])
    m200_vmax = np.nanmax(log_m200[clean_haloes])
    c200_vmin = np.nanmin(c200[clean_haloes])
    c200_vmax = np.nanmax(c200[clean_haloes])

    fig, axs = plt.subplots(1,
                            2,
                            sharey=True,
                            figsize=(16, 8),
                            gridspec_kw={
                                'hspace': 0,
                                'wspace': 0
                            })

    s0 = axs[0].scatter(d_mw[select_dm_haloes],
                        log_j[select_dm_haloes],
                        c=log_m200[select_dm_haloes],
                        cmap=colormap_1,
                        vmin=m200_vmin,
                        vmax=m200_vmax,
                        **scatter_dict)
    s1 = axs[1].scatter(alpha[select_dm_haloes],
                        log_j[select_dm_haloes],
                        c=c200[select_dm_haloes],
                        cmap=colormap_2,
                        vmin=c200_vmin,
                        vmax=c200_vmax,
                        **scatter_dict)
    for i, (dist, a, j, m, cnfw) in enumerate(
            zip(d_mw[select_l_haloes], alpha[select_l_haloes],
                log_j[select_l_haloes], log_m200[select_l_haloes],
                c200[select_l_haloes])):
        axs[0].scatter([dist], [j],
                       c=[m],
                       vmin=m200_vmin,
                       vmax=m200_vmax,
                       cmap=colormap_1,
                       marker=l_hermeian_style['marker'][i],
                       **filtered_l_hermeian_style)
        axs[1].scatter([a], [j],
                       c=[cnfw],
                       vmin=c200_vmin,
                       vmax=c200_vmax,
                       cmap=colormap_2,
                       marker=l_hermeian_style['marker'][i],
                       **filtered_l_hermeian_style)

    axs[1].axvline(0.1, linestyle=':', color=psf_color)
    axs[1].annotate(r'$\gamma$-ray PSF',
                    xy=(0.11, 13.1),
                    xycoords='data',
                    color=psf_color)

    axs[0].set(
        xlabel=r'$d_{\rm MW}\, \left({\rm kpc}\right)$',
        ylabel=
        r'$\log_{10}\ J$-${\rm factor}\, \left({\rm GeV^2\, cm^{-5}}\right)$',
        xlim=[200, 1400],
        ylim=[13, 18])
    axs[1].set(xlabel=r'$\alpha\, \left({}^\circ\right)$',
               xscale='log',
               xlim=[0.004, 0.5])

    axs[0].xaxis.set_major_locator(MultipleLocator(200.))
    ax_nbins = len(axs[0].get_xticklabels())
    axs[0].xaxis.set_major_locator(
        MaxNLocator(nbins=ax_nbins, prune='upper', steps=[2]))

    axs[0].minorticks_on()
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs[0].tick_params(axis='x', which='major', pad=7)
    axs[1].tick_params(axis='x', which='major', pad=7)

    cbar_s0, cax_s0 = colorbar(s0)
    cbar_s1, cax_s1 = colorbar(s1)

    cax_s0.set(xlabel=r'$\log_{10}\, M_{200}\, \left({\rm M_\odot}\right)$')
    cax_s1.set(xlabel=r'$c_{200}$')
    cax_s1.xaxis.set_major_locator(MultipleLocator(10))

    save_figures(fig, jfactor_plot)

    return None


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    plt.sca(last_axes)
    return cbar, cax


def save_figures(fig, location):
    if '.pdf' in location:
        pdf_file = location
        svg_file = location.replace('.pdf', '.svg')
    else:
        pdf_file = location + '.pdf'
        svg_file = location + '.svg'

    fig.savefig(pdf_file, dpi=600, format='pdf', transparent=False)
    fig.savefig(svg_file, dpi=600, format='svg', transparent=False)

    return None


if __name__ == "__main__":
    main()
