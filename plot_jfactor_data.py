#!/usr/bin/env python3

# Place import files below
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

from common_functions import NFWHalo, c_exp, save_figures


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
    below_res_scatter_dict = {
        's': 35,
        'marker': '^',
        'linewidth': 1,
        'edgecolors': 'k',
        'linestyle': ':',
        'alpha': 1
    }
    obs_scatter_dict = {
        's': 80,
        'marker': '.',
        'linewidth': 0.3,
        'edgecolors': 'k',
        'alpha': 0.75
    }
    d_mw_axis_range = [200, 1400]
    log_j_axis_range = [13., 18.6]
    alpha_axis_range = [0.004, 2]

    # Simulation information
    m_dm = 2.e5  # Msun

    # File locations
    data_file = os.path.join('data', '17_11_z0_data.hdf5')
    gammaldi_data_file = os.path.join('data', 'gammaldi_2021_data.hdf5')
    jfactor_plot = 'hermeian_j_factors.pdf'

    # Read data
    with h5py.File(data_file, 'r') as data:
        hermeian_data = data['Hermeian'][...]
    with h5py.File(gammaldi_data_file, 'r') as data:
        gammaldi_data = np.float64(data['Gammaldi 2021'][...][:, 1:])

    select_l_herm = hermeian_data[:, 9].astype(bool)

    ####################################################################
    # Generate data
    ####################################################################
    m_ahf = hermeian_data[:, 8]
    c200 = hermeian_data[:, 3]
    d_mw = hermeian_data[:, 7]  # kpc
    log_j = hermeian_data[:, 10]
    alpha = hermeian_data[:, 11]
    overdensity = hermeian_data[:, 12]

    calc_hermeian_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.)
        for ovdens, cnfw in zip(overdensity, c200)
    ])
    calc_hermeian_c200[c200 <= 0] = -1.
    hermeian_c_ratio = c_exp(calc_hermeian_c200) / c_exp(c200)
    hermeian_m200 = hermeian_c_ratio * m_ahf
    log_m200 = np.log10(hermeian_m200)

    # Load Gammaldi+(2020) data
    g_c200 = gammaldi_data[:, 0]
    g_dmw = gammaldi_data[:, 1]  # kpc
    g_log_m200 = gammaldi_data[:, 2]
    g_alpha = gammaldi_data[:, 3]
    g_logj = gammaldi_data[:, 4]

    clean_haloes = (log_j > 0.)
    select_dm_haloes = clean_haloes * (~select_l_herm)
    select_l_haloes = clean_haloes * (select_l_herm)

    m200_vmin = np.nanmin(log_m200[clean_haloes])
    m200_vmax = np.nanmax(log_m200[clean_haloes])
    c200_vmin = np.nanmin(c200[clean_haloes])
    c200_vmax = np.nanmax(c200[clean_haloes])

    # Impose resolution threshold (we choose 100 DM particles)
    select_by_mass = log_m200 >= np.log10(100. * m_dm)

    # Calculate 68% confidence intervals
    CL = 100. * erf(1. / np.sqrt(2.))
    bounds = np.empty(2)
    bounds[0] = (100. - np.float64(CL)) / 2.
    bounds[1] = 100. - bounds[0]

    # Print out relevant numbers
    dm_median_jfactor_all = np.around(np.nanmedian(log_j[select_dm_haloes]), 2)
    dm_jfactor_confidence_all = np.around(
        np.nanpercentile(log_j[select_dm_haloes], bounds), 2)
    dm_median_jfactor_above_res = np.around(
        np.nanmedian(log_j[select_dm_haloes * select_by_mass]), 2)
    dm_jfactor_confidence_above_res = np.around(
        np.nanpercentile(log_j[select_dm_haloes * select_by_mass], bounds), 2)

    print("Average log_10 J-factor of entire dark Hermeian population")
    print("     +{:0.2f}".format(dm_jfactor_confidence_all[1] -
                                 dm_median_jfactor_all))
    print("{:0.2f}".format(dm_median_jfactor_all))
    print("     -{:0.2f}".format(dm_median_jfactor_all -
                                 dm_jfactor_confidence_all[0]))

    print(
        "Average log_10 J-factor of dark Hermeian population above res. lim.")
    print("     +{:0.2f}".format(dm_jfactor_confidence_above_res[1] -
                                 dm_median_jfactor_above_res))
    print("{:0.2f}".format(dm_median_jfactor_above_res))
    print("     -{:0.2f}".format(dm_median_jfactor_above_res -
                                 dm_jfactor_confidence_above_res[0]))

    print("log_10 J-factors of Hermeian galaxies")
    print("{}".format(np.sort(np.around(log_j[select_l_haloes], 2))))

    ####################################################################
    # Plot J-factor vs. d_MW and alpha plots
    ####################################################################
    hspace = 0.02

    fig, axs = plt.subplots(2,
                            2,
                            figsize=(16, 8),
                            gridspec_kw={
                                'hspace': hspace,
                                'wspace': 0,
                                'width_ratios': [1, 1],
                                'height_ratios': [0.05, 1]
                            })

    # Set shared axes separately as colorbar axes are not shared
    axs[1, 0].get_shared_y_axes().join(axs[1, 0], axs[1, 1])
    axs[1, 1].set_yticklabels([])

    # Plot data
    s0 = axs[1, 0].scatter(d_mw[select_dm_haloes * select_by_mass],
                           log_j[select_dm_haloes * select_by_mass],
                           c=log_m200[select_dm_haloes * select_by_mass],
                           cmap=colormap_1,
                           vmin=m200_vmin,
                           vmax=m200_vmax,
                           zorder=80,
                           **scatter_dict)
    axs[1, 0].scatter(d_mw[select_dm_haloes * (~select_by_mass)],
                      log_j[select_dm_haloes * (~select_by_mass)],
                      c=log_m200[select_dm_haloes * (~select_by_mass)],
                      cmap=colormap_1,
                      vmin=m200_vmin,
                      vmax=m200_vmax,
                      zorder=0,
                      **below_res_scatter_dict)
    s1 = axs[1, 1].scatter(alpha[select_dm_haloes * select_by_mass],
                           log_j[select_dm_haloes * select_by_mass],
                           c=c200[select_dm_haloes * select_by_mass],
                           cmap=colormap_2,
                           vmin=c200_vmin,
                           vmax=c200_vmax,
                           zorder=80,
                           **scatter_dict)
    axs[1, 1].scatter(alpha[select_dm_haloes * (~select_by_mass)],
                      log_j[select_dm_haloes * (~select_by_mass)],
                      c=c200[select_dm_haloes * (~select_by_mass)],
                      cmap=colormap_2,
                      vmin=c200_vmin,
                      vmax=c200_vmax,
                      zorder=0,
                      **below_res_scatter_dict)

    # Plot Gammaldi observation-based data
    axs[1, 0].scatter(g_dmw,
                      g_logj,
                      c=g_log_m200,
                      cmap=colormap_1,
                      vmin=m200_vmin,
                      vmax=m200_vmax,
                      zorder=81,
                      **obs_scatter_dict)
    axs[1, 1].scatter(g_alpha,
                      g_logj,
                      c=g_c200,
                      cmap=colormap_2,
                      vmin=c200_vmin,
                      vmax=c200_vmax,
                      zorder=81,
                      **obs_scatter_dict)

    # Plot luminous Hermeian haloes
    for i, (dist, a, j, m, cnfw) in enumerate(
            zip(d_mw[select_l_haloes], alpha[select_l_haloes],
                log_j[select_l_haloes], log_m200[select_l_haloes],
                c200[select_l_haloes])):
        axs[1, 0].scatter([dist], [j],
                          c=[m],
                          vmin=m200_vmin,
                          vmax=m200_vmax,
                          cmap=colormap_1,
                          marker=l_hermeian_style['marker'][i],
                          zorder=90,
                          **filtered_l_hermeian_style)
        axs[1, 1].scatter([a], [j],
                          c=[cnfw],
                          vmin=c200_vmin,
                          vmax=c200_vmax,
                          cmap=colormap_2,
                          marker=l_hermeian_style['marker'][i],
                          zorder=90,
                          **filtered_l_hermeian_style)

    ####################################################################
    # Plot settings
    ####################################################################
    ylabel = (r'$\log_{10}\!\left(J\right.$' + r'-' +
              r'$\left.{\rm factor}\, /\, {\rm GeV^2\, cm^{-5}}\right)$')

    axs[1, 0].set(xlabel=r'$d_{\rm MW}\, \left({\rm kpc}\right)$',
                  ylabel=ylabel,
                  xlim=d_mw_axis_range,
                  ylim=log_j_axis_range)
    axs[1, 1].set(xlabel=r'$\alpha\, \left({}^\circ\right)$',
                  xscale='log',
                  xlim=alpha_axis_range)

    axs[1, 1].axvline(0.1, linestyle=':', color=psf_color)
    axs[1, 1].annotate(r'$\gamma$-ray PSF',
                       xy=(0.11, 13.1),
                       xycoords='data',
                       color=psf_color)

    axs[1, 0].xaxis.set_major_locator(MultipleLocator(200.))
    ax_nbins = len(axs[1, 0].get_xticklabels())
    axs[1, 0].xaxis.set_major_locator(
        MaxNLocator(nbins=ax_nbins, prune='upper', steps=[2]))

    axs[0, 0].minorticks_on()
    axs[1, 0].minorticks_on()
    axs[1, 1].minorticks_on()
    axs[1, 1].xaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs[1, 0].tick_params(axis='x', which='major', pad=7)
    axs[1, 1].tick_params(axis='x', which='major', pad=7)

    colorbar(fig, s0, axs[0, 0])
    colorbar(fig, s1, axs[0, 1])

    xlabel_0 = r'$\log_{10}\!\left(M_{200}\, /\, {\rm M_\odot}\right)$'
    axs[0, 0].set(xlabel=xlabel_0)
    axs[0, 1].set(xlabel=r'$c_{\rm 200,\, DM}$')
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(10))

    save_figures(fig, jfactor_plot, embed=True)

    return None


def colorbar(fig, mappable, colorbar_axis):
    cbar = fig.colorbar(mappable, cax=colorbar_axis, orientation='horizontal')
    colorbar_axis.xaxis.tick_top()
    colorbar_axis.xaxis.set_label_position('top')
    return cbar


if __name__ == "__main__":
    main()
