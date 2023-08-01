#!/usr/bin/env python3

# Place import files below
import copy
import os

import astropy.constants as const
import astropy.units as units
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.special import erf
from matplotlib.text import Annotation
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Affine2D
from scipy.interpolate import interp1d
from scipy.stats import kde, mstats

from common_functions import NFWHalo, c_exp, nearest_half_decade, save_figures


def main():
    # Plot settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    field_halo_style = {'color': 'grey', 'marker': '.'}
    field_scatter_style = copy.deepcopy(field_halo_style)
    field_scatter_style.update({'s': 200})
    bsplash_1_style = {'color': "red", 'marker': 'P', 'cmap': plt.cm.YlGn_r}
    bsplash_2_style = {
        'color': "mediumseagreen",
        'marker': 'H',
        'cmap': plt.cm.YlGn_r
    }
    hermeian_style = {'color': 'cornflowerblue', 'marker': '^'}
    l_hermeian_style = {
        'color': np.asarray(['orange', 'magenta', 'r', 'b']),
        'marker': ['d', '*', 'X', 'p']
    }
    bound_adjust = 0.05
    n_bins = 100
    specify_contour_levels = [[0.05, 0.5, 0.9, 0.99], [0.05, 0.5, 0.9, 0.99],
                              [0.05, 0.5, 0.9]]
    manual_contour_label_locations = [(3.3, 7.1), (6.9, 8.5), (11.5, 4.6)]
    bsplash_1_manual_label_locations = [(6.3, 23.6), (8.7, 17.8), (20., 12.3)]
    bsplash_2_manual_label_locations = [(8.4, 38.6), (8.2, 22.1)]
    hermeian_label = 'Hermeian'
    bsplash_1_label = r'$N_{\rm peri} = 1$'
    bsplash_2_label = r'$N_{\rm peri} = 2$'
    field_label = 'Regular f\kern0ptield'

    # Simulation information
    m_dm = 2.e5  # Msun
    h = 0.677
    rho_crit = 127.20308482895439  # Msun kpc^-3

    # File locations
    c200_vmax_vmax_m200_plot_file = 'c200_vmax_vmax_m200.pdf'
    data_file = os.path.join('data', '17_11_z0_data.hdf5')
    ludlow_data_file = os.path.join('data', 'ludlow2014_logc_vs_logm200h.csv')

    # Read data
    with h5py.File(data_file, 'r') as data:
        bsplash_data = data['Backsplash'][...]
        field_data = data['Field'][...]
        hermeian_data = data['Hermeian'][...]

    ####################################################################
    # Read data
    ####################################################################
    field_cnfw = field_data[:, 3]
    field_vmax = field_data[:, 4]
    field_mahf = field_data[:, 6]  # Msun
    field_overdensity = field_data[:, 8]
    bsplash_cnfw = bsplash_data[:, 3]
    bsplash_vmax = bsplash_data[:, 4]
    bsplash_nperi = bsplash_data[:, 5]
    backsplash_mahf = bsplash_data[:, 7]  # Msun
    backsplash_overdensity = bsplash_data[:, 9]
    hermeian_cnfw = hermeian_data[:, 3]
    hermeian_vmax = hermeian_data[:, 4]
    hermeian_mahf = hermeian_data[:, 8]  # Msun
    select_l_herm = hermeian_data[:, 9].astype(bool)
    hermeian_overdensity = hermeian_data[:, 12]

    # Read in c200-M200 relation from Ludlow+(2014)
    ludlow_data = 10.**np.genfromtxt(ludlow_data_file)
    ludlow_data[:, 0] /= h
    ludlow_vmax = calculate_analytic_vmax(ludlow_data[:, 0], ludlow_data[:, 1],
                                          rho_crit)
    ludlow_vmax_m200 = interp1d(np.log10(ludlow_data[:, 0]),
                                np.log10(ludlow_vmax),
                                fill_value=np.nan,
                                bounds_error=False)

    ####################################################################
    # Generate data
    ####################################################################
    # Calculate Hermeian c_200, M200 from c_AHF, M_AHF
    hermeian_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.)
        for ovdens, cnfw in zip(hermeian_overdensity, hermeian_cnfw)
    ])
    hermeian_c200[hermeian_cnfw <= 0] = -1.
    hermeian_c_ratio = c_exp(hermeian_c200) / c_exp(hermeian_cnfw)
    hermeian_m200 = hermeian_c_ratio * hermeian_mahf

    ####################################################################
    # Calculate Backsplash c_200, M200 from c_AHF, M_AHF
    bsplash_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.)
        for ovdens, cnfw in zip(backsplash_overdensity, bsplash_cnfw)
    ])
    bsplash_c200[bsplash_cnfw <= 0] = -1.
    bsplash_c_ratio = c_exp(bsplash_c200) / c_exp(bsplash_cnfw)
    bsplash_m200 = bsplash_c_ratio * backsplash_mahf

    ####################################################################
    # Calculate Field c_200, M200 from c_AHF, M_AHF
    field_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.)
        for ovdens, cnfw in zip(field_overdensity, field_cnfw)
    ])
    field_c200[field_cnfw <= 0] = -1.
    field_c_ratio = c_exp(field_c200) / c_exp(field_cnfw)
    field_m200 = field_c_ratio * field_mahf

    ####################################################################
    # Make data selections
    ####################################################################
    field_select = (field_c200 > 0.)
    bsplash_1_select = (bsplash_c200 > 0.) * (bsplash_nperi == 1)
    bsplash_2_select = (bsplash_c200 > 0.) * (bsplash_nperi == 2)
    bsplash_3_select = (bsplash_c200 > 0.) * (bsplash_nperi == 3)
    all_hermeian_select = (hermeian_c200 > 0.)
    dm_hermeian_select = all_hermeian_select * (~select_l_herm)
    l_hermeian_select = all_hermeian_select * select_l_herm

    dm_hermeian_vmax = hermeian_vmax[dm_hermeian_select]
    dm_hermeian_c200 = hermeian_c200[dm_hermeian_select]
    l_hermeian_vmax = hermeian_vmax[l_hermeian_select]
    l_hermeian_c200 = hermeian_c200[l_hermeian_select]

    ####################################################################
    # Create data
    ####################################################################
    low_m200_cut = 10. * m_dm
    high_m200_cut = 4.01e13

    all_hermeian_vmax = hermeian_vmax[all_hermeian_select]
    all_hermeian_m200 = hermeian_m200[all_hermeian_select]
    all_bsplash_1_m200 = bsplash_m200[bsplash_1_select]
    all_bsplash_1_vmax = bsplash_vmax[bsplash_1_select]
    all_bsplash_2_m200 = bsplash_m200[bsplash_2_select]
    all_bsplash_2_vmax = bsplash_vmax[bsplash_2_select]
    all_bsplash_3_m200 = bsplash_m200[bsplash_3_select]
    all_field_m200 = field_m200[field_select]
    all_field_c200 = field_c200[field_select]
    all_field_vmax = field_vmax[field_select]

    m200_cut_hermeians = ((all_hermeian_m200 >= low_m200_cut) *
                          (all_hermeian_m200 <= high_m200_cut))
    min_hermeian_m200 = np.nanmin(all_hermeian_m200[m200_cut_hermeians])
    m200_cut_bsplash_1 = ((all_bsplash_1_m200 >= low_m200_cut) *
                          (all_bsplash_1_m200 <= high_m200_cut))
    m200_cut_bsplash_2 = ((all_bsplash_2_m200 >= low_m200_cut) *
                          (all_bsplash_2_m200 <= high_m200_cut))
    m200_cut_bsplash_3 = ((all_bsplash_3_m200 >= low_m200_cut) *
                          (all_bsplash_3_m200 <= high_m200_cut))
    m200_cut_field = ((all_field_m200 >= low_m200_cut) *
                      (all_field_m200 <= high_m200_cut))

    (m200_hermeian_n_bins, m200_hermeian_bin_edges) = calculate_n_bins(
        all_hermeian_m200[m200_cut_hermeians],
        min_n_per_bin=1,
        bin_distribution='log')
    (m200_bsplash_1_n_bins, m200_bsplash_1_bin_edges) = calculate_n_bins(
        all_bsplash_1_m200[m200_cut_bsplash_1], bin_distribution='log')
    (m200_bsplash_2_n_bins, m200_bsplash_2_bin_edges) = calculate_n_bins(
        all_bsplash_2_m200[m200_cut_bsplash_2], bin_distribution='log')
    (m200_bsplash_3_n_bins, m200_bsplash_3_bin_edges) = calculate_n_bins(
        all_bsplash_3_m200[m200_cut_bsplash_3])
    (m200_field_n_bins,
     m200_field_bin_edges) = calculate_n_bins(all_field_m200[m200_cut_field],
                                              bin_distribution='log')

    all_m200_bin_edges = np.concatenate([
        m200_hermeian_bin_edges, m200_bsplash_1_bin_edges,
        m200_bsplash_2_bin_edges, m200_bsplash_3_bin_edges,
        m200_field_bin_edges
    ])
    m200_lims = np.asarray(
        [np.nanmin(all_m200_bin_edges),
         np.nanmax(all_m200_bin_edges)]) * [0.95, 1.05]

    m200_hermeian_mid_bins = 10.**((np.log10(m200_hermeian_bin_edges[1:]) +
                                    np.log10(m200_hermeian_bin_edges[:-1])) /
                                   2.)
    m200_bsplash_1_mid_bins = 10.**((np.log10(m200_bsplash_1_bin_edges[1:]) +
                                     np.log10(m200_bsplash_1_bin_edges[:-1])) /
                                    2.)
    m200_bsplash_2_mid_bins = 10.**((np.log10(m200_bsplash_2_bin_edges[1:]) +
                                     np.log10(m200_bsplash_2_bin_edges[:-1])) /
                                    2.)
    m200_field_mid_bins = 10.**((np.log10(m200_field_bin_edges[1:]) +
                                 np.log10(m200_field_bin_edges[:-1])) / 2.)

    (hermeian_binned_vmax_m200, hermeian_btst_errs_vmax_m200,
     hermeian_sca_errs_vmax_m200) = calculate_binned_data(
         all_hermeian_m200[m200_cut_hermeians],
         all_hermeian_vmax[m200_cut_hermeians], m200_hermeian_bin_edges)
    (bsplash_1_binned_vmax_m200, bsplash_1_btst_errs_vmax_m200,
     bsplash_1_sca_errs_vmax_m200) = calculate_binned_data(
         all_bsplash_1_m200[m200_cut_bsplash_1],
         all_bsplash_1_vmax[m200_cut_bsplash_1], m200_bsplash_1_bin_edges)
    (bsplash_2_binned_vmax_m200, bsplash_2_btst_errs_vmax_m200,
     bsplash_2_sca_errs_vmax_m200) = calculate_binned_data(
         all_bsplash_2_m200[m200_cut_bsplash_2],
         all_bsplash_2_vmax[m200_cut_bsplash_2], m200_bsplash_2_bin_edges)
    (field_binned_vmax_m200, field_btst_errs_vmax_m200,
     field_sca_errs_vmax_m200) = calculate_binned_data(
         all_field_m200[m200_cut_field], all_field_vmax[m200_cut_field],
         m200_field_bin_edges)

    ####################################################################
    # Combined c200 vs Vmax and Vmax vs M200 (scatter)
    ####################################################################
    # Define the axes in the combined figure
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=(2, 8), hspace=0.)
    c200_v_vmax_ax = fig.add_subplot(gs[1, 0])
    vmax_v_m200_ax = fig.add_subplot(gs[1, 1])
    vmax_hist_ax = fig.add_subplot(gs[0, 0], sharex=c200_v_vmax_ax)
    m200_hist_ax = fig.add_subplot(gs[0, 1], sharex=vmax_v_m200_ax)

    vmax_hist_ax.tick_params(axis="x", labelbottom=False)
    m200_hist_ax.tick_params(axis="x", labelbottom=False)
    ax_hist_m200_ylim = [1.e-14, None]
    ax_hist_vmax_ylim = [1.e-6, None]

    # Collate data for bounds setting
    xdata = [dm_hermeian_vmax, l_hermeian_vmax]
    ydata = [dm_hermeian_c200, l_hermeian_c200]
    markers = [hermeian_style['marker'], l_hermeian_style['marker']]
    colors = [hermeian_style['color'], l_hermeian_style['color']]
    all_x = np.concatenate((np.concatenate(xdata), all_field_vmax))
    all_y = np.concatenate((np.concatenate(ydata), all_field_c200))

    # Set x/y limits
    xlims = [
        np.nanmin(all_x) * (1. - bound_adjust),
        np.nanmax(all_x) * (1. + bound_adjust)
    ]
    xlims = [xlims[0], xlims[1]]
    ylims = [0, np.nanmax(all_y) * (1. + bound_adjust)]
    ylims = [0, 105]

    # Apply kde to haloes
    mi, ci = np.mgrid[0.01:2:n_bins * 1j, 0:100:n_bins * 1j]
    field_zi, field_xbins, field_ybins = generate_kde_density_values(
        np.log10(all_field_vmax), all_field_c200, mi, ci, n_bins)
    (bsplash_1_zi, bsplash_1_xbins,
     bsplash_1_ybins) = generate_kde_density_values(
         np.log10(all_bsplash_1_vmax), bsplash_c200[bsplash_1_select], mi, ci,
         n_bins)
    (bsplash_2_zi, bsplash_2_xbins,
     bsplash_2_ybins) = generate_kde_density_values(
         np.log10(all_bsplash_2_vmax), bsplash_c200[bsplash_2_select], mi, ci,
         n_bins)

    c200_v_vmax_ax.set(xscale='log')

    ####################################################################
    # Prepare contour lines on panel plot
    ####################################################################
    zi_list = [field_zi, bsplash_1_zi, bsplash_2_zi]
    xbins = [field_xbins, bsplash_1_xbins, bsplash_2_xbins]
    ybins = [field_ybins, bsplash_1_ybins, bsplash_2_ybins]
    label_locations = [
        manual_contour_label_locations, bsplash_1_manual_label_locations,
        bsplash_2_manual_label_locations
    ]
    style_dicts = [field_halo_style, bsplash_1_style, bsplash_2_style]
    linestyles = ['solid', 'dashed', [(0, (4, 1, 0.5, 1, 0.5, 1))]]
    c_list = [
        all_field_c200, bsplash_c200[bsplash_1_select],
        bsplash_c200[bsplash_2_select]
    ]
    vmax_list = [all_field_vmax, all_bsplash_1_vmax, all_bsplash_2_vmax]
    label_list = [
        'Regular field', r'$N_{\rm peri} = 1$', r'$N_{\rm peri} = 2$'
    ]
    line_width_list = [1, 2, 2]
    s_list = [55, 30, 30]
    lw_list = [0, 0.5, 0.5]

    blank_clevel_list = set_up_contours(c200_v_vmax_ax, mi, ci, zi_list, xbins,
                                        ybins, specify_contour_levels,
                                        style_dicts, label_locations,
                                        linestyles, line_width_list)

    ####################################################################
    # Scatter plot haloes outside respective contour lines
    # (A bit hacky but works for illustrative purposes.
    # Full data available in the repo..)
    ####################################################################
    scatter_haloes_outside_contour(c200_v_vmax_ax, blank_clevel_list, c_list,
                                   vmax_list, style_dicts, label_list, s_list,
                                   lw_list)

    ####################################################################
    # Plot histograms in the top panel
    ####################################################################
    vmax_list = [all_field_vmax, all_bsplash_1_vmax, all_bsplash_2_vmax]
    concat_hermeian_vmax = np.concatenate((dm_hermeian_vmax, l_hermeian_vmax))
    style_dicts.append(hermeian_style)
    vmax_list.append(concat_hermeian_vmax)
    linestyle_list = np.copy(linestyles).tolist()
    linestyle_list.append('solid')
    linestyle_list = [
        item[0] if isinstance(item, list) else item for item in linestyle_list
    ]
    line_width_list = [1, 3, 3, 3]
    loop_panel_hist(vmax_hist_ax, vmax_list, style_dicts,
                    ax_hist_vmax_ylim[0] * 0.1, linestyle_list,
                    line_width_list)

    # Hermeian galaxies
    multi_scatter_plot(c200_v_vmax_ax,
                       xdata,
                       ydata,
                       markers,
                       colors,
                       xtype='lin',
                       ytype='lin',
                       label_data=[hermeian_label, None],
                       outline=[True, True],
                       zorder=100)

    # Locations of Hermeian galaxy markers in the top panel
    mid_y = 10**(np.sum(np.log10(vmax_hist_ax.get_ylim())) / 2.)

    for l_hermeian_vpeak_it, color, l_marker in zip(
            l_hermeian_vmax, l_hermeian_style['color'],
            l_hermeian_style['marker']):
        vmax_hist_ax.axvline(l_hermeian_vpeak_it,
                             color=color,
                             linestyle=':',
                             zorder=49)
        vmax_hist_ax.scatter(l_hermeian_vpeak_it,
                             mid_y,
                             color=color,
                             marker=l_marker,
                             lw=0.5,
                             edgecolors='k',
                             zorder=50)

    # c200-Vmax relation based on Ludlow+(2014)
    c200_v_vmax_ax.plot(ludlow_vmax,
                        ludlow_data[:, 1],
                        linestyle='-',
                        color='k',
                        zorder=0)

    # 100 DM particle line
    c200_v_vmax_ax.axvline(10.**ludlow_vmax_m200(np.log10(100. * m_dm)),
                           linestyle='--',
                           color='k')

    ####################################################################
    # c200 vs Vmax plot settings
    ####################################################################
    # Combined plot
    c200_v_vmax_ax.set(
        xlabel=r"$V_{\rm max}\, \left({\rm km\, s^{-1}}\right)$",
        ylabel=r"$c_{200}$",
        xlim=xlims,
        ylim=ylims)
    c200_v_vmax_ax.tick_params(axis='x', which='major', pad=7)
    c200_v_vmax_ax.minorticks_on()

    # Upper panel
    vmax_hist_ax.set(ylabel=r"$dP\, /\, d\log V_{\rm max}$")
    vmax_hist_ax.yaxis.set_major_locator(LogLocator(numticks=4))

    ####################################################################
    # Vmax vs M200 panel
    ####################################################################
    # HESTIA data
    m200_bins = [
        m200_hermeian_mid_bins, m200_bsplash_1_mid_bins,
        m200_bsplash_2_mid_bins, m200_field_mid_bins
    ]
    vmax_medians = [
        hermeian_binned_vmax_m200, bsplash_1_binned_vmax_m200,
        bsplash_2_binned_vmax_m200, field_binned_vmax_m200
    ]
    vmax_scatters = [
        hermeian_sca_errs_vmax_m200, bsplash_1_sca_errs_vmax_m200,
        bsplash_2_sca_errs_vmax_m200, field_sca_errs_vmax_m200
    ]
    style_dicts = [
        hermeian_style, bsplash_1_style, bsplash_2_style, field_scatter_style
    ]
    labels = [hermeian_label, bsplash_1_label, bsplash_2_label, field_label]
    zorders = [100, 96, 98, 5]
    loop_scatter_errors(vmax_v_m200_ax, m200_bins, vmax_medians, vmax_scatters,
                        style_dicts, labels, zorders)

    # Vmax-M200 relation based on Ludlow+(2014)
    vmax_v_m200_ax.plot(ludlow_data[:, 0],
                        ludlow_vmax,
                        color='k',
                        linestyle='-',
                        label='Ludlow+(2014)',
                        zorder=0)

    # 100 DM particle line
    vmax_v_m200_ax.axvline(100. * m_dm, linestyle='--', color='k')
    m200_hist_ax.axvline(100. * m_dm, linestyle='--', color='k')
    vmax_v_m200_ax.annotate(r'100 DM particles',
                            xy=(100. * m_dm, 70.),
                            ha='right',
                            va='top',
                            rotation=90,
                            xytext=(0, 0),
                            textcoords='offset points')

    # Reference lines of fixed concentration
    m200_vals = np.logspace(*np.log10([m200_lims[0], 1.e10]))
    for c in [5, 15, 30, 60]:
        line, = vmax_v_m200_ax.plot(m200_vals,
                                    calculate_analytic_vmax(
                                        m200_vals, c, rho_crit),
                                    linestyle='--',
                                    color='silver')
        line_annotate(r'$c_{{200}} = {}$'.format(c),
                      line,
                      9.e9,
                      ha='right',
                      va='bottom',
                      ax=vmax_v_m200_ax,
                      xytext=(0., 0.),
                      color='silver')

    # 100 DM particle line
    vmax_hist_ax.axvline(10.**ludlow_vmax_m200(np.log10(100. * m_dm)),
                         linestyle='--',
                         color='k')

    ####################################################################
    # Plot histograms in the top panel
    ####################################################################
    m200_list = [
        all_field_m200[all_field_m200 >= min_hermeian_m200],
        all_bsplash_1_m200[all_bsplash_1_m200 >= min_hermeian_m200],
        all_bsplash_2_m200[all_bsplash_2_m200 >= min_hermeian_m200],
        all_hermeian_m200[m200_cut_hermeians]
    ]
    style_dicts = [
        field_halo_style, bsplash_1_style, bsplash_2_style, hermeian_style
    ]
    loop_panel_hist(m200_hist_ax, m200_list, style_dicts,
                    ax_hist_m200_ylim[0] * 0.1, linestyle_list,
                    line_width_list)

    ####################################################################
    # Vmax vs M200 plot settings
    ####################################################################
    order = [1, 2, 3, 4, 0]

    # Combined figure panel
    vmax_v_m200_ax.tick_params(axis='x', which='major', pad=7)
    vmax_v_m200_ax.set(
        xlabel=r'$M_{200}\, \left[{\rm M_\odot}\right]$',
        ylabel=r'$V_{\rm max}\, \left({\rm km\, s^{-1}}\right)$',
        xscale='log',
        yscale='log',
        xlim=[m200_lims[0], 1.e10],
        ylim=[2., 80.])
    handles, labels = vmax_v_m200_ax.get_legend_handles_labels()
    vmax_v_m200_ax.legend([handles[idx] for idx in order],
                          [labels[idx] for idx in order],
                          frameon=True,
                          fancybox=True,
                          framealpha=0.75,
                          loc='lower right')

    m200_hist_ax.set(ylabel=r"$dP\, /\, d\log_{10} M_{200}$")
    m200_hist_ax.yaxis.set_major_locator(LogLocator(numticks=4))
    m200_hist_ax.axvline(2.e7, linestyle='--', color='k')
    vmax_v_m200_ax.set(xlabel=r"$M_{200}\, \left({\rm M_\odot}\right)$")
    vmax_v_m200_ax.minorticks_on()
    vmax_v_m200_ax.tick_params(axis='x', which='major', pad=7)

    save_figures(fig, c200_vmax_vmax_m200_plot_file)

    return None


def calculate_analytic_vmax(m200, c200, rho_crit):
    """Calculate Vmax assuming a NFW profile

    Args:
        m200 (fl): M200 mass [Msun].
        c200 (fl): Concentration parameter
        rho_crit (fl): Critical density [Msun / kpc^3].

    Returns:
        fl: Vmax [km/s].
    """
    chi_max = 2.16258 / c200
    mass_ratio_num = np.log(1. + (c200 * chi_max)) - ((c200 * chi_max) /
                                                      (1. + (c200 * chi_max)))
    mass_ratio_denom = np.log(1. + c200) - (c200 / (1. + c200))

    m_over_r = m200**(2. / 3.) * (
        (800. * np.pi * rho_crit) /
        (3. * units.Unit('kpc').to('km')**3.))**(1. / 3.)

    vmax = np.sqrt(
        const.G.to('km3 / (Msun s2)').value * m_over_r * mass_ratio_num /
        (mass_ratio_denom * chi_max))

    return vmax


def calculate_binned_data(x, y, bin_edges):
    # Bootstrap sampling
    n_samples = 1000

    # Error bounds
    CL = 100 * erf(1. / np.sqrt(2.))
    l_bound = (100. - np.float64(CL)) / 2.
    u_bound = 100. - l_bound

    bin_idxs = np.empty(len(x), dtype=int)
    binned_x = np.empty(len(bin_edges) - 1)
    binned_low_btst_err = np.empty(len(bin_edges) - 1)
    binned_high_btst_err = np.empty(len(bin_edges) - 1)
    binned_low_sca_err = np.empty(len(bin_edges) - 1)
    binned_high_sca_err = np.empty(len(bin_edges) - 1)
    bin_idxs[:] = -1
    binned_x[:] = np.nan
    binned_low_btst_err[:] = np.nan
    binned_high_btst_err[:] = np.nan
    binned_low_sca_err[:] = np.nan
    binned_high_sca_err[:] = np.nan

    for i, (low_bin, high_bin) in enumerate(zip(bin_edges[:-1],
                                                bin_edges[1:])):
        select_data = (np.around(x, 2) >= low_bin) * (x < np.around(
            high_bin, 2))
        if i == (len(bin_edges) - 2):
            select_data = ((np.around(x, 2) >= low_bin) *
                           (x <= np.around(high_bin, 2)))
        bin_idxs[select_data] = i + 1
        binned_x[i] = np.nanmedian(y[select_data])

        # Bootstrap errors
        idx_samples = np.random.randint(select_data.sum(),
                                        size=(select_data.sum(), n_samples))
        sample_median = np.nanmedian(y[select_data][idx_samples], axis=0)
        binned_low_btst_err[i] = np.nanpercentile(sample_median, l_bound)
        binned_high_btst_err[i] = np.nanpercentile(sample_median, u_bound)

        # Scatter errors
        binned_low_sca_err[i] = np.nanpercentile(y[select_data], l_bound)
        binned_high_sca_err[i] = np.nanpercentile(y[select_data], u_bound)

    bin_idxs[bin_idxs == -1] = len(bin_edges) - 1

    binned_bootstrap_errs = np.row_stack(
        (binned_x - binned_low_btst_err, binned_high_btst_err - binned_x))
    binned_scatter_errs = np.row_stack(
        (binned_x - binned_low_sca_err, binned_high_sca_err - binned_x))

    return binned_x, binned_bootstrap_errs, binned_scatter_errs


def calculate_n_bins(values_to_bin,
                     bin_distribution='equal',
                     min_n_per_bin=15):
    """Calculate number of bins and bin edges of data

    Args:
        values_to_bin (arr): Array of values to be binned.
        bin_distribution (str, optional): Can take one of: 'equal',
            'lin', or 'log'. Defines how the data are to be binned.
            Defaults to 'equal'.
        min_n_per_bin (int, optional): Minimum number of objects per
            bin. Defaults to 15. Ignored if bin_distribution != 'equal'.

    Returns:
        tuple: (Number of bins, and the bin edges)
    """
    if bin_distribution == 'equal':
        if len(values_to_bin)**(2. / 3.) < (2. * min_n_per_bin):
            n_bins = np.nanmax([1., int(len(values_to_bin) / min_n_per_bin)])
        else:
            n_bins = int(2. * len(values_to_bin)**(1. / 3.))

        if len(values_to_bin) >= 500:
            n_bins = int(n_bins / 2)

        bin_edges = mstats.mquantiles(values_to_bin,
                                      np.arange(n_bins + 1) / (n_bins))
    elif (bin_distribution == 'lin'):
        n_bins = int(2. * len(values_to_bin)**(1. / 3.))
        if len(values_to_bin) >= 500:
            n_bins = int(n_bins / 2)
        bin_edges = np.linspace(np.nanmin(values_to_bin),
                                np.nanmax(values_to_bin), n_bins)
    elif (bin_distribution == 'log'):
        n_bins = int(2. * len(values_to_bin)**(1. / 3.))
        if len(values_to_bin) >= 500:
            n_bins = int(n_bins / 2)
        bin_edges = np.logspace(np.log10(np.nanmin(values_to_bin)),
                                np.log10(np.nanmax(values_to_bin)), n_bins)
    else:
        raise ValueError(
            "bin_distribution must be one of: ['equal', 'lin', 'log']")

    return n_bins, bin_edges


def generate_kde_density_values(x, y, xi, yi, n_bins):
    xi, yi = np.mgrid[0.01:2:n_bins * 1j, 0:100:n_bins * 1j]

    k = kde.gaussian_kde((x, y))
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = (zi - zi.min()) / (zi.max() - zi.min())
    zi = zi.reshape(xi.shape)
    zi = 1. - zi

    counts, xbins, ybins = np.histogram2d(x, y, bins=50)
    return zi, xbins, ybins


class LineAnnotation(Annotation):
    """A sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can
    be drawn.

    Usage
    -----
    fig, ax = subplots()
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    ax.add_artist(LineAnnotation("text", line, 1.5))

    Credit:
    https://bit.ly/3rC3Dgh
    Modified from the class written by Jonas Hörsch on StackOverflow.
    """
    def __init__(self,
                 text,
                 line,
                 x,
                 xytext=(0, 5),
                 textcoords="offset points",
                 ax=None,
                 **kwargs):
        """Annotate the point at *x* of the graph *line* with text
        *text*.

        By default, the text is displayed with the same rotation as the
        slope of the graph at a relative position *xytext* above it
        (perpendicularly above).

        An arrow pointing from the text to the annotated point *xy* can
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.
        line : Line2D
            Matplotlib line object to annotate
        x : float
            The point *x* to annotate. y is calculated from the points
            on the line.
        xytext : (float, float), default: (0, 5)
            The position *(x, y)* relative to the point *x* on the
            *line* to place the
            text at. The coordinate system is determined by *textcoords*
        **kwargs
            Additional keyword arguments are passed on to `Annotation`.

        See also
        --------
        `Annotation`
        `line_annotate`
        """
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        if ax is not None:
            x_scale = ax.get_xscale()
            y_scale = ax.get_yscale()
        else:
            x_scale = 'lin'
            y_scale = 'lin'

        def neighbours(x, xs, ys, try_invert=True):
            inds, = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])

        self.neighbours = n1, n2 = neighbours(x, xs, ys)

        if x_scale == 'log':
            n1 = np.log10(self.neighbours[0])
            x_val = np.log10(x)
        else:
            n1 = self.neighbours[0]
            x_val = x

        if y_scale == 'log':
            n2 = np.log10(self.neighbours[1])
        else:
            n2 = self.neighbours[1]

        # Calculate y by interpolating neighbouring points
        y = n1[1] + ((x_val - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        if y_scale == 'log':
            y = 10.**y

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            **kwargs,
        }
        super().__init__(text, (x, y),
                         xytext=xytext,
                         textcoords=textcoords,
                         **kwargs)

    def get_rotation(self):
        """Determines angle of the slope of the neighbours in display
        coordinate system
        """
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours),
                         axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        """Updates relative position of annotation text
        Note
        ----
        Called during annotation `draw` call
        """
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(
            self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


def line_annotate(text, line, x, *args, **kwargs):
    """Add a sloped annotation to *line* at position *x* with *text*

    Optionally an arrow pointing from the text to the graph at *x* can
    be drawn.

    Usage
    -----
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    line_annotate("sin(x)", line, 1.5)

    See also
    --------
    `LineAnnotation`
    `plt.annotate`

    Credit:
    https://bit.ly/3rC3Dgh
    Thanks to Jonas Hörsch on StackOverflow for writing this function.
    """
    ax = line.axes
    a = LineAnnotation(text, line, x, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a


def loop_panel_hist(ax, x_list, style_dicts, bottom, ls_list, lw_list):
    for x, ls, lw, style_dict in zip(x_list, ls_list, lw_list, style_dicts):
        indiv_panel_hist(x,
                         ax,
                         color=style_dict['color'],
                         bin_type='log',
                         ls=ls,
                         lw=lw,
                         bottom=bottom)
    return None


def loop_scatter_errors(ax, m200_bins, vmax_medians, vmax_scatters,
                        style_dicts, labels, zorders):
    for m200_bin, vmax_med, vmax_scatter, style_dict, label, zorder in zip(
            m200_bins, vmax_medians, vmax_scatters, style_dicts, labels,
            zorders):
        scatter_with_errors(ax,
                            m200_bin,
                            vmax_med,
                            vmax_scatter,
                            style_dict,
                            label=label,
                            zorder=zorder)
    return None


def multi_scatter_plot(ax,
                       xdata,
                       ydata,
                       markers,
                       colors,
                       xtype='log',
                       ytype='log',
                       label_data=None,
                       outline=None,
                       zorder=None):
    if label_data is None:
        label_data = np.empty(len(xdata))
        label_data[:] = None

    if outline is None:
        outline = np.empty(len(xdata))
        outline[:] = None

    small_size = 15
    normal_size = 35
    large_size = 50

    for x, y, marker, color, label, edge in zip(xdata, ydata, markers, colors,
                                                label_data, outline):
        if edge is not None:
            if len(marker) > 1:
                for x_i, y_i, m_i, c_i in zip(x, y, marker, color):
                    if marker == 's':
                        s = small_size + 7
                    else:
                        s = large_size + 7
                    ax.scatter(x_i,
                               y_i,
                               marker=m_i,
                               color=c_i,
                               label=label,
                               linewidth=0.5,
                               edgecolors='k',
                               s=s,
                               zorder=zorder)
            else:
                if marker == 's':
                    s = small_size
                else:
                    s = normal_size
                ax.scatter(x,
                           y,
                           marker=marker,
                           color=color,
                           label=label,
                           linewidth=0.5,
                           edgecolors='k',
                           s=s,
                           zorder=zorder)
        else:
            if len(marker) > 1:
                for x_i, y_i, m_i, c_i in zip(x, y, marker, color):
                    if marker == 's':
                        s = small_size + 7
                    else:
                        s = large_size + 7
                    ax.scatter(x_i,
                               y_i,
                               marker=m_i,
                               color=c_i,
                               label=label,
                               s=s,
                               zorder=zorder)
            else:
                if marker == 's':
                    s = small_size
                else:
                    s = normal_size
                ax.scatter(x,
                           y,
                           marker=marker,
                           color=color,
                           label=label,
                           s=s,
                           zorder=zorder)

    conc_x = np.concatenate(xdata)
    conc_y = np.concatenate(ydata)

    if xtype == 'log':
        conc_y = conc_y[conc_x >= 0.]
        conc_x = conc_x[conc_x >= 0.]

    if ytype == 'log':
        conc_x = conc_x[conc_y >= 0.]
        conc_y = conc_y[conc_y >= 0.]

    if (len(conc_x) > 0) and (len(conc_y) > 0):
        xlim = np.asarray([
            nearest_half_decade(np.nanmin(conc_x), scale_type=xtype),
            nearest_half_decade(np.nanmax(conc_x), 'u', scale_type=xtype)
        ])
        ylim = np.asarray([
            nearest_half_decade(np.nanmin(conc_y), scale_type=ytype),
            nearest_half_decade(np.nanmax(conc_y), 'u', scale_type=ytype)
        ])

        xlim[xlim != xlim] = None
        ylim[ylim != ylim] = None

        ax.set(xlim=xlim, ylim=ylim)

    return None


def panel_arrow(ax, loc, color, outline=False, axis='y'):
    if axis == 'x':
        headlength = 0.05 * ax.get_ylim()[1]
        linelength = 0.06 * ax.get_ylim()[1]
        location = [loc, linelength, 0, 0. - linelength]
    else:
        headlength = 0.05 * ax.get_xlim()[1]
        linelength = 0.06 * ax.get_xlim()[1]
        location = [linelength, loc, 0. - linelength, 0.]

    headlength = 0.6 * linelength

    if outline:
        ax.arrow(*location,
                 width=1.,
                 length_includes_head=True,
                 head_length=headlength,
                 ec='k',
                 linestyle='--',
                 fc=color)
    else:
        ax.arrow(*location,
                 width=1.,
                 length_includes_head=True,
                 head_length=headlength,
                 ec='none',
                 fc=color)
    return None


def indiv_panel_hist(x,
                     x_ax,
                     color,
                     select=None,
                     bin_type='lin',
                     lw=None,
                     ls=None,
                     bin_edges=None,
                     **kwargs):

    if select is not None:
        x = x[select]

    x_n_bins = int(2 * len(x)**(1. / 3.))

    if bin_type == 'log':
        x_bins = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)),
                             x_n_bins + 1)
        set_x_log = True
    else:
        x_bins = x_n_bins
        set_x_log = False

    if bin_edges is not None:
        x_bins = bin_edges

    hist, bin_edges = np.histogram(x, bins=x_bins)

    hist_obj = x_ax.hist(x,
                         bins=x_bins,
                         histtype='step',
                         density=True,
                         color=color,
                         log=set_x_log,
                         linestyle=ls,
                         lw=lw,
                         **kwargs)
    return hist_obj


def panel_hist(x,
               y,
               ax_histx,
               ax_histy,
               color,
               xselect=None,
               yselect=None,
               xtype='lin',
               ytype='lin',
               lw=None,
               ls=None,
               **kwargs):

    indiv_panel_hist(x,
                     ax_histx,
                     color,
                     select=xselect,
                     bin_type=xtype,
                     lw=lw,
                     ls=ls,
                     **kwargs)
    indiv_panel_hist(y,
                     ax_histy,
                     color,
                     select=yselect,
                     bin_type=ytype,
                     lw=lw,
                     ls=ls,
                     **kwargs)

    return None


def set_up_contours(ax, mi, ci, zi_list, xbins, ybins, contour_level_list,
                    style_dicts, label_locations, linestyles, lw_list):
    blank_clevel_list = np.empty(len(zi_list)).tolist()
    visible_clevel_list = np.empty(len(zi_list)).tolist()

    for i, (zi, xbin, ybin, style_dict, linestyle, label_location,
            contour_levels, lw) in enumerate(
                zip(zi_list, xbins, ybins, style_dicts, linestyles,
                    label_locations, contour_level_list, lw_list)):
        # Invisible/unlabelled contour lines
        blank_clevel_list[i] = ax.contour(
            mi,
            ci,
            zi.reshape(mi.shape),
            contour_levels,
            extent=[10**xbin.min(), 10**xbin.max(),
                    ybin.min(),
                    ybin.max()],
            linewidths=3,
            colors=style_dict['color'],
            alpha=0.,
            linestyles=linestyle)

        # Contour lines/regions (lines to be labelled)
        visible_clevel_list[i] = ax.contour(
            10**mi,
            ci,
            zi.reshape(mi.shape),
            blank_clevel_list[i].levels,
            extent=[10**xbin.min(), 10**xbin.max(),
                    ybin.min(),
                    ybin.max()],
            linewidths=lw,
            colors=style_dict['color'],
            linestyles=linestyle,
            alpha=0.5)

        # Label contour lines
        ax.clabel(visible_clevel_list[i],
                  inline=True,
                  fontsize=10,
                  colors='k',
                  fmt="%1.2f",
                  manual=label_location[:len(visible_clevel_list[i].levels)],
                  inline_spacing=1)

    return blank_clevel_list


def scatter_haloes_outside_contour(ax, blank_clevels, c_list, vmax_list,
                                   style_dicts, label_list, s_list, lw_list):
    for blank_clevel, c, vmax, style_dict, label, s, lw in zip(
            blank_clevels, c_list, vmax_list, style_dicts, label_list, s_list,
            lw_list):
        # Contours
        p = blank_clevel.collections[-1].get_paths()
        inside = np.full_like(c, False, dtype=bool)
        for level in p:
            copy_level = copy.deepcopy(level)
            copy_vertices = copy_level._vertices
            end_vertex = [0, 0]
            copy_level._vertices = np.row_stack((copy_vertices, end_vertex))
            inside |= copy_level.contains_points(
                np.column_stack((np.log10(vmax), c)))
        # Scatter haloes outside highest contour
        if lw > 0:
            edgecolors = 'k'
        else:
            edgecolors = 'none'
        ax.scatter(vmax[~inside],
                   c[~inside],
                   s=s,
                   lw=lw,
                   edgecolors=edgecolors,
                   color=style_dict['color'],
                   marker=style_dict['marker'],
                   label=label,
                   zorder=51,
                   alpha=0.5)

        for coll in blank_clevel.collections:
            coll.remove()

    return None


def scatter_with_errors(ax, x, y, y_errs, style_dict, label=None, zorder=None):
    if zorder is not None:
        err_zorder = zorder - 1
    else:
        err_zorder = zorder
    ax.errorbar(x,
                y,
                y_errs,
                color=style_dict['color'],
                capthick=2,
                capsize=5,
                linestyle='None',
                zorder=err_zorder)
    ax.scatter(x,
               y,
               **style_dict,
               lw=0.5,
               edgecolors='k',
               label=label,
               zorder=zorder)
    return None


if __name__ == "__main__":
    main()
