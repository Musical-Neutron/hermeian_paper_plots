#!/usr/bin/env python3

# Place import files below
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from common_functions import NFWHalo, c_exp, save_figures


def main():
    # Plot settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    field_halo_style = {'color': 'grey', 'marker': '.'}
    bsplash_1_style = {'color': "red", 'marker': 'P', 'cmap': plt.cm.YlGn_r}
    bsplash_2_style = {
        'color': "mediumseagreen",
        'marker': 'H',
        'cmap': plt.cm.YlGn_r
    }
    hermeian_style = {
        'color': 'cornflowerblue',
        'marker': '^',
    }
    l_hermeian_style = {
        'color': np.asarray(['orange', 'magenta', 'r', 'b']),
        'marker': ['d', '*', 'X', 'p'],
        's': 80,
        'linewidth': 0.5,
        'edgecolors': 'k'
    }
    filtered_l_hermeian_style = {
        x: l_hermeian_style[x]
        for x in l_hermeian_style if x not in ['marker', 'color']
    }
    log_j_lims = [[15.3, 19.5], [14.7, 19.2], [13.5, 17.2], [13., 17.7]]
    low_dist = 250
    high_dist = 1400
    n_bins = 23
    x_label = r'$d_{\rm MW}\, \left({\rm kpc}\right)$'
    y_label = (r'$\log_{10}\!\left(J\right.$' + r'-' +
               r'$\left.{\rm factor}\, /\, {\rm GeV^2\, cm^{-5}}\right)$')

    # # Simulation information
    # h = 0.677

    # File locations
    data_file = os.path.join('data', '17_11_z0_data.hdf5')
    # ludlow_file = os.path.join('data', 'ludlow2014_logc_vs_logm200h.csv')

    ####################################################################
    # Read data
    ####################################################################
    with h5py.File(data_file, 'r') as data:
        hermeian_data = data['Hermeian'][()]
        backsplash_data = data['Backsplash'][()]
        field_data = data['Field'][()]

    ####################################################################
    # Hermeian data
    hermeian_c200 = hermeian_data[:, 3]
    hermeian_d_mw = hermeian_data[:, 7]  # kpc
    hermeian_mahf = hermeian_data[:, 8]  # Msun
    select_l_herm = hermeian_data[:, 9].astype(bool)
    hermeian_dm_log_j = hermeian_data[:, 11][~select_l_herm]
    hermeian_l_log_j = hermeian_data[:, 11][select_l_herm]
    hermeian_overdensity = hermeian_data[:, 12]

    hermeian_dm_d_mw = hermeian_d_mw[~select_l_herm]  # kpc
    hermeian_l_d_mw = hermeian_d_mw[select_l_herm]  # kpc
    hermeian_dm_log_j[hermeian_dm_log_j == -1.] = np.nan
    hermeian_l_log_j[hermeian_l_log_j == -1.] = np.nan

    ####################################################################
    # Backsplash data
    backsplash_cnfw = backsplash_data[:, 3]
    n_peri_backsplash = backsplash_data[:, 5]
    d_mw_backsplash = backsplash_data[:, 6]  # kpc
    backsplash_mahf = backsplash_data[:, 7]  # Msun
    log_j_backsplash = backsplash_data[:, 8]
    backsplash_overdensity = backsplash_data[:, 9]

    log_j_backsplash[log_j_backsplash == -1.] = np.nan

    select_nperi_1 = n_peri_backsplash == 1
    select_nperi_2 = n_peri_backsplash == 2

    ####################################################################
    # Field data
    field_cnfw = field_data[:, 3]
    d_mw_field = field_data[:, 5]  # kpc
    field_mahf = field_data[:, 6]  # Msun
    log_j_field = field_data[:, 7]
    field_overdensity = field_data[:, 8]

    log_j_field[log_j_field == -1.] = np.nan

    ####################################################################
    # # Read in c200-M200 relation from Ludlow+(2014)
    # ludlow_data = 10.**np.genfromtxt(ludlow_file)
    # ludlow_data[:, 0] /= h
    # ludlow_cM_relation = interp1d(ludlow_data[:, 0],
    #                               ludlow_data[:, 1],
    #                               fill_value=np.nan,
    #                               bounds_error=False)

    ####################################################################
    # Generate data
    ####################################################################
    # Calculate Hermeian M_200 from M_AHF
    hermeian_dm_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.)
        for ovdens, cnfw in zip(hermeian_overdensity[~select_l_herm],
                                hermeian_c200[~select_l_herm])
    ])
    hermeian_dm_c200[hermeian_c200[~select_l_herm] <= 0.] = -1.
    hermeian_l_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.) for ovdens, cnfw in zip(
            hermeian_overdensity[select_l_herm], hermeian_c200[select_l_herm])
    ])
    hermeian_l_c200[hermeian_c200[select_l_herm] <= 0.] = -1.

    hermeian_dm_c_ratio = c_exp(hermeian_dm_c200) / c_exp(
        hermeian_c200[~select_l_herm])
    hermeian_l_c_ratio = c_exp(hermeian_l_c200) / c_exp(
        hermeian_c200[select_l_herm])

    hermeian_dm_m200 = hermeian_dm_c_ratio * hermeian_mahf[~select_l_herm]
    hermeian_l_m200 = hermeian_l_c_ratio * hermeian_mahf[select_l_herm]

    ####################################################################
    # Calculate Backsplash M_200 from M_AHF
    backsplash_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.)
        for ovdens, cnfw in zip(backsplash_overdensity, backsplash_cnfw)
    ])
    backsplash_c200[backsplash_cnfw <= 0] = -1.
    backsplash_c_ratio = c_exp(backsplash_c200) / c_exp(backsplash_cnfw)
    backsplash_m200 = backsplash_c_ratio * backsplash_mahf

    ####################################################################
    # Calculate Field M_200 from M_AHF
    field_c200 = np.asarray([
        NFWHalo(ovdens, cnfw).redefine_halo(200.)
        for ovdens, cnfw in zip(field_overdensity, field_cnfw)
    ])
    field_c200[field_cnfw <= 0] = -1.
    field_c_ratio = c_exp(field_c200) / c_exp(field_cnfw)
    field_m200 = field_c_ratio * field_mahf

    ####################################################################
    # Plot J-factor vs. d_MW
    ####################################################################
    mass_bins = np.asarray([2.e7, 8.e7, 3.2e8, 1.25e9, 5.e9])
    mass_bins = mass_bins[::-1]
    mass_bin_exponents = np.floor(np.log10(mass_bins))
    mass_bin_labels = [
        (r'${}\times 10^{}$'.format(low_val / 10**low_e, int(low_e)) +
         r'$\leq M_{{200}}\, /\, {{\rm M_\odot}} <$' +
         r'${}\times 10^{}$').format(high_val / 10**high_e, int(high_e))
        for high_val, low_val, high_e, low_e in zip(
            mass_bins[:-1], mass_bins[1:], mass_bin_exponents[:-1],
            mass_bin_exponents[1:])
    ]

    ####################################################################
    # Plot all data as scatter points
    ####################################################################
    fig, axs = plt.subplots(2,
                            2,
                            sharex=True,
                            sharey='row',
                            figsize=(16, 16),
                            gridspec_kw={
                                'hspace': 0,
                                'wspace': 0,
                                'width_ratios': [1, 1],
                                'height_ratios': [1, 1],
                            })

    l_i = 0
    for i, (ax, high_mass, low_mass, log_j_lim, bin_label) in enumerate(
            zip(axs.reshape(-1), mass_bins[:-1], mass_bins[1:], log_j_lims,
                mass_bin_labels)):
        # Select haloes in mass range
        select_field_mass = ((field_m200 >= low_mass) *
                             (field_m200 < high_mass))
        select_bsplash_mass = ((backsplash_m200 >= low_mass) *
                               (backsplash_m200 < high_mass))
        select_hermeian_mass = ((hermeian_dm_m200 >= low_mass) *
                                (hermeian_dm_m200 < high_mass))

        # Select haloes in distance range
        select_field_dist = (d_mw_field >= low_dist) * (d_mw_field < high_dist)
        select_bsplash_dist = (d_mw_backsplash >= low_dist) * (d_mw_backsplash
                                                               < high_dist)
        select_hermeian_dist = (hermeian_dm_d_mw >=
                                low_dist) * (hermeian_dm_d_mw < high_dist)

        # Plot settings
        ax.set(ylim=log_j_lim)
        ax.set_title(bin_label, y=0.95, va='top')
        ax.set(xlim=[low_dist, high_dist])

        ####################################################################
        # Plot data in mass and distance bins
        ####################################################################
        # Dark Hermeians
        ax.scatter(
            hermeian_dm_d_mw[select_hermeian_dist * select_hermeian_mass],
            hermeian_dm_log_j[select_hermeian_dist * select_hermeian_mass],
            s=57,
            lw=0.5,
            edgecolors='k',
            zorder=99,
            color=hermeian_style['color'],
            marker=hermeian_style['marker'],
            label='Hermeian')
        # Backsplash N_peri == 2
        ax.scatter(d_mw_backsplash[select_bsplash_dist * select_bsplash_mass *
                                   select_nperi_2],
                   log_j_backsplash[select_bsplash_dist * select_bsplash_mass *
                                    select_nperi_2],
                   s=30,
                   lw=0.5,
                   edgecolors='k',
                   color=bsplash_2_style['color'],
                   marker=bsplash_2_style['marker'],
                   label=r'$N_{\rm peri}=2$')
        # Backsplash N_peri == 1
        ax.scatter(d_mw_backsplash[select_bsplash_dist * select_bsplash_mass *
                                   select_nperi_1],
                   log_j_backsplash[select_bsplash_dist * select_bsplash_mass *
                                    select_nperi_1],
                   s=30,
                   lw=0.5,
                   edgecolors='k',
                   color=bsplash_1_style['color'],
                   marker=bsplash_1_style['marker'],
                   label=r'$N_{\rm peri}=1$')
        # Field
        ax.scatter(d_mw_field[select_field_dist * select_field_mass],
                   log_j_field[select_field_dist * select_field_mass],
                   s=55,
                   lw=0.5,
                   edgecolors='k',
                   color=field_halo_style['color'],
                   marker=field_halo_style['marker'],
                   alpha=0.5,
                   label='Regular f\kern0ptield')
        # Luminous Hermeians
        select_l_hermeian_mass = ((hermeian_l_m200 >= low_mass) *
                                  (hermeian_l_m200 < high_mass))
        if select_l_hermeian_mass.sum():
            for d, lj in zip(hermeian_l_d_mw[select_l_hermeian_mass],
                             hermeian_l_log_j[select_l_hermeian_mass]):
                ax.scatter([d], [lj],
                           marker=l_hermeian_style['marker'][l_i],
                           color=l_hermeian_style['color'][l_i],
                           zorder=90,
                           **filtered_l_hermeian_style)
                l_i += 1

        # Plot settings
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    #########################################################################
    # Plot settings
    #########################################################################
    axs[0][0].minorticks_on()
    axs[1][0].minorticks_on()
    axs[1][0].tick_params(axis='x', which='major', pad=7)
    axs[1][1].tick_params(axis='x', which='major', pad=7)
    axs[1][0].legend(loc='lower left', fancybox=True, frameon=True)
    # Grand axes
    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.tick_params(labelcolor='none',
                       which='both',
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)
    big_ax.set(xlabel=x_label)
    big_ax.set(ylabel=y_label)
    big_ax.xaxis.labelpad = 7
    big_ax.yaxis.labelpad = 7

    save_figures(fig, 'binned_jfactors_all_scatter.pdf')

    ####################################################################
    # Plot all data as a mix of scatter points and shaded regions
    ####################################################################
    fig, axs = plt.subplots(2,
                            2,
                            sharex=True,
                            sharey='row',
                            figsize=(16, 16),
                            gridspec_kw={
                                'hspace': 0,
                                'wspace': 0,
                                'width_ratios': [1, 1],
                                'height_ratios': [1, 1],
                            })

    l_i = 0
    for i, (ax, high_mass, low_mass, log_j_lim, bin_label) in enumerate(
            zip(axs.reshape(-1), mass_bins[:-1], mass_bins[1:], log_j_lims,
                mass_bin_labels)):
        # Select haloes in mass range
        select_field_mass = ((field_m200 >= low_mass) *
                             (field_m200 < high_mass))
        select_bsplash_mass = ((backsplash_m200 >= low_mass) *
                               (backsplash_m200 < high_mass))
        select_hermeian_mass = ((hermeian_dm_m200 >= low_mass) *
                                (hermeian_dm_m200 < high_mass))

        # Select haloes in distance range
        select_field_dist = (d_mw_field >= low_dist) * (d_mw_field < high_dist)
        select_bsplash_dist = (d_mw_backsplash >= low_dist) * (d_mw_backsplash
                                                               < high_dist)
        select_hermeian_dist = (hermeian_dm_d_mw >=
                                low_dist) * (hermeian_dm_d_mw < high_dist)

        # Plot settings
        ax.set(ylim=log_j_lim)
        ax.set_title(bin_label, y=0.95, va='top')
        ax.set(xlim=[low_dist, high_dist])
        ax.yaxis.set_major_locator(MultipleLocator(2.))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        ####################################################################
        # Plot data in mass and distance bins
        ####################################################################
        # Dark Hermeians
        ax.scatter(
            hermeian_dm_d_mw[select_hermeian_dist * select_hermeian_mass],
            hermeian_dm_log_j[select_hermeian_dist * select_hermeian_mass],
            s=57,
            lw=0.5,
            edgecolors='k',
            zorder=99,
            color=hermeian_style['color'],
            marker=hermeian_style['marker'],
            label='Hermeian')
        # Backsplash N_peri == 2
        ax.scatter(d_mw_backsplash[select_bsplash_dist * select_bsplash_mass *
                                   select_nperi_2],
                   log_j_backsplash[select_bsplash_dist * select_bsplash_mass *
                                    select_nperi_2],
                   s=30,
                   lw=0.5,
                   edgecolors='k',
                   color=bsplash_2_style['color'],
                   marker=bsplash_2_style['marker'],
                   label=r'$N_{\rm peri}=2$')
        if i > 1:
            # Number of data points too large so plot median and
            # 16/84 percentiles
            # Backsplash N_peri == 1
            nperi_1_x, nperi_1_med, nperi_1_spread = binned_median(
                d_mw_backsplash[select_bsplash_dist * select_bsplash_mass *
                                select_nperi_1],
                log_j_backsplash[select_bsplash_dist * select_bsplash_mass *
                                 select_nperi_1],
                n_bins=n_bins)
            ax.plot(nperi_1_x,
                    nperi_1_med,
                    markeredgewidth=0.5,
                    markeredgecolor='k',
                    marker=bsplash_1_style['marker'],
                    color=bsplash_1_style['color'],
                    label=r'$N_{\rm peri}=1$')
            ax.fill_between(nperi_1_x,
                            *nperi_1_spread,
                            facecolor=bsplash_1_style['color'],
                            edgecolor=None,
                            zorder=0,
                            alpha=0.2)
            # Field
            field_x, field_med, field_spread = binned_median(
                d_mw_field[select_field_dist * select_field_mass],
                log_j_field[select_field_dist * select_field_mass],
                n_bins=n_bins)
            ax.plot(field_x,
                    field_med,
                    markeredgewidth=0.5,
                    markeredgecolor='k',
                    marker=field_halo_style['marker'],
                    color=field_halo_style['color'],
                    label='Regular f\kern0ptield')
            ax.fill_between(field_x,
                            *field_spread,
                            facecolor=field_halo_style['color'],
                            edgecolor=None,
                            zorder=0,
                            alpha=0.2)
        else:
            # Plot scatter points for N_peri == 1
            ax.scatter(d_mw_backsplash[select_bsplash_dist *
                                       select_bsplash_mass * select_nperi_1],
                       log_j_backsplash[select_bsplash_dist *
                                        select_bsplash_mass * select_nperi_1],
                       s=30,
                       lw=0.5,
                       edgecolors='k',
                       color=bsplash_1_style['color'],
                       marker=bsplash_1_style['marker'],
                       label=r'$N_{\rm peri}=1$')
            # Plot scatter points for Field
            ax.scatter(d_mw_field[select_field_dist * select_field_mass],
                       log_j_field[select_field_dist * select_field_mass],
                       s=55,
                       lw=0.5,
                       edgecolors='k',
                       color=field_halo_style['color'],
                       marker=field_halo_style['marker'],
                       alpha=0.5,
                       label='Regular f\kern0ptield')
        # Luminous Hermeians
        select_l_hermeian_mass = ((hermeian_l_m200 >= low_mass) *
                                  (hermeian_l_m200 < high_mass))
        if select_l_hermeian_mass.sum():
            for d, lj in zip(hermeian_l_d_mw[select_l_hermeian_mass],
                             hermeian_l_log_j[select_l_hermeian_mass]):
                ax.scatter([d], [lj],
                           marker=l_hermeian_style['marker'][l_i],
                           color=l_hermeian_style['color'][l_i],
                           zorder=90,
                           **filtered_l_hermeian_style)
                l_i += 1

        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ###########################################################################
    # Figure settings
    ###########################################################################
    axs[0][0].minorticks_on()
    axs[1][0].minorticks_on()
    axs[1][0].tick_params(axis='x', which='major', pad=7)
    axs[1][1].tick_params(axis='x', which='major', pad=7)
    axs[0][0].legend(loc='lower left', fancybox=True, frameon=True)
    # Grand axes
    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.tick_params(labelcolor='none',
                       which='both',
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)
    big_ax.set(xlabel=x_label)
    big_ax.set(ylabel=y_label)
    big_ax.xaxis.labelpad = 7
    big_ax.yaxis.labelpad = 7

    save_figures(fig, 'binned_jfactors_running_med.pdf')

    return None


def binned_median(x_data,
                  y_data,
                  n_bins,
                  bin_lims=None,
                  percentile=(16., 84.),
                  logspace=False):
    if bin_lims is not None:
        if logspace:
            bins = np.logspace(*np.log10(bin_lims), n_bins)
        else:
            bins = np.linspace(*bin_lims, n_bins)
    else:
        if logspace:
            bins = np.logspace(np.log10(np.nanmin(x_data)),
                               np.log10(np.nanmax(x_data)), n_bins)
        else:
            bins = np.linspace(np.nanmin(x_data), np.nanmax(x_data), n_bins)

    idx = np.digitize(x_data, bins)
    binned_median = [np.nanmedian(y_data[idx == k]) for k in range(n_bins)]
    binned_percentiles = [
        np.nanpercentile(y_data[idx == k], percentile) for k in range(n_bins)
    ]
    filtered_binned_percentiles = []
    for pcent in binned_percentiles:
        if np.isnan(pcent).any():
            filtered_binned_percentiles.append([np.nan, np.nan])
        else:
            filtered_binned_percentiles.append(pcent)
    filtered_binned_percentiles = np.row_stack(filtered_binned_percentiles).T

    if logspace:
        delta = np.log10(bins[1]) - np.log10(bins[0])
        plot_x_data = 10.**(np.log10(bins) - delta / 2.)
    else:
        delta = bins[1] - bins[0]
        plot_x_data = bins - delta / 2.

    return plot_x_data, binned_median, filtered_binned_percentiles


if __name__ == "__main__":
    main()
