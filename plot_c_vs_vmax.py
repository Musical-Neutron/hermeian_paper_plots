#!/usr/bin/env python3

# Place import files below
import copy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import LogLocator
from scipy.interpolate import interp1d
from scipy.stats import kde, kstest


def main():
    # Plot settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    field_halo_style = {'color': 'grey', 'marker': '.'}
    bsplash_style = {'color': "#7E317B", 'marker': 's'}
    bsplash_2_style = {
        'color': "mediumseagreen",
        'marker': 's',
        'cmap': plt.cm.YlGn_r
    }
    hermeian_style = {'color': 'cornflowerblue', 'marker': '^'}
    l_hermeian_style = {
        'color': np.asarray(['orange', 'magenta', 'r', 'b']),
        'marker': ['d', '*', 'X', 'p']
    }
    bound_adjust = 0.05
    n_bins = 50
    vmax_cut = 33
    specify_contour_levels = [0.05, 0.5, 0.9, 0.99]
    manual_contour_label_locations = [(3.3, 7.1), (6.9, 8.5), (11.5, 4.6)]
    bsplash_manual_label_locations = [(6.11, 22.45), (11.4, 23.1), (20., 12.3)]

    # File locations
    c_vs_vmax_plot = 'lg_c_vs_vmax.pdf'
    data_file = os.path.join('data', '17_11_z0_data.hdf5')

    # Read data
    with h5py.File(data_file, 'r') as data:
        bsplash_data = data['Backsplash'][...]
        field_data = data['Field'][...]
        hermeian_data = data['Hermeian'][...]

    select_l_herm = hermeian_data[:, 9].astype(bool)

    ####################################################################
    # Generate data
    ####################################################################

    field_cnfw = field_data[:, 3]
    field_vmax = field_data[:, 4]
    bsplash_cnfw = bsplash_data[:, 3]
    bsplash_vmax = bsplash_data[:, 4]
    bsplash_nperi = bsplash_data[:, 5]
    hermeian_cnfw = hermeian_data[:, 3][~select_l_herm]
    hermeian_vmax = hermeian_data[:, 4][~select_l_herm]
    l_hermeian_cnfw = hermeian_data[:, 3][select_l_herm]
    l_hermeian_vmax = hermeian_data[:, 4][select_l_herm]

    field_select = (field_cnfw > 0.)
    bsplash_select = (bsplash_cnfw > 0.)
    bsplash_2_select = (bsplash_cnfw > 0.) * (bsplash_nperi == 2)
    hermeian_select = (hermeian_cnfw > 0.)

    ####################################################################
    # Make plot
    ####################################################################
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2,
                          2,
                          width_ratios=(8, 4),
                          height_ratios=(2, 8),
                          left=0.1,
                          right=0.9,
                          bottom=0.1,
                          top=0.9,
                          wspace=0.,
                          hspace=0.)

    # Define the axes
    ax = fig.add_subplot(gs[1, 0])
    ax_hist_vmax = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_hist_c = fig.add_subplot(gs[1, 1], sharey=ax)

    # Collate data for bounds setting
    xdata = [hermeian_vmax[hermeian_select], l_hermeian_vmax]
    ydata = [hermeian_cnfw[hermeian_select], l_hermeian_cnfw]
    markers = [hermeian_style['marker'], l_hermeian_style['marker']]
    colors = [hermeian_style['color'], l_hermeian_style['color']]
    all_x = np.concatenate((np.concatenate(xdata), field_vmax[field_select]))
    all_y = np.concatenate((np.concatenate(ydata), field_cnfw[field_select]))

    # Set x/y limits
    xlims = [
        np.nanmin(all_x) * (1. - bound_adjust),
        np.nanmax(all_x) * (1. + bound_adjust)
    ]
    ylims = [0, np.nanmax(all_y) * (1. + bound_adjust)]

    # Apply kde to haloes
    mi, ci = np.mgrid[0.01:2:n_bins * 1j, 0:100:n_bins * 1j]
    field_zi, field_xbins, field_ybins = generate_kde_density_values(
        np.log10(field_vmax[field_select]), field_cnfw[field_select], mi, ci,
        n_bins)
    bsplash_zi, bsplash_xbins, bsplash_ybins = generate_kde_density_values(
        np.log10(bsplash_vmax[bsplash_select]), bsplash_cnfw[bsplash_select],
        mi, ci, n_bins)
    (bsplash_2_zi, bsplash_2_xbins,
     bsplash_2_ybins) = generate_kde_density_values(
         np.log10(bsplash_vmax[bsplash_2_select]),
         bsplash_cnfw[bsplash_2_select], mi, ci, n_bins)

    ax.set(xscale='log')

    ####################################################################
    # Invisible/unlabelled contour lines
    ####################################################################
    # Field haloes
    blank_clevels = ax.contour(mi,
                               ci,
                               field_zi.reshape(mi.shape),
                               specify_contour_levels,
                               extent=[
                                   10**field_xbins.min(),
                                   10**field_xbins.max(),
                                   field_ybins.min(),
                                   field_ybins.max()
                               ],
                               linewidths=3,
                               colors=field_halo_style['color'],
                               alpha=0.,
                               linestyles='solid')
    # Backsplash haloes
    blank_bsplash_clevels = ax.contour(mi,
                                       ci,
                                       bsplash_zi.reshape(mi.shape),
                                       specify_contour_levels,
                                       extent=[
                                           10**bsplash_xbins.min(),
                                           10**bsplash_xbins.max(),
                                           bsplash_ybins.min(),
                                           bsplash_ybins.max()
                                       ],
                                       linewidths=3,
                                       colors=bsplash_style['color'],
                                       alpha=0.,
                                       linestyles='dashed')

    ####################################################################
    # Contour lines/regions (lines to be labelled)
    ####################################################################
    # Field haloes
    field_clevels = ax.contour(10**mi,
                               ci,
                               field_zi.reshape(mi.shape),
                               specify_contour_levels,
                               extent=[
                                   10**field_xbins.min(),
                                   10**field_xbins.max(),
                                   field_ybins.min(),
                                   field_ybins.max()
                               ],
                               linewidths=1,
                               colors=field_halo_style['color'],
                               linestyles='solid')
    # Backsplash haloes
    bsplash_clevels = ax.contour(10**mi,
                                 ci,
                                 bsplash_zi.reshape(mi.shape),
                                 specify_contour_levels,
                                 extent=[
                                     10**bsplash_xbins.min(),
                                     10**bsplash_xbins.max(),
                                     bsplash_ybins.min(),
                                     bsplash_ybins.max()
                                 ],
                                 linewidths=1,
                                 colors=bsplash_style['color'],
                                 linestyles='dashed')
    # Backsplash haloes (N_peri == 2 only)
    bsplash_2_clevels = ax.contourf(10**mi,
                                    ci,
                                    bsplash_2_zi.reshape(mi.shape),
                                    specify_contour_levels[:-1],
                                    extent=[
                                        10**bsplash_2_xbins.min(),
                                        10**bsplash_2_xbins.max(),
                                        bsplash_2_ybins.min(),
                                        bsplash_2_ybins.max()
                                    ],
                                    linewidths=1,
                                    cmap=bsplash_2_style['cmap'],
                                    alpha=0.2)

    ####################################################################
    # Label contour lines
    ####################################################################
    # Field contour lines
    ax.clabel(field_clevels,
              inline=True,
              fontsize=10,
              colors='k',
              fmt="%1.2f",
              manual=manual_contour_label_locations,
              inline_spacing=1)

    # Backsplash contour lines
    ax.clabel(bsplash_clevels,
              inline=True,
              fontsize=10,
              colors='k',
              fmt="%1.2f",
              manual=bsplash_manual_label_locations,
              inline_spacing=1)

    ####################################################################
    # Scatter plot haloes outside respective contour lines
    # (A bit hacky but works for illustrative purposes.
    # Full data available in the repo..)
    ####################################################################
    # Field haloes
    p = blank_clevels.collections[-1].get_paths()
    inside = np.full_like(field_cnfw[field_select], False, dtype=bool)
    for level in p:
        copy_level = copy.deepcopy(level)
        copy_vertices = copy_level._vertices
        end_vertex = [0, 0]
        copy_level._vertices = np.row_stack((copy_vertices, end_vertex))
        inside |= copy_level.contains_points(
            np.column_stack((np.log10(field_vmax[field_select]),
                             field_cnfw[field_select])))
    # Scatter field haloes outside highest contour
    ax.scatter(field_vmax[field_select][~inside],
               field_cnfw[field_select][~inside],
               color=field_halo_style['color'],
               marker=field_halo_style['marker'],
               label="Field")

    # Backsplash haloes
    p = blank_bsplash_clevels.collections[-1].get_paths()
    inside = np.full_like(bsplash_cnfw[bsplash_select], False, dtype=bool)
    for level in p:
        copy_level = copy.deepcopy(level)
        copy_vertices = copy_level._vertices
        end_vertex = [0, 0]
        copy_level._vertices = np.row_stack((copy_vertices, end_vertex))
        inside |= copy_level.contains_points(
            np.column_stack((np.log10(bsplash_vmax[bsplash_select]),
                             bsplash_cnfw[bsplash_select])))
    # Scatter backsplash haloes outside highest contour
    ax.scatter(bsplash_vmax[bsplash_select][~inside],
               bsplash_cnfw[bsplash_select][~inside],
               s=20,
               color=bsplash_style['color'],
               marker=bsplash_style['marker'],
               label="Backsplash")

    ax_hist_vmax.tick_params(axis="x", labelbottom=False)
    ax_hist_c.tick_params(axis="y", labelleft=False)

    ####################################################################
    # Plot histograms in the side panels
    ####################################################################
    # Field
    panel_hist(field_vmax[field_select],
               field_cnfw[field_select],
               ax_hist_vmax,
               ax_hist_c,
               yselect=(field_vmax[field_select] <= vmax_cut),
               color=field_halo_style['color'],
               xtype='log')
    # Backsplash
    panel_hist(bsplash_vmax[bsplash_select],
               bsplash_cnfw[bsplash_select],
               ax_hist_vmax,
               ax_hist_c,
               yselect=(bsplash_vmax[bsplash_select] <= vmax_cut),
               color=bsplash_style['color'],
               xtype='log',
               ls='--',
               lw=3)
    # Hermeian
    panel_hist(np.concatenate(
        (hermeian_vmax[hermeian_select], l_hermeian_vmax)),
               np.concatenate(
                   (hermeian_cnfw[hermeian_select], l_hermeian_cnfw)),
               ax_hist_vmax,
               ax_hist_c,
               color=hermeian_style['color'],
               xtype='log',
               lw=3)
    # Hermeian galaxies
    multi_scatter_plot(ax,
                       xdata,
                       ydata,
                       markers,
                       colors,
                       xtype='lin',
                       ytype='lin',
                       label_data=['Hermeian', None],
                       outline=[True, True])

    # Locations of Hermeian galaxy markers in the side panels
    mid_x = np.sum(ax_hist_c.get_xlim()) / 2.
    mid_y = 10**(np.sum(np.log10(ax_hist_vmax.get_ylim())) / 2.)

    for l_hermeian_vpeak_it, l_herm_c, color, l_marker in zip(
            l_hermeian_vmax, l_hermeian_cnfw, l_hermeian_style['color'],
            l_hermeian_style['marker']):
        ax_hist_vmax.axvline(l_hermeian_vpeak_it,
                             color=color,
                             linestyle=':',
                             zorder=49)
        ax_hist_c.axhline(l_herm_c, color=color, linestyle=':', zorder=49)
        ax_hist_vmax.scatter(l_hermeian_vpeak_it,
                             mid_y,
                             color=color,
                             marker=l_marker,
                             lw=0.5,
                             edgecolors='k',
                             zorder=50)
        ax_hist_c.scatter(mid_x,
                          l_herm_c,
                          color=color,
                          marker=l_marker,
                          lw=0.5,
                          edgecolors='k',
                          zorder=50)

    ####################################################################
    # Plot settings
    ####################################################################
    # Main panel
    ax.set(xlabel=r"$V_{\rm max}\, \left({\rm km\, s^{-1}}\right)$",
           ylabel=r"$c_{200}$",
           xlim=xlims,
           ylim=ylims)
    ax.tick_params(axis='x', which='major', pad=7)

    # Side panels
    ax_hist_vmax.set(ylabel=r"$dP\, /\, d\log V_{\rm max}$")
    ax_hist_vmax.yaxis.set_major_locator(LogLocator(numticks=4))
    ax_hist_c.set(xlabel=r"$dP\, /\, dc_{200}$")
    ax_hist_c.minorticks_on()
    ax_hist_c.tick_params(axis='x', which='major', pad=7)

    ####################################################################
    # Calculate/plot median values of the distributions
    ####################################################################
    field_med_c = np.nanmedian(
        field_cnfw[field_select][field_vmax[field_select] <= vmax_cut])
    bsplash_med_c = np.nanmedian(
        bsplash_cnfw[bsplash_select][bsplash_vmax[bsplash_select] <= vmax_cut])
    bsplash_2_med_c = np.nanmedian(
        bsplash_cnfw[bsplash_select *
                     (bsplash_nperi
                      == 2)][bsplash_vmax[bsplash_select *
                                          (bsplash_nperi == 2)] <= vmax_cut])
    hermeian_med_c = np.nanmedian(
        np.concatenate((hermeian_cnfw[hermeian_select], l_hermeian_cnfw)))
    med_h_med_f = hermeian_med_c / field_med_c
    med_h_med_b = hermeian_med_c / bsplash_med_c
    med_h_med_b_2 = hermeian_med_c / bsplash_2_med_c

    print("")
    print("Field median c_200: {}".format(field_med_c))
    print("Backsplash median c_200: {}".format(bsplash_med_c))
    print("Backsplash (N=2) median c_200: {}".format(bsplash_2_med_c))
    print("Hermeian median c_200: {}".format(hermeian_med_c))
    print("Median Hermeian c / Median Field c: {}".format(med_h_med_f))
    print("Median Hermeian c / Median Backsplash c: {}".format(med_h_med_b))
    print("Median Hermeian c / Median Backsplash (N=2) c: {}".format(
        med_h_med_b_2))
    print("")

    panel_arrow(ax_hist_c, field_med_c, field_halo_style['color'])
    panel_arrow(ax_hist_c, bsplash_med_c, bsplash_style['color'])
    panel_arrow(ax_hist_c,
                bsplash_2_med_c,
                plt.cm.get_cmap(bsplash_2_style['cmap'])(0.5),
                outline=True)
    panel_arrow(ax_hist_c, hermeian_med_c, hermeian_style['color'])

    ax.legend(frameon=True, fancybox=True, framealpha=0.75)

    save_figures(fig, c_vs_vmax_plot)

    ####################################################################
    # K-S Testing
    ####################################################################
    srt_lg_cnfw = np.sort(field_cnfw[field_select])
    lg_cdf = np.ones(len(srt_lg_cnfw)).cumsum()
    lg_cdf /= lg_cdf[-1]
    lg_cnfw_cdf_func = interp1d(srt_lg_cnfw, lg_cdf)

    srt_bsplash_cnfw = np.sort(bsplash_cnfw[bsplash_select])
    bsplash_cdf = np.ones(len(srt_bsplash_cnfw)).cumsum()
    bsplash_cdf /= bsplash_cdf[-1]

    srt_bsplash_2_cnfw = np.sort(bsplash_cnfw[bsplash_select *
                                              (bsplash_nperi == 2)])
    bsplash_2_cdf = np.ones(len(srt_bsplash_2_cnfw)).cumsum()
    bsplash_2_cdf /= bsplash_2_cdf[-1]

    bsplash_cnfw_cdf_func = interp1d(srt_bsplash_cnfw, bsplash_cdf)
    bsplash_2_cnfw_cdf_func = interp1d(srt_bsplash_2_cnfw,
                                       bsplash_2_cdf,
                                       fill_value=(0., 1.),
                                       bounds_error=False)

    print("Total pop drawn from LG distribution?")
    print(kstest(hermeian_cnfw[hermeian_select], lg_cnfw_cdf_func))
    print("Total pop drawn from backsplash distribution?")
    print(kstest(hermeian_cnfw[hermeian_select], bsplash_cnfw_cdf_func))
    print("Total pop drawn from backsplash (N=2) distribution?")
    print(kstest(hermeian_cnfw[hermeian_select], bsplash_2_cnfw_cdf_func))

    return None


def generate_kde_density_values(x, y, xi, yi, n_bins):
    xi, yi = np.mgrid[0.01:2:n_bins * 1j, 0:100:n_bins * 1j]

    k = kde.gaussian_kde((x, y))
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = (zi - zi.min()) / (zi.max() - zi.min())
    zi = zi.reshape(xi.shape)
    zi = 1. - zi

    counts, xbins, ybins = np.histogram2d(x, y, bins=50)
    return zi, xbins, ybins


def multi_scatter_plot(ax,
                       xdata,
                       ydata,
                       markers,
                       colors,
                       xtype='log',
                       ytype='log',
                       label_data=None,
                       outline=None):
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
                               s=s)
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
                           s=s)
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
                               s=s)
            else:
                if marker == 's':
                    s = small_size
                else:
                    s = normal_size
                ax.scatter(x, y, marker=marker, color=color, label=label, s=s)

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


def nearest_half_decade(x, direction='down', adjust=0.1, scale_type='log'):
    """Finds the value of the nearest half-decade in a given direction.

    Args:
        x (fl): Scalar value from which to find the floor or ceiling.
        direction (str, optional): Either 'up' or 'down'. Indicates the
            direction that the function works in. Defaults to 'down'.
        adjust (fl, optional): The fraction by which to adjust ret_val
            if x == ret_val. Defaults to 0.02.

    Returns:
        fl: Nearest half-decade in a given direction.
    """
    # Validation
    if direction not in ['up', 'down', 'u', 'd']:
        raise ValueError("'direction' must be either 'up' or 'down'." +
                         "Current value: {}".format(direction))

    x = x[(x == x) * (~np.isinf(x))]

    if direction in ['down', 'd']:
        if scale_type == 'log':
            dec = 10**np.floor(np.log10(x))
        else:
            dec = np.floor(x)
        half_dec = 5. * dec

        diff_dec = x - dec
        diff_hdec = x - half_dec

        if (x > half_dec) and (diff_hdec < diff_dec):
            ret_val = half_dec
        else:
            ret_val = dec

        if x == ret_val:
            ret_val *= (1. - adjust)

    else:
        if scale_type == 'log':
            dec = 10**np.ceil(np.log10(x))
        else:
            dec = np.ceil(x)
        half_dec = dec / 2.

        diff_dec = dec - x
        diff_hdec = half_dec - x

        if (x < half_dec) and (diff_hdec < diff_dec):
            ret_val = half_dec
        else:
            ret_val = dec

        if x == ret_val:
            ret_val *= (1. + adjust)

    # Handle x==0 exception
    if x == 0:
        ret_val = np.nan

    return ret_val


def panel_arrow(ax, loc, color, outline=False):
    if outline:
        ax.arrow(0.005,
                 loc,
                 -0.005,
                 0.,
                 width=1.,
                 length_includes_head=True,
                 head_length=0.003,
                 ec='k',
                 linestyle='--',
                 fc=color)
    else:
        ax.arrow(0.005,
                 loc,
                 -0.005,
                 0.,
                 width=1.,
                 length_includes_head=True,
                 head_length=0.003,
                 ec='none',
                 fc=color)
    return None


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
               ls=None):

    if xselect is not None:
        x = x[xselect]
    if yselect is not None:
        y = y[yselect]

    x_n_bins = int(2 * len(x)**(1. / 3.))
    y_n_bins = int(2 * len(y)**(1. / 3.))

    if xtype == 'log':
        x_bins = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)),
                             x_n_bins + 1)
        set_x_log = True
    else:
        x_bins = x_n_bins
        set_x_log = False

    if ytype == 'log':
        y_bins = np.logspace(np.log10(np.nanmin(y)), np.log10(np.nanmax(y)),
                             y_n_bins + 1)
        set_y_log = True
    else:
        y_bins = y_n_bins
        set_y_log = False

    ax_histx.hist(x,
                  bins=x_bins,
                  histtype='step',
                  density=True,
                  color=color,
                  log=set_x_log,
                  linestyle=ls,
                  lw=lw)
    ax_histy.hist(y,
                  bins=y_bins,
                  orientation='horizontal',
                  histtype='step',
                  density=True,
                  color=color,
                  log=set_y_log,
                  linestyle=ls,
                  lw=lw)
    return None


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
