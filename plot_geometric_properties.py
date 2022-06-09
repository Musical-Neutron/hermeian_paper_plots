#!/usr/bin/env python3

# Place import files below
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from common_functions import save_figures, Angle


def main():
    # Plotting settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    fig_aspect_ratio = 1.

    field_halo_style = {'color': 'darkgrey', 'marker': '.'}
    backsplash_style = {'color': "#7E317B", 'marker': 's'}
    hermeian_style = {'color': 'cornflowerblue', 'marker': '^'}
    l_hermeian_style = {
        'color': np.asarray(['orange', 'magenta', 'r', 'b']),
        'marker': ['d', '*', 'X', 'p']
    }
    field_hist_style = {
        'ls': 'solid',
        'lw': 1,
        'color': field_halo_style['color'],
    }
    bsplash_hist_style = {
        'ls': 'dashed',
        'lw': 3,
        'color': backsplash_style['color'],
    }
    hermeian_hist_style = {
        'ls': 'solid',
        'lw': 3,
        'color': hermeian_style['color'],
    }
    histogram_leg_handles = [
        plt.Line2D((0, 1), (0, 0), **iter_dict) for iter_dict in
        [field_hist_style, bsplash_hist_style, hermeian_hist_style]
    ]
    histogram_leg_labels = ['Field', 'Backsplash', 'Hermeian']

    # Output location
    data_file = os.path.join('data', '17_11_z0_data.hdf5')
    midpoint_dSter_dR_plot = 'midpoint_steradian_dVol_histogram.pdf'
    lg_aitoff_hermeian_plot = 'lg_hermeian_aitoff_projection.pdf'

    # Read data
    with h5py.File(data_file, 'r') as data:
        primary_data = data['Main haloes'][...]
        bsplash_data = data['Backsplash'][...]
        field_data = data['Field'][...]
        hermeian_data = data['Hermeian'][...]

    select_l_herm = hermeian_data[:, 9].astype(bool)

    ####################################################################
    # Generate data
    ####################################################################
    # Distance from LG midpoint
    bsplash_dist_lg = bsplash_data[:, 1] * np.sign(bsplash_data[:, 2])
    field_dist_lg = field_data[:, 1] * np.sign(field_data[:, 2])
    dm_hermeian_dist_lg = (hermeian_data[:, 1] *
                           np.sign(hermeian_data[:, 2]))[~select_l_herm]
    l_hermeian_dist_lg = (hermeian_data[:, 1] *
                          np.sign(hermeian_data[:, 2]))[select_l_herm]
    primary_dist_lg = (primary_data[:, 1] * np.sign(primary_data[:, 2]))
    m31_dist = primary_dist_lg[0]
    mw_dist = primary_dist_lg[1]

    # Angles
    bsplash_angles = Angle(np.arccos(bsplash_data[:, 2]), fromunit='radian')
    field_angles = Angle(np.arccos(field_data[:, 2]), fromunit='radian')
    dm_hermeian_angles = Angle(np.arccos(hermeian_data[:, 2][~select_l_herm]),
                               fromunit='radian')
    l_hermeian_angles = Angle(np.arccos(hermeian_data[:, 2][select_l_herm]),
                              fromunit='radian')

    # Aitoff projection coordinates
    hermeian_l = hermeian_data[:, 5][~select_l_herm]
    l_hermeian_l = hermeian_data[:, 5][select_l_herm]
    primary_l = primary_data[:, 3]
    hermeian_b = hermeian_data[:, 6][~select_l_herm]
    l_hermeian_b = hermeian_data[:, 6][select_l_herm]
    primary_b = primary_data[:, 4]
    l_m31 = primary_l[0]
    l_mw = primary_l[1]
    b_m31 = primary_b[0]
    b_mw = primary_b[1]

    ####################################################################
    # Plot dSter and dR plots
    ####################################################################
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'hspace': 0})

    # dSter plot (left panel)
    plot_dn_dster_hist(axs[0], field_angles, field_hist_style)
    plot_dn_dster_hist(axs[0], bsplash_angles, bsplash_hist_style)
    plot_dn_dster_hist(axs[0], dm_hermeian_angles, hermeian_hist_style)

    axs[0].set(xlabel=r'$\theta_{\rm mid}\, \left(^\circ\right)$',
               ylabel=r'$dP\, /\, d\Omega_{\rm mid}$',
               xlim=[0, 180],
               ylim=[0, None])
    axs[0].xaxis.set_major_locator(MultipleLocator(20))
    axs[0].minorticks_on()

    mid_y = np.sum(axs[0].get_ylim()) / 2.
    for l_angle, color, marker in zip(l_hermeian_angles.degree,
                                      l_hermeian_style['color'],
                                      l_hermeian_style['marker']):
        axs[0].axvline(l_angle, color=color, linestyle=':', zorder=9)
        axs[0].scatter(l_angle,
                       mid_y,
                       color=color,
                       marker=marker,
                       s=200,
                       linewidth=1,
                       edgecolors='k',
                       zorder=5)

    # dR plot (right panel)
    plot_dn_dR_hist(axs[1], field_dist_lg, field_hist_style)
    plot_dn_dR_hist(axs[1], bsplash_dist_lg, bsplash_hist_style)
    plot_dn_dR_hist(axs[1], dm_hermeian_dist_lg, hermeian_hist_style)

    axs[1].set(xlabel=(
        r'$r_{\rm mid}\, \cdot $' +
        r'$\frac{\cos \theta_{\rm mid}}{\left|\cos \theta_{\rm mid}\right|}\,$'
        + r'$\left({\rm Mpc}\right)$'),
               ylabel=r'$dP\, /\, dV_{\rm mid}$',
               xlim=[-2.5, 2.5],
               ylim=[0, None])

    mid_y = np.sum(axs[1].get_ylim()) / 2.
    for l_dist, color, marker in zip(l_hermeian_dist_lg,
                                     l_hermeian_style['color'],
                                     l_hermeian_style['marker']):
        axs[1].axvline(l_dist, color=color, linestyle=':', zorder=5)
        axs[1].scatter(l_dist,
                       mid_y,
                       color=color,
                       marker=marker,
                       s=200,
                       linewidth=1,
                       edgecolors='k',
                       zorder=10)

    axs[1].arrow(m31_dist,
                 0.125,
                 0.,
                 -0.125,
                 width=0.05,
                 lw=2,
                 length_includes_head=True,
                 head_length=0.07,
                 ec='k',
                 fc='none')
    axs[1].arrow(mw_dist,
                 0.125,
                 0.,
                 -0.125,
                 width=0.05,
                 length_includes_head=True,
                 head_length=0.07,
                 ec='none',
                 fc='k')

    leg_obj = axs[1].legend(
        histogram_leg_handles,
        histogram_leg_labels,
        loc='upper right',
    )
    axs[0].set_aspect(1. / axs[0].get_data_ratio() * fig_aspect_ratio)
    axs[1].set_aspect(1. / axs[1].get_data_ratio() * fig_aspect_ratio)
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1].minorticks_on()
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.set_tick_params(labelright=True, labelleft=False)
    leg_obj.set_zorder(25)

    axs[0].annotate('M31', (0, 1), (-15, 20),
                    'axes fraction',
                    textcoords='offset points',
                    va='top')
    axs[0].annotate('MW', (1, 1), (-15, 20),
                    'axes fraction',
                    textcoords='offset points',
                    va='top')

    axs[1].annotate('MW', (0, 0), (-30, -5),
                    'data',
                    textcoords='offset points',
                    va='top',
                    ha='center')
    axs[1].annotate('M31', (0, 0), (30, -5),
                    'data',
                    textcoords='offset points',
                    va='top',
                    ha='center')

    save_figures(fig, midpoint_dSter_dR_plot)

    ####################################################################
    # LG Aitoff plot
    ####################################################################
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='aitoff')
    ax.set_longitude_grid_ends(90)
    ax.set(xlabel=r'$l\, \left(^\circ\right)$',
           ylabel=r'$b\, \left(^\circ\right)$')
    ax.scatter(l_mw,
               b_mw,
               s=30,
               marker='s',
               edgecolors='none',
               facecolors='k',
               zorder=100)
    ax.scatter(l_m31,
               b_m31,
               s=30,
               marker='s',
               edgecolors='k',
               facecolors='none',
               zorder=100)
    ax.scatter(hermeian_l,
               hermeian_b,
               s=40,
               linewidth=0.5,
               edgecolors='k',
               zorder=99,
               **hermeian_style)
    for l_angle, b_angle, color, marker in zip(l_hermeian_l, l_hermeian_b,
                                               l_hermeian_style['color'],
                                               l_hermeian_style['marker']):
        ax.scatter(l_angle,
                   b_angle,
                   s=60,
                   color=color,
                   marker=marker,
                   zorder=100,
                   linewidth=0.3,
                   edgecolors='k')

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, ls='solid', lw=1, color='gray')
    ax.set_axisbelow(True)

    save_figures(fig, lg_aitoff_hermeian_plot)

    return None


def plot_dn_dR_hist(ax, distances, style_dict):
    """Plot dP/dVolume_mid

    Args:
        ax (Axes object): Axis on which to plot the histogram.
        distances (np.ndarray): Data to plot.
        style_dict (dict): Dict containing the linestyle information and
            label (if given).

    Returns:
        None
    """
    # Compute volume information
    bins = int(2 * len(distances)**(1. / 3.))
    dR = 5. / bins
    # dR = 2.5 / bins
    bin_edges = np.arange(bins + 1, dtype=float) * dR - 2.5
    # bin_edges = np.arange(bins + 1, dtype=float) * dR
    if 0 in bin_edges:
        zero_idx = np.argwhere(bin_edges == 0)[0][0]
        right_bins = bin_edges[zero_idx:]
        dVolume = 2. * np.pi * (right_bins[1:]**3 - right_bins[:-1]**3) / 3.

        dVolume = np.concatenate((dVolume[::-1], dVolume))

    else:
        right_bins = bin_edges[int(len(bin_edges) / 2):]
        dVolume = 2. * np.pi * (right_bins**3. - np.concatenate(
            ([0.], right_bins[:-1]))**3.) / 3.
        # dVolume = 4. * np.pi * (bin_edges[1:]**3. - bin_edges[:-1]**3.) / 3.

        dVolume = np.concatenate(
            (dVolume[:0:-1], [dVolume[0] * 2.], dVolume[1:]))

    # Make histogram
    histogram = np.histogram(distances, bin_edges)

    ax.hist((bin_edges[1:] + bin_edges[:-1]) / 2,
            weights=histogram[0] / dVolume,
            bins=bin_edges,
            histtype='step',
            density=True,
            **style_dict)

    return None


def plot_dn_dster_hist(ax, angle_object, style_dict):
    """Plot dP/dVolume_mid

    Args:
        ax (Axes object): Axis on which to plot the histogram.
        angle_object (Angle): Data to plot.
        style_dict (dict): Dict containing the linestyle information and
            label (if given).

    Returns:
        None
    """
    # Compute steradian information
    bins = int(2 * len(angle_object.degree)**(1. / 3.))
    dtheta = 180 / bins
    bin_edges = np.arange(bins + 1, dtype=float) * dtheta
    dcolatitude = np.cos(Angle(bin_edges[:-1]).radian) - np.cos(
        Angle(bin_edges[1:]).radian)
    dVolume = 2. * np.pi * (2.5**3) * dcolatitude / 3.

    # Make histogram
    histogram = np.histogram(angle_object.degree, bin_edges)

    ax.hist((bin_edges[1:] + bin_edges[:-1]) / 2,
            weights=histogram[0] / dVolume,
            bins=bin_edges,
            histtype='step',
            density=True,
            **style_dict)

    return None


if __name__ == "__main__":
    main()
