#!/usr/bin/env python3

# Place import files below
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from common_functions import save_figures


def main():
    # Plotting settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    host_lstyle = {
        '127000000000003': {
            'linestyle': '-'
        },
        '127000000000002': {
            'linestyle': '--'
        }
    }
    hermeian_style = {
        '127000000000277': {
            'color': 'orange',
            'marker': 'd'
        },
        '127000000000789': {
            'color': 'magenta',
            'marker': '*'
        }
    }
    hermeian_interaction_snaps = {
        '127000000000277': [86, 110],
        '127000000000789': [84, 107]
    }

    # Simulation information
    h = 0.677

    # Data settings
    host_ids = ['127000000000002', '127000000000003']
    hermeian_ids = ['127000000000277', '127000000000789']
    draw_radius = [87, 109]

    # File locations
    halo_positions_file = os.path.join('data', 'halo_positions.hdf5')
    plot_file = 'traj_xy_gas_transfer.pdf'

    ####################################################################
    # Initialize figures
    ####################################################################
    traj_xy_fig = plt.figure(figsize=(8, 8))
    traj_xy_ax = traj_xy_fig.add_subplot(111)

    all_x = []
    all_y = []
    host_lines = []
    hermeian_lines = []

    ####################################################################
    # Read and plot host data
    ####################################################################
    for i, (h_id, s_id) in enumerate(zip(host_ids, draw_radius)):
        with h5py.File(halo_positions_file, 'r') as halo_position_data:
            snap_id = halo_position_data[h_id][:, 0]
            host_z = halo_position_data[h_id][:, 1]
            host_r200 = halo_position_data[h_id][:, 3]
            host_position = halo_position_data[h_id][:, [4, 5, 6]]

        c_host_pos = (host_position * h * (1. + host_z)[:, np.newaxis]
                      )  # Co-moving

        xy_line, = traj_xy_ax.plot(c_host_pos[:, 0],
                                   c_host_pos[:, 1],
                                   color='k',
                                   lw=6,
                                   **host_lstyle[h_id],
                                   zorder=0)

        host_lines.append(xy_line)

        for h_i_key in hermeian_interaction_snaps:
            if i == 0:
                s_id = hermeian_interaction_snaps[h_i_key][0]
            else:
                s_id = hermeian_interaction_snaps[h_i_key][1]
            select_snap = snap_id == s_id
            draw_object = plt.Circle((c_host_pos[:, 0][select_snap][0],
                                      c_host_pos[:, 1][select_snap][0]),
                                     host_r200[select_snap][0],
                                     color='k',
                                     fill=False)
            traj_xy_ax.add_patch(draw_object)

            all_x.append(c_host_pos[:, 0][select_snap][0] +
                         host_r200[select_snap][0])
            all_x.append(c_host_pos[:, 0][select_snap][0] -
                         host_r200[select_snap][0])
            all_y.append(c_host_pos[:, 1][select_snap][0] +
                         host_r200[select_snap][0])
            all_y.append(c_host_pos[:, 1][select_snap][0] -
                         host_r200[select_snap][0])

    ####################################################################
    # Read and plot Hermeian data
    for i, (h_id, s_id) in enumerate(zip(hermeian_ids, draw_radius)):
        with h5py.File(halo_positions_file, 'r') as halo_position_data:
            snap_id = halo_position_data[h_id][:, 0]
            host_z = halo_position_data[h_id][:, 1]
            host_r200 = halo_position_data[h_id][:, 3]
            host_position = halo_position_data[h_id][:, [4, 5, 6]]

        c_host_pos = (host_position * h * (1. + host_z)[:, np.newaxis]
                      )  # Co-moving

        xy_line, = traj_xy_ax.plot(c_host_pos[:, 0],
                                   c_host_pos[:, 1],
                                   alpha=1,
                                   zorder=100,
                                   markersize=11,
                                   lw=3,
                                   markeredgewidth=0.5,
                                   markeredgecolor='k',
                                   markevery=slice(10 * i, len(snap_id), 15),
                                   **hermeian_style[h_id])

        hermeian_lines.append(xy_line)

        ################################################################
        # Draw halo boundaries at selected interaction snapshots
        for s_i, s_id in enumerate(hermeian_interaction_snaps[h_id]):
            select_snap = snap_id == (s_id + (s_i - 1))
            draw_object = plt.Circle((c_host_pos[:, 0][select_snap][0],
                                      c_host_pos[:, 1][select_snap][0]),
                                     host_r200[select_snap][0],
                                     color=hermeian_style[h_id]['color'],
                                     fill=False)
            traj_xy_ax.add_patch(draw_object)

            all_x.append(c_host_pos[:, 0][select_snap][0] +
                         host_r200[select_snap][0])
            all_x.append(c_host_pos[:, 0][select_snap][0] -
                         host_r200[select_snap][0])
            all_y.append(c_host_pos[:, 1][select_snap][0] +
                         host_r200[select_snap][0])
            all_y.append(c_host_pos[:, 1][select_snap][0] -
                         host_r200[select_snap][0])

    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)

    xlim = np.asarray([np.nanmin(all_x), np.nanmax(all_x)]) * [0.995, 1.005]
    ylim = np.asarray([np.nanmin(all_y), np.nanmax(all_y)]) * [0.995, 1.005]

    ####################################################################
    # Add arrows to trajectory lines
    for line_item, lim in zip(host_lines,
                              [xlim, xlim, ylim, xlim, xlim, ylim]):
        add_arrow(line_item, size=50, position=lim.sum() / 2.)
    for line_item, lim in zip(hermeian_lines,
                              [xlim, xlim, ylim, xlim, xlim, ylim]):
        add_arrow(line_item, size=30, position=lim.sum() / 2.)

    ####################################################################
    # Add annotations to figure
    ####################################################################
    traj_xy_ax.annotate(r"$z=0.786$",
                        xy=(50.5, 44.38),
                        xycoords='data',
                        color='k')
    traj_xy_ax.annotate(r"$z=0.731$",
                        xy=(50.3, 44.49),
                        xycoords='data',
                        color='k')
    traj_xy_ax.annotate(r"Gas accretion",
                        xytext=(49.5, 44.5),
                        xy=(50.75, 44.85),
                        xycoords='data',
                        color='k',
                        arrowprops=dict(arrowstyle='-|>', color='k'))
    traj_xy_ax.annotate(r"$z=0.275$",
                        xy=(50.05, 46.2),
                        xycoords='data',
                        color='k')
    traj_xy_ax.annotate(r"$z=0.230$",
                        xy=(49.6, 46.68),
                        xycoords='data',
                        color='k')
    traj_xy_ax.annotate("Gas deposition and\nstar formation",
                        xytext=(50.5, 46.53),
                        xy=(49.83, 46.19),
                        xycoords='data',
                        color='k',
                        horizontalalignment='center',
                        arrowprops=dict(arrowstyle='-|>', color='k'))
    traj_xy_ax.annotate(r"MW", xy=(51, 45.5), xycoords='data', color='k')
    traj_xy_ax.annotate(r"M31",
                        xy=(49.75, 45.5),
                        xycoords='data',
                        ha='right',
                        color='k')

    ####################################################################
    # Figure settings
    ####################################################################
    traj_xy_ax.set(xlabel=r'X $\left({\rm cMpc}\, /\, h\right)$',
                   ylabel=r'Y $\left({\rm cMpc}\, /\, h\right)$',
                   xlim=xlim,
                   ylim=ylim)

    traj_xy_ax.minorticks_on()
    traj_xy_ax.set_aspect('equal')

    save_figures(traj_xy_fig, plot_file)

    return None


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="-|>", color=color),
                       size=size)


if __name__ == "__main__":
    main()
