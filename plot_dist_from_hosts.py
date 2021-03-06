#!/usr/bin/env python3

# Place import files below
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from common_functions import save_figures


def main():
    # Plotting settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    xlim = [0, 12.]  # Gyr
    ylim = [0, 2000]  # kpc
    l_hermeian_style = {
        'color': ['orange', 'magenta', 'r', 'b'],
        'marker': ['d', '*', 'X', 'p']
    }
    hermeian_galaxy_style = {
        'alpha': 1,
        'zorder': 100,
        'lw': 3,
        'markersize': 11,
        'markeredgewidth': 0.5,
        'markeredgecolor': 'k'
    }
    dark_hermeian_style = {
        'color': 'silver',
        'alpha': 0.5,
        'zorder': 0,
        'lw': 1
    }

    # File locations
    dist_file = os.path.join('data', '17_11_hermeian_r_host_info.hdf5')
    orbits_plot = 'hermeian_orbits.pdf'

    ####################################################################
    # Read data
    ####################################################################
    with h5py.File(dist_file, 'r') as dist_data:
        main_halo_ids = np.uint64([key for key in dist_data.keys()])
        main_halo_ids = np.sort(main_halo_ids)

    ####################################################################
    # Plot distance to Hermeian from each respective host halo
    ####################################################################
    fig, axs = plt.subplots(1,
                            2,
                            sharey=True,
                            figsize=(16, 8),
                            gridspec_kw={
                                'hspace': 0,
                                'wspace': 0
                            })
    """Halo IDs are assigned from most-massive to least massive. By
        convention, the M31 analogue is chosen to be the more massive
        (i.e. lower ID) analogue. We want the MW analogue on the left,
        so reverse the ID order..."""
    for m_i, main_id in enumerate(main_halo_ids[::-1]):
        with h5py.File(dist_file, 'r') as dist_data:
            host_r_virial = dist_data['{}'.format(main_id)]['R_vir'][...]
            hermeian_tlb = dist_data['{}'.format(main_id)]['Distance'][:, 0]
            hermeian_dists = dist_data['{}'.format(main_id)]['Distance'][:, 1:]

        # Primary halo virial radius evolution
        axs[m_i].plot(host_r_virial[:, 0],
                      host_r_virial[:, 1],
                      linestyle=':',
                      color='k',
                      zorder=99)

        # Plot the dark Hermeian haloes
        for d_h_i in np.arange(4, len(hermeian_dists[0])):
            axs[m_i].plot(hermeian_tlb, hermeian_dists[:, d_h_i],
                          **dark_hermeian_style)

        # Plot the Hermeian galaxies (i.e. containing stars). Order is
        # from most-massive to least.
        for g_h_i in np.arange(4):
            axs[m_i].plot(hermeian_tlb,
                          hermeian_dists[:, g_h_i],
                          color=l_hermeian_style['color'][g_h_i],
                          marker=l_hermeian_style['marker'][g_h_i],
                          markevery=slice(10 * g_h_i, len(hermeian_tlb), 30),
                          **hermeian_galaxy_style)

    axs[0].minorticks_on()
    axs[1].minorticks_on()
    axs[0].set(xlabel=r'$t_{\rm lookback}\, \left({\rm Gyr}\right)$',
               ylabel=r'$r_{\rm host}\, \left({\rm kpc}\right)$',
               xlim=xlim,
               ylim=ylim,
               title=r"MW")
    axs[1].set(xlabel=r'$t_{\rm lookback}\, \left({\rm Gyr}\right)$',
               xlim=xlim,
               title=r"M31")

    # Avoid tick label overlap
    ax_nbins = len(axs[0].get_xticklabels())
    axs[0].xaxis.set_major_locator(MaxNLocator(nbins=ax_nbins, prune='upper'))

    save_figures(fig, orbits_plot)

    return None


if __name__ == "__main__":
    main()
