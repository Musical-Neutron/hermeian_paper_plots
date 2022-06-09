#!/usr/bin/env python3

# Place import files below
import plot_c_vs_vmax
import plot_dist_from_hosts
import plot_geometric_properties
import plot_jfactor_data
import plot_jfactors_all_types
import plot_trajectory


def main():
    # Plot Fig. 1
    plot_dist_from_hosts.main()

    # Plot Figs. 2 & 3
    plot_geometric_properties.main()

    # Plot Fig. 4
    plot_c_vs_vmax.main()

    # Plot Fig. 5
    plot_trajectory.main()

    # Plot Fig. 6
    plot_jfactor_data.main()

    # Plot Fig. A1
    plot_jfactors_all_types.main()

    return None


if __name__ == "__main__":
    main()
