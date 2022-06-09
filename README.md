# Plot scripts for the HESTIA Hermeian halo paper

**Last reviewed:** v1.1

A set of scripts and a data repository to reproduce the plots in the HESTIA
Hermeian halo discovery paper.

This README file is located in the main repository for the plotting scripts.
All plotting scripts should be executed from this directory.

## 1.0 Scripts

There are six scripts that can be executed independently:

* Fig. 1: [plot_dist_from_hosts.py](/plot_dist_from_hosts.py)
  * Plots r relative to both primary haloes as a function of lookback time
    for the Hermeian haloes.
* Figs. 2 & 3: [plot_geometric_properties.py](/plot_geometric_properties.py)
  * Plots the angular and radial distributions of the halo populations relative
  to the midpoint of the line connecting the Milky Way and M31.
  * Plots an Aitoff projection of the Hermeian haloes relative to the midpoint
  of the Milky Way&ndash;M31 line.
* Fig. 4: [plot_c_vs_vmax.py](/plot_c_vs_vmax.py)
  * Plots concentration vs. V<sub>max</sub> for each field halo population.
* Fig. 5: [plot_trajectory.py](/plot_trajectory.py)
  * Plots the trajectories of the Milky Way and M31 and two interacting
  Hermeian galaxies projected into the x-y plane.
* Fig. 6: [plot_jfactor_data.py](/plot_jfactor_data.py)
  * Plots the *J*-factor of the Hermeian haloes as a function of the distance
  from the MW analogue and angular extent of the photon emission region.
* Fig. A1: [plot_jfactors_all_types.py](/plot_jfactors_all_types.py)
  * Plots the *J*-factors of the Hermeian, backsplash, and field haloes as a
  function of the distance from the MW analogue, binned by halo mass.

There is also a master script, [plot_paper_plots.py](/plot_paper_plots.py),
that will run all of the above scripts when executed. This produces .svg
and .pdf versions of each figure in the paper.

## 2.0 Data

The [data](/data) directory that contains all files necessary to reproduce the
figures in the paper. There are five files:

* [17_11_hermeian_r_host_info.hdf5](/data/17_11_hermeian_r_host_info.hdf5)
  * Only required for Fig. 1
* [17_11_z0_data.hdf5](/data/17_11_z0_data.hdf5)
  * Required for Figs. 2&ndash;4, 6 and A1.
* [ludlow2014_logc_vs_logm200h.csv](/data/ludlow2014_logc_vs_logm200h.csv)
  * Required for Figs. 4 \& A1.
* [halo_positions.hdf5](/data/halo_positions.hdf5)
  * Only required for Fig. 5.
* [gammaldi_2021_data.hdf5](/data/gammaldi_2021_data.hdf5)
  * Only required for Fig. 6.

## 3.0 Citations

This code and the accompanying data are freely available.

### If you use this code or derivative work

* [O. Newton et al. (2022)](https://doi.org/10.1093/mnras/stac1316)
* [O. Newton (2021)](https://doi.org/10.5281/zenodo.4708338)

### If you use these data, a derivative work, or results thereof

* [O. Newton et al. (2022)](https://doi.org/10.1093/mnras/stac1316)
* [O. Newton (2021)](https://doi.org/10.5281/zenodo.4708338)
* [N. Libeskind et al. (2020)](https://doi.org/10.1093/mnras/staa2541)

If you have any questions or would like help in using the scripts, please
email:
> o 'dot' j 'dot' newton 'at' ljmu.ac.uk
