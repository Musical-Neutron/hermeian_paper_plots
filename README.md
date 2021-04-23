# Plot scripts for the HESTIA Hermeian halo paper
**Last reviewed:** v1.0

A set of scripts and a data repository to reproduce the plots in the HESTIA
Hermeian halo discovery paper.

This README file is located in the main repository for the plotting scripts.
All plotting scripts should be executed from this directory.

## 1.0 Scripts
There are four scripts that can be executed independently:
* Fig. 1: [plot_dist_from_hosts.py](/plot_dist_from_hosts.py)
  - Plots r relative to both primary haloes as a function of lookback time
    for the Hermeian haloes.
* Figs. 2 & 3: [plot_geometric_properties.py](/plot_geometric_properties.py)
  - Plots the angular and radial distributions of the halo populations relative
  to the midpoint of the line connecting the Milky Way and M31.
  - Plots an Aitoff projection of the Hermeian haloes relative to the midpoint
  of the Milky Way&ndash;M31 line.
* Fig. 4: [plot_c_vs_vmax.py](/plot_c_vs_vmax.py)
  - Plots concentration vs. V<sub>max</sub> for each field halo population.
* Fig. 5: [plot_jfactor_data.py](/plot_jfactor_data.py)
  - Plots the *J*-factor of the Hermeian haloes as a function of the distance
  from the MW analogue and angular extent of the photon emission region.

There is also a master script, [plot_paper_plots.py](/plot_paper_plots.py),
which will run all of the above scripts when executed. This will produce .svg
and .pdf versions of each figure in the paper.

## 2.0 Data
The [data](/data) directory that contains all files necessary to reproduce the
figures in the paper. There are two files:
* [17_11_hermeian_r_host_info.hdf5](/data/17_11_hermeian_r_host_info.hdf5)
  - Only required for Fig. 1
* [17_11_z0_data.hdf5](/data/17_11_z0_data.hdf5)
  - Required for Figs. 2&ndash;5.

## 3.0 Citations
This code and accompanying input data are freely available. If using this code,
the data, a derivative work, or results thereof, please cite:
* [Newton (2021)](http://doi.org/10.5281/zenodo.4708338)
> Will be updated upon acceptance
<!-- [Newton+(2021)](https://arxiv.org/abs/) -->

If you have any questions or would like help in using the code, please email:
> olivier 'dot' newton 'at' univ-lyon1.fr
