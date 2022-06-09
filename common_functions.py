#!/usr/bin/env python3

# Place import files below
import os

import numpy as np
from scipy.optimize import bisect


def c_exp(c):
    """NFW concentration function.

    Args:
        c (fl/arr): Halo concentration parameter.

    Returns:
        fl/arr: Value(s) of the NFW concentration function.
    """
    return np.log(1. + c) - (c / (1. + c))


def embed_symbols(pdf_file):
    """Embeds symobls in pdf files.

    Args:
        pdf_file (str): Filepath to the file

    Returns:
        None
    """
    os.system('gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress '
              '-dEmbedAllFonts=true -sOutputFile={} -f {}'.format(
                  pdf_file.replace('.pdf', '_embedded.pdf'), pdf_file))
    return None


def nearest_half_decade(x, direction='down', adjust=0.1, scale_type='log'):
    """Finds the value of the nearest half-decade in a given direction.

    Args:
        x (fl): Scalar value from which to find the floor or ceiling.
        direction (str, optional): Either 'up' or 'down'. Indicates the
            direction that the function works in. Defaults to 'down'.
        adjust (fl, optional): The fraction by which to adjust ret_val
            if x == ret_val. Defaults to 0.1.

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


def save_figures(fig, location, embed=False):
    """Saves svg and pdf versions of figures.

    Args:
        fig (Matplotlib figure object): The figure to save
        location (str): Filepath to the save file
        embed (bool): If True, embeds the symbols in the pdf file.
            Default: False.

    Returns:
        None
    """
    if '.pdf' in location:
        pdf_file = location
        svg_file = location.replace('.pdf', '.svg')
    else:
        pdf_file = location + '.pdf'
        svg_file = location + '.svg'

    fig.savefig(pdf_file, dpi=600, format='pdf', transparent=False)
    fig.savefig(svg_file, dpi=600, format='svg', transparent=False)

    if embed:
        embed_symbols(pdf_file)

    return None


class Angle(object):
    degree_conversion_factors = {
        'degree': 1.,
        'radian': np.pi / 180.,
        'quadrant': 1. / 90.,
        'sextant': 1. / 60.,
        'octant': 1. / 45.,
        'hexacontade': 1. / 6.,
        'binary_degree': 256. / 360.,
        'gradian': 400. / 360.,
        'arcminute': 21600. / 360.,
        'arcsecond': 1296000. / 360.
    }

    def __init__(self, angles, fromunit='degree') -> None:
        self.angles = angles
        if fromunit not in Angle.degree_conversion_factors.keys():
            raise ValueError("fromunit must be one of: {}".format(
                Angle.degree_conversion_factors.keys()))
        self.fromunit = fromunit

        super().__init__()

    # Angle getter
    def get_angles(self):
        return self._angles

    # Fromunit getter
    def get_fromunit(self):
        return self._fromunit

    # Angle setter
    def set_angles(self, angles):
        # Initial type-checking
        angle_permitted_types = (float, list, np.ndarray)
        if not isinstance(angles, angle_permitted_types):
            raise TypeError(
                '"angles" must be one of: {}'.format(angle_permitted_types))
        self._angles = angles

        if hasattr(self, '_fromunit'):
            self.set_conversions(self._angles, self._fromunit)
        return None

    # Fromunit setter
    def set_fromunit(self, fromunit):
        # Initial type-checking
        fromunit_permitted_types = (str)
        if not isinstance(fromunit, fromunit_permitted_types):
            raise TypeError('"fromunit" must be one of: {}'.format(
                fromunit_permitted_types))
        self._fromunit = fromunit

        if hasattr(self, '_angles'):
            self.set_conversions(self._angles, self._fromunit)
        return None

    def set_conversions(self, angles, fromunit):
        # Initial type-checking
        if fromunit not in Angle.degree_conversion_factors.keys():
            raise ValueError('"fromunit" must be one of {}'.format(
                Angle.degree_conversion_factors.keys()))

        # Convert lists to np.ndarray
        if isinstance(angles, list):
            angles = np.asarray(angles)

        # self.fromunit = fromunit
        degrees = angles / Angle.degree_conversion_factors[fromunit]

        for key in Angle.degree_conversion_factors.keys():
            setattr(self, key, degrees * Angle.degree_conversion_factors[key])
        return None

    angles = property(get_angles, set_angles)
    fromunit = property(get_fromunit, set_fromunit)


class NFWHalo(object):
    def __init__(self, delta, concentration) -> None:
        """Basic class to hold information about NFW halo properties.

        Args:
            delta (fl): Density contrast of halo relative to rho_crit.
            concentration (fl): Concentration of halo.
        """
        self.delta = delta
        self.concentration = concentration

        super().__init__()

    def redefine_halo(self, new_delta):
        """Calculates the concentration of the NFW defined using a
            different density contrast.

        Args:
            new_delta (fl): Density contrast of halo relative to
                rho_crit.

        Returns:
            concentration: Concentration in the new halo definition.
        """
        return bisect(self.__redefinition_minimisation,
                      1,
                      200,
                      args=(new_delta))

    def __redefinition_minimisation(self, c_two, delta_two):
        numerator = ((np.log(1. + self.concentration) -
                      (self.concentration / (1. + self.concentration))) *
                     delta_two * c_two**3.)
        denominator = ((np.log(1. + c_two) - (c_two / (1. + c_two))) *
                       self.delta * self.concentration**3.)

        return (numerator / denominator) - 1.
