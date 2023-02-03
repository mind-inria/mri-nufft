"""Kernel function for GPUArray data."""
import numpy as np


def update_density(density, update):
    """Perform an element wise normalization.

    Parameters
    ----------
    density: array
    update: array complex

    Notes
    -----
    performs :math:`d / \|u\|_2` element wise.
    """
    density /= np.abs(update)
    return density


def sense_adj_mono(dest, coil, smap, **kwargs):
    """Perform a sense reduction for one coil.

    Parameters
    ----------
    dest: array
        The image to update with the sense updated data
    coil_img: array
        The coil image estimation
    smap: array
        The sensitivity profile of the coil.
    """
    dest += coil * smap.conjugate()
    return dest
