"""Field map generator module."""

import numpy as np


def make_b0map(shape, b0range=(-300, 300), mask=None):
    """
    Make radial B0 map.

    Parameters
    ----------
    shape : tuple[int]
        Matrix size. Only supports isotropic matrices.
    b0range : tuple[float], optional
        Frequency shift range in [Hz]. The default is (-300, 300).
    mask : np.ndarray
        Spatial support of the objec. If not provided,
        build a radial mask with radius = 0.3 * shape

    Returns
    -------
    np.ndarray
        B0 map of shape (*shape) in [Hz],
        with values included in (*b0range).
    mask : np.ndarray, optional
        Spatial support binary mask.

    """
    assert np.unique(shape).size, ValueError("Only isotropic matriex are supported.")
    ndim = len(shape)
    if ndim == 2:
        radial_mask, fieldmap = _make_disk(shape)
    elif ndim == 3:
        radial_mask, fieldmap = _make_sphere(shape)
    if mask is None:
        mask = radial_mask

    # build map
    fieldmap *= mask
    fieldmap = (b0range[1] - b0range[0]) * fieldmap / fieldmap.max() + b0range[0]  # Hz
    fieldmap *= mask

    # remove nan
    fieldmap = np.nan_to_num(fieldmap, neginf=0.0, posinf=0.0)

    return fieldmap.astype(np.float32), mask


def make_t2smap(shape, t2svalue=15.0, mask=None):
    """
    Make homogeneous T2* map.

    Parameters
    ----------
    shape : tuple[int]
        Matrix size.
    t2svalue : float, optional
        Object T2* in [ms]. The default is 15.0.
    mask : np.ndarray
        Spatial support of the objec. If not provided,
        build a radial mask with radius = 0.3 * shape

    Returns
    -------
    np.ndarray
        T2* map of shape (*shape) in [ms].
    mask : np.ndarray, optional
        Spatial support binary mask.

    """
    assert np.unique(shape).size, ValueError("Only isotropic matriex are supported.")
    ndim = len(shape)
    if ndim == 2:
        radial_mask, fieldmap = _make_disk(shape)
    elif ndim == 3:
        radial_mask, fieldmap = _make_sphere(shape)
    if mask is None:
        mask = radial_mask

    # build map
    fieldmap = t2svalue * mask  # ms

    # remove nan
    fieldmap = np.nan_to_num(fieldmap, neginf=0.0, posinf=0.0)

    return fieldmap.astype(np.float32), mask


def _make_disk(shape, frac_radius=0.3):
    """Make circular binary mask."""
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    yy, xx = yy - ny // 2, xx - nx // 2
    yy, xx = yy / ny, xx / nx
    rr = (xx**2 + yy**2) ** 0.5
    return rr < frac_radius, rr


def _make_sphere(shape, frac_radius=0.3):
    """Make spherical binary mask."""
    nz, ny, nx = shape
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    zz, yy, xx = zz - nz // 2, yy - ny // 2, xx - nx // 2
    zz, yy, xx = zz / nz, yy / ny, xx / nx
    rr = (xx**2 + yy**2 + zz**2) ** 0.5
    return rr < frac_radius, rr
