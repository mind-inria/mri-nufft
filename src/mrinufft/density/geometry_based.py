"""Compute density compensation weights using geometry-based methods."""

import itertools
import logging

import numpy as np
from scipy.spatial import Voronoi

from .utils import flat_traj, _normalize_weights, register_density

logger = logging.getLogger(__name__)


def _ridge_measures(vertices, ridge_vertices, radius):
    """Compute the length (2D) or area (3D) of Voronoi ridges.

    Parameters
    ----------
    vertices: array_like
        array of shape (V, d) containing the Voronoi vertices.
    ridge_vertices: list of list of int
        Vertices of each ridge, cyclically ordered in 3D. -1 denotes a vertex at
        infinity.
    radius: float
        Ridges with a vertex further than this from the origin are flagged.

    Returns
    -------
    measure: array_like
        array of shape (R,) with the ridge measures, inf for unbounded ridges or
        ridges reaching beyond ``radius``.
    """
    lengths = np.fromiter(map(len, ridge_vertices), int, len(ridge_vertices))
    flat = np.fromiter(itertools.chain.from_iterable(ridge_vertices), int, sum(lengths))
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    pts = vertices[flat]  # -1 entries are garbage, flagged below.
    bad = (flat == -1) | (np.sum(pts**2, axis=1) > radius**2)
    bad = np.logical_or.reduceat(bad, starts)

    if vertices.shape[1] == 2:
        measure = np.linalg.norm(pts[starts + 1] - pts[starts], axis=1)
    else:
        # Fan triangulation from the first vertex of each (convex, planar) ridge.
        ridge_id = np.repeat(np.arange(len(lengths)), lengths)
        pos = np.arange(len(flat)) - starts[ridge_id]
        tri = np.flatnonzero((pos >= 1) & (pos <= lengths[ridge_id] - 2))
        origin = pts[starts[ridge_id[tri]]]
        cross = np.cross(pts[tri] - origin, pts[tri + 1] - origin)
        cross = np.stack(
            [np.bincount(ridge_id[tri], c, len(lengths)) for c in cross.T], axis=1
        )
        measure = 0.5 * np.linalg.norm(cross, axis=1)
    measure[bad] = np.inf
    return measure


def _voronoi_unique(traj, *args, **kwargs):
    """Estimate  density compensation weight using voronoi parcellation.

    This assume unicity of the point in the kspace.

    Parameters
    ----------
    kspace: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    *args, **kwargs:
        Dummy arguments to be compatible with other methods.

    Returns
    -------
    wi: array_like
        array of shape (M,) containing the density compensation weights.
    """
    M, d = traj.shape
    rho = np.sum(traj**2, axis=1)
    v = Voronoi(traj)
    rp = v.ridge_points
    # Cells that are open, or closing beyond the sampled k-space, are meaningless.
    measure = _ridge_measures(v.vertices, v.ridge_vertices, np.sqrt(np.max(rho)))
    is_open = np.zeros(M, dtype=bool)
    is_open[rp[np.isinf(measure)].ravel()] = True

    # A cell is the union of pyramids with a ridge as base and its generator as
    # apex, the height being half the distance to the neighbouring generator.
    height = 0.5 * np.linalg.norm(traj[rp[:, 0]] - traj[rp[:, 1]], axis=1)
    pyramid = np.where(np.isinf(measure), 0, measure * height / d)
    wi = np.bincount(rp[:, 0], pyramid, M) + np.bincount(rp[:, 1], pyramid, M)

    # For edge point (open voronoi cells) we extrapolate from neighbours
    # Initial implementation in Jeff Fessler's MIRT
    igood = (rho > 0.6 * np.max(rho)) & ~is_open
    if np.sum(igood) < 10:
        logger.info("dubious extrapolation with %d points", np.sum(igood))
    poly = np.polynomial.Polynomial.fit(rho[igood], wi[igood], 3)
    wi[is_open] = poly(rho[is_open])
    return wi


@register_density
@flat_traj
def voronoi(traj, *args, **kwargs):
    """Estimate  density compensation weight using voronoi parcellation.

    In case of duplicated points (e.g. the k-space center), the weight is split
    evenly.

    Parameters
    ----------
    traj: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.

    *args, **kwargs:
        Dummy arguments to be compatible with other methods.

    References
    ----------
    Based on the MATLAB implementation in MIRT: https://github.com/JeffFessler/mirt/blob/main/mri/ir_mri_density_comp.m
    """
    # Rounding catches near-duplicates that qhull would silently drop.
    _, first, inverse, counts = np.unique(
        np.round(traj, 12),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    wi = _voronoi_unique(traj[first])
    wi = (wi / counts)[inverse.ravel()]
    return 1 / _normalize_weights(wi)


@register_density
@flat_traj
def radial(traj, *args, tol=1e-6, **kwargs):
    """Compute density compensation weights for isotropic radial trajectories.

    Samples are grouped by their distance to the k-space center. Each group of
    samples shares the volume of the spherical shell (annulus in 2D) bounded by the
    midpoints to the neighboring radii. This yields weights proportional to
    :math:`|k|^{d-1}`, and works for both center-out and in-out trajectories.

    Parameters
    ----------
    traj: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    tol: float
        Relative tolerance (w.r.t. the largest radius) under which two radii are
        considered equal. default 1e-6
    *args, **kwargs:
        Dummy arguments to be compatible with other methods.

    Returns
    -------
    weights: array_like
        array of shape (M,) containing the density compensation weights.
    """
    dim = traj.shape[-1]
    r = np.linalg.norm(traj, axis=-1)
    order = np.argsort(r)
    r_sorted = r[order]

    new_group = np.diff(r_sorted) > tol * r_sorted[-1]
    group_id = np.concatenate([[0], np.cumsum(new_group)])
    counts = np.bincount(group_id)
    radii = r_sorted[np.concatenate([[0], np.flatnonzero(new_group) + 1])]

    if len(radii) == 1:
        return np.full(len(traj), 1 / len(traj))

    mid = (radii[1:] + radii[:-1]) / 2
    inner = np.concatenate([[0], mid])
    outer = np.concatenate([mid, [radii[-1] + (radii[-1] - radii[-2]) / 2]])
    shell = (outer**dim - inner**dim) / counts

    weights = np.empty(len(traj))
    weights[order] = shell[group_id]
    return weights / np.sum(weights)


@register_density
@flat_traj
def cell_count(traj, shape, osf=1.0):
    """
    Compute the number of points in each cell of the grid.

    Parameters
    ----------
    traj: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    shape: tuple
        shape of the grid.
    osf: float
        oversampling factor for the grid. default 1

    Returns
    -------
    weights: array_like
        array of shape (M,) containing the density compensation weights.

    """
    bins = [np.linspace(-0.5, 0.5, int(osf * s) + 1) for s in shape]

    h, edges = np.histogramdd(traj, bins)
    if len(shape) == 2:
        hsum = [np.sum(h, axis=1).astype(int), np.sum(h, axis=0).astype(int)]
    else:
        hsum = [
            np.sum(h, axis=(1, 2)).astype(int),
            np.sum(h, axis=(0, 2)).astype(int),
            np.sum(h, axis=(0, 1)).astype(int),
        ]

    # indices of ascending coordinate in each dimension.
    locs_sorted = [np.argsort(traj[:, i]) for i in range(len(shape))]

    weights = np.ones(len(traj))
    set_xyz = [[], [], []]
    for i in range(len(hsum)):
        ind = 0
        for binsize in hsum[i]:
            s = set(locs_sorted[i][ind : ind + binsize])
            if s:
                set_xyz[i].append(s)
            ind += binsize

    for sx in set_xyz[0]:
        for sy in set_xyz[1]:
            sxy = sx.intersection(sy)
            if not sxy:
                continue
            if len(shape) == 2:
                weights[list(sxy)] = len(sxy)
                continue
            for sz in set_xyz[2]:
                sxyz = sxy.intersection(sz)
                if sxyz:
                    weights[list(sxyz)] = len(sxyz)

    return _normalize_weights(weights)
