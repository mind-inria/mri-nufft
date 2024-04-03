"""Compute density compensation weights using geometry-based methods."""

import numpy as np
from scipy.spatial import Voronoi

from .utils import flat_traj, normalize_weights, register_density


def _vol3d(points):
    """Compute the volume of a convex 3D polygon.

    Parameters
    ----------
    points: array_like
        array of shape (N, 3) containing the coordinates of the points.

    Returns
    -------
    volume: float
    """
    base_point = points[0]
    A = points[:-2] - base_point
    B = points[1:-1] - base_point
    C = points[2:] - base_point
    return np.sum(np.abs(np.dot(np.cross(B, C), A.T))) / 6.0


def _vol2d(points):
    """Compute the area of a convex 2D polygon.

    Parameters
    ----------
    points: array_like
        array of shape (N, 2) containing the coordinates of the points.

    Returns
    -------
    area: float
    """
    # https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    area = 0
    for i in range(1, len(points) - 1):
        area += points[i, 0] * (points[i + 1, 1] - points[i - 1, 1])
    area += points[-1, 0] * (points[0, 1] - points[-2, 1])
    # we actually don't provide the last point, so we have to do another edge case.
    area += points[0, 0] * (points[1, 1] - points[-1, 1])
    return abs(area) / 2.0


@register_density
@flat_traj
def voronoi_unique(traj, *args, **kwargs):
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
    M = traj.shape[0]
    if traj.shape[1] == 2:
        vol = _vol2d
    else:
        vol = _vol3d
    wi = np.zeros(M)
    v = Voronoi(traj)
    for mm in range(M):
        idx_vertices = v.regions[v.point_region[mm]]
        if np.all([i != -1 for i in idx_vertices]):
            wi[mm] = vol(v.vertices[idx_vertices])
        else:
            wi[mm] = np.inf
    # some voronoi cell are considered closed, but have a too big area.
    # (They are closing near infinity).
    # we classify them as open cells as well.
    outlier_thresh = np.percentile(wi, 95)
    wi[wi > outlier_thresh] = np.inf

    # For edge point (infinite voronoi cells) we extrapolate from neighbours
    # Initial implementation in Jeff Fessler's MIRT
    rho = np.sum(traj**2, axis=1)
    igood = (rho > 0.6 * np.max(rho)) & ~np.isinf(wi)
    if len(igood) < 10:
        print("dubious extrapolation with", len(igood), "points")
    poly = np.polynomial.Polynomial.fit(rho[igood], wi[igood], 3)
    wi[np.isinf(wi)] = poly(rho[np.isinf(wi)])
    return wi


@register_density
@flat_traj
def voronoi(traj, *args, **kwargs):
    """Estimate  density compensation weight using voronoi parcellation.

    In case of multiple point in the center of kspace, the weight is split evenly.

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
    # deduplication only works for the 0,0 coordinate !!
    i0 = np.sum(np.abs(traj), axis=1) == 0
    if np.any(i0):
        i0f = np.where(i0)
        i0f = i0f[0]
        i0[i0f] = False
        wi = np.zeros(len(traj))
        wi[~i0] = voronoi_unique(traj[~i0])
        i0[i0f] = True
        wi[i0] = wi[i0f] / np.sum(i0)
    else:
        wi = voronoi_unique(traj)
    return 1 / normalize_weights(wi)


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

    return normalize_weights(weights)
