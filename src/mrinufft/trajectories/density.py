"""
Estimation of the density compensation array methods.

Those methods are agnostic of the NUFFT operator.
"""
import numpy as np
from scipy.spatial import Voronoi


def compute_tetrahedron_volume(A, B, C, D):
    """Compute the volume of a tetrahedron."""
    return np.abs(np.dot(np.cross(B - A, C - A), D - A)) / 6.0


def vol3d(points):
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


def vol2d(points):
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


def _voronoi(kspace):
    """Estimate  density compensation weight using voronoi parcellation.

    This assume unicity of the point in the kspace.

    Parameters
    ----------
    kspace: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.

    Returns
    -------
    wi: array_like
        array of shape (M,) containing the density compensation weights.
    """
    M = kspace.shape[0]
    if kspace.shape[1] == 2:
        vol = vol2d
    else:
        print("using vol3d")
        vol = vol3d
    wi = np.zeros(M)
    v = Voronoi(kspace)
    for mm in range(M):
        idx_vertices = v.regions[v.point_region[mm]]
        if np.all([i != -1 for i in idx_vertices]):
            wi[mm] = vol(v.vertices[idx_vertices])
    # For edge point (infinite voronoi cells) we extrapolate from neighbours
    # Initial implementation in Jeff Fessler's MIRT
    rho = np.sum(kspace**2, axis=1)
    igood = rho > 0.6 * np.max(rho)
    if len(igood) < 10:
        print("dubious extrapolation with", len(igood), "points")
    poly = np.polynomial.Polynomial.fit(rho[igood], wi[igood], 3)
    wi[wi == 0] = poly(rho[wi == 0])
    return wi


def voronoi(kspace):
    """Estimate  density compensation weight using voronoi parcellation.

    In case of multiple point in the center of kspace, the weight is split evenly.

    Parameters
    ----------
    kspace: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    """
    # deduplication only works for the 0,0 coordinate !!
    i0 = np.sum(np.abs(kspace), axis=1) == 0
    if np.any(i0):
        i0f = np.where(i0)
        i0f = i0f[0]
        i0[i0f] = False
        wi = np.zeros(len(kspace))
        wi[~i0] = _voronoi(kspace[~i0])
        i0[i0f] = True
        wi[i0] = wi[i0f] / np.sum(i0)
    else:
        wi = _voronoi(kspace)
    wi /= np.sum(wi)
    return wi
