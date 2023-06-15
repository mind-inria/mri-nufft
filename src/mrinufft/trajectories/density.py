"""
Estimation of the density compensation array methods.

Those methods are agnostic of the NUFFT operator.
"""
import numpy as np
from scipy.spatial import Voronoi


def vol3d(points):
    """Compute the volume of a 3D (convex) polyhedron."""
    base_point = points[0]
    A = points[:-2] - base_point
    B = points[1:-1] - base_point
    C = points[2:] - base_point
    return np.sum(np.abs(np.dot(np.cross(B, C), A.T))) / 6.0


def vol2d(points):
    """Compute the area of a 2D (convex) polygon."""
    # https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    area = 0
    area = np.sum(points[1:-1, 0] * (points[2:, 1] - points[:-2, 0]))
    # Finishing the cycle
    area += points[-1, 0] * (points[0, 1] - points[-2, 1])
    area += points[0, 0] * (points[1, 1] - points[-1, 1])
    return abs(area) / 2.0


def _voronoi(kspace):
    M = kspace.shape[0]
    wi = np.zeros(M)
    v = Voronoi(kspace.astype(float))
    if kspace.shape[1] == 2:
        vol = vol2d
    else:
        vol = vol3d
    for mm in range(M):
        idx_vertices = v.regions[v.point_region[mm]]
        if np.all([i != -1 for i in idx_vertices]):
            wi[mm] = vol(v.vertices[idx_vertices])
    # For edge point (infinite voronoi cells) we extrapolate from neighbours
    # Initial implementation in Jeff Fessler's MIRT
    rho = np.sum(kspace**2, axis=1)
    igood = (rho > 0.7 * np.max(rho)) & (wi > 0)
    if len(igood) < 10:
        print("dubious extrapolation with", len(igood), "points")

    # pp, _, pc = np.polyfit(rho[igood], wi[igood], 2, full=True)
    # wi[wi == 0] = np.polyval(pp, (rho[wi == 0] - pc[0]) / pc[1])
    poly, [resid, rank, sv, rcond] = np.polynomial.Polynomial.fit(
        rho[igood], wi[igood], 3, full=True
    )
    wi[wi == 0] = poly(rho[wi == 0])
    return wi


def voronoi(kspace):
    """
    Voronoi density compensation method.

    Compute the density compensation array using the area/volume of a Voronoi cell at each point.
    Original MATLAB implementation by Jeff Fessler's MIRT [1]_.

    Parameters
    ----------
    kspace : np.array
        k-space trajectory as a (M, 2 or 3) array

    Returns
    -------
    wi : np.array
        density compensation array

    Notes
    -----
    This method is agnostic of the NUFFT operator, and use the area/volume of a Voronoi
    cell at each point of the k-space trajectory. At the edges, the voronoi cell are
    infinite, and the weight is thus extrapolated from neighbours.
    In the case of overlapping point at the center (and only there!) the weight is split in
    equal parts.

    References
    ----------
    .. [1] https://github.com/JeffFessler/mirt/blob/main/mri/ir_mri_density_comp.m
    """
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
    return wi
