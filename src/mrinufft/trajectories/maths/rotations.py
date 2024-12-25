"""Rotation functions in 2D & 3D spaces."""

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray


def R2D(theta: float) -> NDArray:
    """Initialize 2D rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    NDArray
        2D rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def Rx(theta: float) -> NDArray:
    """Initialize 3D rotation matrix around x axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    NDArray
        3D rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta: float) -> NDArray:
    """Initialize 3D rotation matrix around y axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    NDArray
        3D rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta: float) -> NDArray:
    """Initialize 3D rotation matrix around z axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    NDArray
        3D rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def Rv(
    v1: NDArray,
    v2: NDArray,
    eps: float = 1e-8,
    *,
    normalize: bool = True,
) -> NDArray:
    """Initialize 3D rotation matrix from two vectors.

    Initialize a 3D rotation matrix from two vectors using Rodrigues's rotation
    formula. Note that the rotation is carried around the axis orthogonal to both
    vectors from the origin, and therefore is undetermined when both vectors
    are colinear. While this case is handled manually, close cases might result
    in approximative behaviors.

    Parameters
    ----------
    v1 : NDArray
        Source vector.
    v2 : NDArray
        Target vector.
    eps : float, optional
        Tolerance to consider two vectors as colinear. The default is 1e-8.
    normalize : bool, optional
        Normalize the vectors. The default is True.

    Returns
    -------
    NDArray
        3D rotation matrix.
    """
    # Check for colinearity, not handled by Rodrigues' coefficients
    if nl.norm(np.cross(v1, v2)) < eps:
        sign = np.sign(np.dot(v1, v2))
        return sign * np.identity(3)

    if normalize:
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    v3 = np.cross(v1, v2)
    cross_matrix = np.cross(v3, np.identity(v3.shape[0]) * -1)
    return np.identity(3) + cross_matrix + cross_matrix @ cross_matrix / (1 + cos_theta)


def Ra(vector: NDArray, theta: float) -> NDArray:
    """Initialize 3D rotation matrix around an arbitrary vector.

    Initialize a 3D rotation matrix to rotate around `vector` by an angle `theta`.
    It corresponds to a generalized formula with `Rx`, `Ry` and `Rz` as subcases.

    Parameters
    ----------
    vector : NDArray
        Vector defining the rotation axis, automatically normalized.
    theta : float
        Angle in radians defining the rotation around `vector`.

    Returns
    -------
    NDArray
        3D rotation matrix.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    v_x, v_y, v_z = vector / np.linalg.norm(vector)
    return np.array(
        [
            [
                cos_t + v_x**2 * (1 - cos_t),
                v_x * v_y * (1 - cos_t) + v_z * sin_t,
                v_x * v_z * (1 - cos_t) - v_y * sin_t,
            ],
            [
                v_y * v_x * (1 - cos_t) - v_z * sin_t,
                cos_t + v_y**2 * (1 - cos_t),
                v_y * v_z * (1 - cos_t) + v_x * sin_t,
            ],
            [
                v_z * v_x * (1 - cos_t) + v_y * sin_t,
                v_z * v_y * (1 - cos_t) - v_x * sin_t,
                cos_t + v_z**2 * (1 - cos_t),
            ],
        ]
    )
