import numpy as np

from .expansions import (
    stack_2D_to_3D_expansion,
    rotate_2D_to_3D_expansion,
    cone_2D_to_3D_expansion,
    helix_2D_to_3D_expansion,
)
from .trajectory2D import (
    initialize_2D_radial,
    initialize_2D_spiral,
    initialize_2D_rosette,
    initialize_2D_cones,
)
from .utils import KMAX, Rv, initialize_tilt


################################
# 3D TRAJECTORY INITIALIZATION #
################################


def initialize_3D_from_2D_expansion(
    basis, expansion, Nc, Ns, nb_repetitions, basis_kwargs={}, expansion_kwargs={}
):
    # Initialization and warnings for 2D trajectory basis
    bases = {
        "radial": initialize_2D_radial,
        "spiral": initialize_2D_spiral,
        "rosette": initialize_2D_rosette,
        "cones": initialize_2D_cones,
    }
    if isinstance(basis, np.ndarray):
        trajectory2D = basis
    elif basis not in bases.keys():
        raise NotImplementedError(f"Unknown 2D trajectory basis: {basis}")
    else:
        basis_function = bases[basis]
        trajectory2D = basis_function(Nc, Ns, **basis_kwargs)

    # Initialization and warnings for 3D trajectory expansion
    expansions = {
        "stacks": stack_2D_to_3D_expansion,
        "rotations": rotate_2D_to_3D_expansion,
        "cones": cone_2D_to_3D_expansion,
        "helices": helix_2D_to_3D_expansion,
    }
    if expansion not in expansions.keys():
        raise NotImplementedError(f"Unknown 3D expansion: {expansion}")
    expansion_function = expansions[expansion]
    trajectory3D = expansion_function(trajectory2D, nb_repetitions, **expansion_kwargs)
    return trajectory3D.reshape((nb_repetitions * Nc, Ns, 3))


def initialize_3D_cones(
    Nc, Ns, tilt="golden", in_out=False, nb_zigzags=5, nb_overlaps=0
):
    # Initialize first cone characteristics
    radius = np.linspace(-KMAX if (in_out) else 0, KMAX, Ns)
    angles = np.linspace(
        -2 * np.pi * nb_zigzags if (in_out) else 0, 2 * np.pi * nb_zigzags, Ns
    )
    trajectory = np.zeros((Nc, Ns, 3))
    trajectory[:, :, 0] = radius
    trajectory[:, :, 1] = (
        radius
        * np.cos(angles)
        * (np.abs(nb_overlaps) + 1)
        * 2
        * np.pi
        / Nc ** (2 / 3)
        / (1 + in_out)
    )
    trajectory[:, :, 2] = (
        radius
        * np.sin(angles)
        * (np.abs(nb_overlaps) + 1)
        * 2
        * np.pi
        / Nc ** (2 / 3)
        / (1 + in_out)
    )

    # Determine mostly evenly distributed points on sphere
    points = np.zeros((Nc, 3))
    phi = initialize_tilt(tilt) * np.arange(Nc) / (1 + in_out)
    points[:, 0] = np.linspace(-1, 1, Nc)
    radius = np.sqrt(1 - points[:, 0] ** 2)
    points[:, 1] = np.cos(phi) * radius
    points[:, 2] = np.sin(phi) * radius

    # Rotate initial cone Nc times
    for i in np.arange(1, Nc)[::-1]:
        v1 = np.array((1, 0, 0))
        v2 = points[i]
        rotation = Rv(v1, v2, normalize=False)
        trajectory[i] = (rotation @ trajectory[0].T).T
    return trajectory.reshape((Nc, Ns, 3))


def duplicate_per_axes(trajectory, axes=(0, 1, 2)):
    # Copy input trajectory along other axes
    new_trajectory = []
    if 0 in axes:
        new_trajectory.append(trajectory)
    if 1 in axes:
        dp_trajectory = np.copy(trajectory)
        dp_trajectory[..., [1, 2]] = dp_trajectory[..., [2, 1]]
        new_trajectory.append(dp_trajectory)
    if 2 in axes:
        dp_trajectory = np.copy(trajectory)
        dp_trajectory[..., [2, 0]] = dp_trajectory[..., [0, 2]]
        new_trajectory.append(dp_trajectory)
    new_trajectory = np.concatenate(new_trajectory, axis=0)
    return new_trajectory


def _radialize_center_out(trajectory, nb_samples):
    Nc, Ns = trajectory.shape[:2]
    new_trajectory = np.copy(trajectory)
    for i in range(Nc):
        point = trajectory[i, nb_samples]
        new_trajectory[i, :nb_samples, 0] = np.linspace(0, point[0], nb_samples)
        new_trajectory[i, :nb_samples, 1] = np.linspace(0, point[1], nb_samples)
        new_trajectory[i, :nb_samples, 2] = np.linspace(0, point[2], nb_samples)
    return new_trajectory


def _radialize_in_out(trajectory, nb_samples):
    Nc, Ns = trajectory.shape[:2]
    new_trajectory = np.copy(trajectory)
    first, half, second = (Ns - nb_samples) // 2, Ns // 2, (Ns + nb_samples) // 2
    for i in range(Nc):
        p1 = trajectory[i, first]
        new_trajectory[i, first:half, 0] = np.linspace(0, p1[0], nb_samples // 2)
        new_trajectory[i, first:half, 1] = np.linspace(0, p1[1], nb_samples // 2)
        new_trajectory[i, first:half, 2] = np.linspace(0, p1[2], nb_samples // 2)
        p2 = trajectory[i, second]
        new_trajectory[i, half:second, 0] = np.linspace(0, p2[0], nb_samples // 2)
        new_trajectory[i, half:second, 1] = np.linspace(0, p2[1], nb_samples // 2)
        new_trajectory[i, half:second, 2] = np.linspace(0, p2[2], nb_samples // 2)
    return new_trajectory


def radialize_center(trajectory, nb_samples, in_out=False):
    # Make nb_samples into straight lines around the center
    if in_out:
        return _radialize_in_out(trajectory, nb_samples)
    else:
        return _radialize_center_out(trajectory, nb_samples)
