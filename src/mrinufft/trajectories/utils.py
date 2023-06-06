import numpy as np


#############
# CONSTANTS #
#############

KMAX = 0.5
DEFAULT_CONE_ANGLE = np.pi / 2  # rad
DEFAULT_HELIX_ANGLE = np.pi  # rad

DEFAULT_RESOLUTION = 6e-4  # m
DEFAULT_RASTER_TIME = 10e-6  # s
DEFAULT_GYROMAGNETIC_RATIO = 42.576e6  # Hz/T

DEFAULT_GMAX = 0.04  # T/m
DEFAULT_SMAX = 100.0  # T/m/s


###############
# CONSTRAINTS #
###############

def compute_gradients(trajectory,
                      resolution=DEFAULT_RESOLUTION,
                      raster_time=DEFAULT_RASTER_TIME,
                      g_ratio=DEFAULT_GYROMAGNETIC_RATIO):
    trajectory = trajectory / KMAX / (2 * resolution * g_ratio)
    gradients = np.diff(trajectory, axis=1) / raster_time
    slews = np.diff(gradients, axis=1) / raster_time
    return gradients, slews

def check_gradient_constraints(trajectory,
                               resolution=DEFAULT_RESOLUTION,
                               raster_time=DEFAULT_RASTER_TIME,
                               g_ratio=DEFAULT_GYROMAGNETIC_RATIO,
                               gmax=DEFAULT_GMAX, smax=DEFAULT_SMAX):
    gradients, slews = compute_gradients(trajectory, resolution, raster_time, g_ratio)
    max_grad = np.max(np.linalg.norm(gradients, axis=-1))
    max_slew = np.max(np.linalg.norm(slews, axis=-1))
    return (max_grad < gmax) and (max_slew < smax), max_grad, max_slew


###############
# MATHEMATICS #
###############

def compute_greatest_common_divider(p, q):
    while q != 0:
        p, q = q, p % q
    return p

def compute_coprime_factors(Nc, length, start=1, update=1):
    count = start
    coprimes = []
    while len(coprimes) < length:
        if compute_greatest_common_divider(Nc, count) == 1:
            coprimes.append(count)
        count += update
    return coprimes


#############
# ROTATIONS #
#############

# Initialize 2D rotation matrix
def R2D(theta):
  return np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])


# Initialize 3D rotation matrix around x axis
def Rx(theta):
  return np.array([[1,             0,              0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])

# Initialize 3D rotation matrix around y axis
def Ry(theta):
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [             0, 1,             0],
                   [-np.sin(theta), 0, np.cos(theta)]])

# Initialize 3D rotation matrix around z axis
def Rz(theta):
  return np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [            0,              0, 1]])


def Rv(v1, v2, normalize=True):
    if normalize:
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    v3 = np.cross(v1, v2)
    cross_matrix = np.cross(v3, np.identity(v3.shape[0]) * -1)
    return np.identity(3) + cross_matrix + cross_matrix @ cross_matrix / (1 + cos_theta)


###########
# OPTIONS #
###########


def initialize_tilt(tilt, nb_partitions=1):
    if not isinstance(tilt, str):
        return tilt
    elif tilt == "none":
        return 0
    elif tilt == "uniform":
        return 2 * np.pi / nb_partitions
    elif tilt == "intergaps":
        return np.pi / nb_partitions / 2
    elif tilt == "inverted":
        return np.pi / nb_partitions + np.pi
    elif tilt == "golden":
        return np.pi * (3 - np.sqrt(5))
    elif tilt == "mri golden":
        return np.pi * (np.sqrt(5) - 1) / 2
    else:
        raise NotImplementedError(f"Unknown tilt name: {tilt}")


def initialize_spiral(spiral):
    """Initialize the spiral type.

    Parameters
    ----------
    spiral : str or int
        Spiral type or number of interleaves.

    Returns
    -------
    int
        Spiral type.
    """
    if not isinstance(spiral, str):
        return spiral
    elif spiral == "archimedes":
        return 1
    elif spiral == "fermat":
        return 2
    else:
        raise NotImplementedError(f"Unknown spiral name: {spiral}")
