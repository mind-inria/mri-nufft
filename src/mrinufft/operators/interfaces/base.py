"""
Base Fourier Operator interface.

from https://github.com/CEA-COSMIC/pysap-mri

:author: Pierre-Antoine Comby
"""
import warnings
import numpy as np


def proper_trajectory(trajectory, normalize=True):
    """Normalize the trajectory to be used by NUFFT operators.

    Parameters
    ----------
    trajectory: np.ndarray
        The trajectory to normalize, it might be of shape (Nc, Ns, dim) of (Ns, dim)

    normalize: bool
        If True and if the trajectory is in [-0.5,0.5] the trajectory is
        multiplied by 2pi to lie in [-pi, pi]

    Returns
    -------
    new_traj: np.ndarray
        The normalized trajectory of shape (Nc * Ns, dim) or (Ns, dim) in -pi, pi
    """
    # flatten to a list of point
    try:
        new_traj = np.asarray(trajectory).copy()
    except Exception as e:
        raise ValueError(
            "trajectory should be array_like, with the last dimension being coordinates"
        ) from e
    new_traj = new_traj.reshape(-1, trajectory.shape[-1])

    # normalize the trajectory to -pi, pi if i
    if normalize and np.isclose(np.max(abs(new_traj)), 0.5):
        warnings.warn("samples will be rescaled in [-pi, pi]")
        new_traj *= 2 * np.pi
    return new_traj


class FourierOperatorBase:
    """Base Fourier Operator class.

    Every (Linear) Fourier operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by ModOpt.

    Attributes
    ----------
    shape: tuple
        The shape of the image space (in 2D or 3D)
    n_coils: int
        The number of coils.
    uses_sense: bool
        True if the operator uses sensibility maps.

    Methods
    -------
    op(data)
        The forward operation (image -> kspace)
    adj_op(coeffs)
        The adjoint operation (kspace -> image)
    """

    def __init__(self):
        self._uses_sense = False

    def op(self, data):
        """Compute operator transform.

        Parameters
        ----------
        data: np.ndarray
            input as array.

        Returns
        -------
        result: np.ndarray
            operator transform of the input.
        """
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, coeffs):
        """Compute adjoint operator transform.

        Parameters
        ----------
        x: np.ndarray
            input data array.

        Returns
        -------
        results: np.ndarray
            adjoint operator transform.
        """
        raise NotImplementedError("'adj_op' is an abstract method.")

    @property
    def uses_sense(self):
        """Return True if the operator uses sensitivity maps."""
        return self._uses_sense

    @property
    def shape(self):
        """Shape of the image space of the operator."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    @property
    def n_coils(self):
        """Return number of coil of the image space of the operator."""
        return self._n_coils

    @n_coils.setter
    def n_coils(self, n_coils):
        if n_coils < 1 or not int(n_coils) == n_coils:
            raise ValueError(f"n_coils should be a positive integer, {type(n_coils)}")
        self._n_coils = int(n_coils)

    def with_off_resonnance_correction(self, B, C, indices):
        """Return a new operator with Off Resonnance Correction."""
        from ..off_resonnance import MRIFourierCorrected

        return MRIFourierCorrected(self, B, C, indices)
