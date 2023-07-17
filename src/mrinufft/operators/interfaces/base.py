"""
Base Fourier Operator interface.

from https://github.com/CEA-COSMIC/pysap-mri

:author: Pierre-Antoine Comby
"""
from abc import ABC, abstractmethod

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


class FourierOperatorBase(ABC):
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

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

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


class FourierOperatorCPU(FourierOperatorBase):
    """Base class for CPU-based NUFFT operator."""

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        smaps=None,
    ):
        super().__init__()
        self.shape = shape
        self.n_samples = len(samples)
        if samples.max() > np.pi:
            warnings.warn("samples will be normalized in [-pi, pi]")
            samples *= np.pi / samples.max()
        # we will access the samples by their coordinate first.
        self.samples = np.asfortranarray(samples)

        self._dtype = self.samples.dtype
        self._cpx_dtype = np.complex128 if self._dtype == "float64" else np.complex64
        self._uses_sense = False

        # Density Compensation Setup
        if density is True:
            self.density = self.estimate_density(samples, shape)
        elif isinstance(density, np.ndarray):
            if len(density) != len(samples):
                raise ValueError(
                    "Density array and samples array should have the same length."
                )
            self.density = np.asfortranarray(density)
        else:
            self.density = None
        # Multi Coil Setup
        if n_coils < 1:
            raise ValueError("n_coils should be ≥ 1")
        self.n_coils = n_coils
        if smaps is not None:
            self._uses_sense = True
            if isinstance(smaps, np.ndarray):
                raise ValueError("Smaps should be either a C-ordered ndarray")
            self._smaps = smaps
        else:
            self._uses_sense = False

        # Raw_op should be instantiated by subclasses.
        self.raw_op = None

    def op(self, data, ksp=None):
        r"""Non Cartesian MRI forward operator.

        Parameters
        ----------
        data: np.ndarray
        The uniform (2D or 3D) data in image space.

        Returns
        -------
        Results array on the same device as data.

        Notes
        -----
        this performs for every coil \ell:
        ..math:: \mathcal{F}\mathcal{S}_\ell x
        """
        if data.dtype != self._cpx_dtype:
            warnings.warn(
                f"Data should be of dtype {self._cpx_dtype}. Casting it for you."
            )
            data = data.astype(self._cpx_dtype)
        # sense
        if self.uses_sense:
            ret = self._op_sense(data, ksp)
        # calibrationless or monocoil.
        else:
            ret = self._op_calibless(data, ksp)
        return ret

    def _op_sense(self, data, ksp_d=None):
        coil_img = np.empty((self.n_coils, *self.shape), dtype=data.dtype)
        ksp = np.zeros((self.n_coils, self.n_samples), dtype=data.dtype)
        coil_img = data * self._smaps
        self._op(coil_img)
        return ksp

    def _op_calibless(self, data, ksp=None):
        if ksp is None:
            ksp = np.empty((self.n_coils, self.n_samples), dtype=data.dtype)
        for i in range(self.n_coils):
            self._op(data[i], ksp[i])
        return ksp

    def _op(self, image, coeffs):
        return self.raw_op.op(coeffs, image)

    def adj_op(self, coeffs, img=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        if coeffs.dtype != self._cpx_dtype:
            warnings.warn(
                f"coeffs should be of dtype {self._cpx_dtype}. Casting it for you."
            )
            coeffs = coeffs.astype(self._cpx_dtype)
        if self.uses_sense:
            ret = self._adj_op_sense(coeffs, img)
        # calibrationless or monocoil.
        else:
            ret = self._adj_op_calibless(coeffs, img)
        return ret

    def _adj_op_sense(self, coeffs, img=None):
        coil_img = np.empty(self.shape, dtype=coeffs.dtype)
        if img is None:
            img = np.zeros(self.shape, dtype=coeffs.dtype)
        self._adj_op(coeffs, coil_img)
        img = np.sum(coil_img * self._smaps.conjugate(), axis=0)
        return img

    def _adj_op_calibless(self, coeffs, img=None):
        if img is None:
            img = np.empty((self.n_coils, *self.shape), dtype=coeffs.dtype)
        self._adj_op(coeffs, img)
        return img

    def _apply_dc(self, coeffs):
        if self.density is not None:
            return coeffs * self.density
        return coeffs

    def _adj_op(self, coeffs, image):
        return self.raw_op.adj_op(self._apply_dc(coeffs), image) / self.norm_factor

    def data_consistency(self, image_data, obs_data):
        """Compute the gradient estimation directly on gpu.

        This mixes the op and adj_op method to perform F_adj(F(x-y))
        on a per coil basis. By doing the computation coil wise,
        it uses less memory than the naive call to adj_op(op(x)-y)

        Parameters
        ----------
        image: array
            Image on which the gradient operation will be evaluated.
            N_coil x Image shape is not using sense.
        obs_data: array
            Observed data.
        """
        if self.uses_sense:
            return self._data_consistency_sense(image_data, obs_data)
        return self._data_consistency_calibless(image_data, obs_data)

    def _data_consistency_sense(self, image_data, obs_data):
        img = np.empty_like(image_data)
        coil_img = np.empty(self.shape, dtype=image_data.dtype)
        coil_ksp = np.empty(self.n_samples, dtype=obs_data.dtype)
        for i in range(self.n_coils):
            np.copyto(coil_img, img)
            coil_img *= self._smap
            self._op(coil_img, coil_ksp)
            coil_ksp -= obs_data[i]
            if self.uses_density:
                coil_ksp *= self.density_d
            self._adj_op(coil_ksp, coil_img)
            img += coil_img * self._smaps[i].conjugate()
        return img

    def _data_consistency_calibless(self, image_data, obs_data):
        img = np.empty((self.n_coils, *self.shape), dtype=image_data.dtype)
        ksp = np.empty(self.n_samples, dtype=obs_data.dtype)
        for i in range(self.n_coils):
            self._op(image_data[i], ksp)
            ksp -= obs_data[i]
            if self.uses_density:
                ksp *= self.density_d
            self._adj_op(ksp, img[i])
        return img

    @property
    def eps(self):
        """Return the underlying precision parameter."""
        return self.raw_op.eps

    @classmethod
    def estimate_density(cls, samples, shape, n_iter=1, **kwargs):
        """Estimate the density compensation array."""
        oper = cls(samples, shape, density=False, **kwargs)
        density = np.ones(len(samples), dtype=oper._cpx_dtype)
        update = np.empty_like(density, dtype=oper._cpx_dtype)
        img = np.empty(shape, dtype=oper._cpx_dtype)
        for _ in range(n_iter):
            oper._adj_op(density, img)
            oper._op(img, update)
            density /= np.abs(update)
        return density.real

    def __repr__(self):
        """Return info about the MRICufiNUFFT Object."""
        return (
            "MRICufiNUFFT(\n"
            f"  shape: {self.shape}\n"
            f"  n_coils: {self.n_coils}\n"
            f"  n_samples: {self.n_samples}\n"
            f"  uses_density: {self.uses_density}\n"
            f"  uses_sense: {self._uses_sense}\n"
            f"  smaps_cached: {self.smaps_cached}\n"
            f"  plan_setup: {self.plan_setup}\n"
            f"  eps:{self.raw_op.eps:.0e}\n"
            ")"
        )
