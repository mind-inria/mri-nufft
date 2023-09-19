"""
Base Fourier Operator interface.

from https://github.com/CEA-COSMIC/pysap-mri

:author: Pierre-Antoine Comby
"""
from abc import ABC, abstractmethod

import warnings
import numpy as np

# Mapping between numpy float and complex types.
DTYPE_R2C = {"float32": "complex64", "float64": "complex128"}


def proper_trajectory(trajectory, normalize="pi"):
    """Normalize the trajectory to be used by NUFFT operators.

    Parameters
    ----------
    trajectory: np.ndarray
        The trajectory to normalize, it might be of shape (Nc, Ns, dim) of (Ns, dim)

    normalize: str
        if "pi" trajectory will be rescaled in [-pi, pi], if it was in [-0.5, 0.5]
        if "unit" trajectory will be rescaled in [-0.5, 0.5] if it was not [-0.5, 0.5]

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

    if normalize == "pi" and np.max(abs(new_traj)) - 1e-4 < 0.5:
        warnings.warn(
            "Samples will be rescaled to [-pi, pi), assuming they were in [-0.5, 0.5)"
        )
        new_traj *= 2 * np.pi
    elif normalize == "unit" and np.max(abs(new_traj)) - 1e-4 > 0.5:
        warnings.warn(
            "Samples will be rescaled to [-0.5, 0.5), assuming they were in [-pi, pi)"
        )
        new_traj /= 2 * np.pi
    if normalize == "unit" and np.max(new_traj) >= 0.5:
        new_traj = (new_traj + 0.5) % 1 - 0.5
    return new_traj


def check_backend(backend_name: str):
    """Check if a specific backend is available."""
    try:
        return FourierOperatorBase.interfaces[backend_name][0]
    except KeyError as e:
        raise ValueError(f"unknown backend: '{backend_name}'") from e


def list_backends(available_only=False):
    """Return a list of backend.

    Parameters
    ----------
    available_only: bool, optional
        If True, only return backends that are available. If False, return all
        backends, regardless of whether they are available or not.
    """
    return [
        name
        for name, (available, _) in FourierOperatorBase.interfaces.items()
        if available or not available_only
    ]


def get_operator(backend_name: str, *args, **kwargs):
    """Return an MRI Fourier operator interface using the correct backend.

    Parameters
    ----------
    backend_name: str
        Backend name
    *args, **kwargs:
        Arguments to pass to the operator constructor.

    Returns
    -------
    FourierOperator
        class or instance of class if args or kwargs are given.

    Raises
    ------
    ValueError if the backend is not available.
    """
    try:
        available, operator = FourierOperatorBase.interfaces[backend_name]
    except KeyError as exc:
        raise ValueError("backend is not available") from exc
    if not available:
        raise ValueError("backend is registered, but dependencies are not met.")

    if args or kwargs:
        operator = operator(*args, **kwargs)
    return operator


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

    interfaces = {}

    def __init__(self):
        if not self.available:
            raise RuntimeError(f"'{self.backend}' backend is not available.")
        self._smaps = None
        self._density = None
        self._n_coils = 1

    def __init_subclass__(cls):
        """Register the class in the list of available operators."""
        super().__init_subclass__()
        available = getattr(cls, "available", True)
        if callable(available):
            available = available()
        if backend := getattr(cls, "backend", None):
            cls.interfaces[backend] = (available, cls)

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

    def with_off_resonnance_correction(self, B, C, indices):
        """Return a new operator with Off Resonnance Correction."""
        from ..off_resonnance import MRIFourierCorrected

        return MRIFourierCorrected(self, B, C, indices)

    @property
    def uses_sense(self):
        """Return True if the operator uses sensitivity maps."""
        return self._smaps is not None

    @property
    def uses_density(self):
        """Return True if the operator uses density compensation."""
        return getattr(self, "density", None) is not None

    @property
    def shape(self):
        """Shape of the image space of the operator."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    @property
    def n_coils(self):
        """Number of coils for the operator."""
        return self._n_coils

    @property
    def ndim(self):
        """Number of dimensions in image space of the operator."""
        return len(self._shape)

    @n_coils.setter
    def n_coils(self, n_coils):
        if n_coils < 1 or not int(n_coils) == n_coils:
            raise ValueError(f"n_coils should be a positive integer, {type(n_coils)}")
        self._n_coils = int(n_coils)

    @property
    def smaps(self):
        """Sensitivity maps of the operator."""
        return self._smaps

    @smaps.setter
    def smaps(self, smaps):
        if smaps is None:
            self._smaps = None
        elif len(smaps) != self.n_coils:
            raise ValueError(
                f"Number of sensitivity maps ({len(smaps)})"
                f"should be equal to n_coils ({self.n_coils})"
            )
        else:
            self._smaps = smaps

    @property
    def density(self):
        """Density compensation of the operator."""
        return self._density

    @density.setter
    def density(self, density):
        if density is None:
            self._density = None
        elif len(density) != self.n_samples:
            raise ValueError("Density and samples should have the same length")
        else:
            self._density = density

    @property
    def dtype(self):
        """Return floating precision of the operator."""
        return self._dtype

    @property
    def cpx_dtype(self):
        """Return complex floating precision of the operator."""
        return np.dtype(DTYPE_R2C[str(self.dtype)])

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = np.dtype(dtype)

    @property
    def samples(self):
        """Return the samples used by the operator."""
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def n_samples(self):
        """Return the number of samples used by the operator."""
        return self._samples.shape[0]

    @property
    def norm_factor(self):
        """Normalization factor of the operator."""
        return np.sqrt(np.prod(self.shape) * (2 ** len(self.shape)))

    def __repr__(self):
        """Return info about the Fourier operator."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  shape: {self.shape}\n"
            f"  n_coils: {self.n_coils}\n"
            f"  n_samples: {self.n_samples}\n"
            f"  uses_sense: {self.uses_sense}\n"
            ")"
        )


class FourierOperatorCPU(FourierOperatorBase):
    """Base class for CPU-based NUFFT operator.

    Parameters
    ----------
    samples: np.ndarray
        The samples used by the operator.
    shape: tuple
        The shape of the image space (in 2D or 3D)
    density: bool or np.ndarray
        If True, the density compensation is estimated from the samples.
        If False, no density compensation is applied.
        If np.ndarray, the density compensation is applied from the array.
    n_coils: int
        The number of coils.
    smaps: np.ndarray
        The sensitivity maps.
    raw_op: object
        An object implementing the NUFFT API. Ut should be responsible to compute a
        single type 1 /type 2 NUFFT.
    """

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        smaps=None,
        raw_op=None,
    ):
        super().__init__()
        self.shape = shape
        self.samples = proper_trajectory(samples, normalize="unit")
        self._dtype = self.samples.dtype
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

        self.raw_op = raw_op

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
        if data.dtype != self.cpx_dtype:
            warnings.warn(
                f"Data should be of dtype {self.cpx_dtype} (is {data.dtype}). "
                "Casting it for you."
            )
            data = data.astype(self.cpx_dtype)
        # sense
        if self.uses_sense:
            ret = self._op_sense(data, ksp)
        # calibrationless or monocoil.
        else:
            if data.ndim == self.ndim:
                data = np.expand_dims(data, axis=0)  # add coil dimension
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
        self.raw_op.op(coeffs, image)
        coeffs /= self.norm_factor
        return coeffs

    def adj_op(self, coeffs, img=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        if coeffs.dtype != self.cpx_dtype:
            warnings.warn(
                f"coeffs should be of dtype {self.cpx_dtype}. Casting it for you."
            )
            coeffs = coeffs.astype(self.cpx_dtype)
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
            img = np.zeros((self.n_coils, *self.shape), dtype=coeffs.dtype)
        self._adj_op(coeffs, img)
        return img

    def _apply_dc(self, coeffs):
        if self.density is not None:
            return coeffs * self.density
        return coeffs

    def _adj_op(self, coeffs, image):
        self.raw_op.adj_op(self._apply_dc(coeffs), image)
        image /= self.norm_factor
        return image

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
