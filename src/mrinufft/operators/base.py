"""
Base Fourier Operator interface.

from https://github.com/CEA-COSMIC/pysap-mri

:author: Pierre-Antoine Comby
"""
from abc import ABC, abstractmethod
from functools import partial
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
    available = True
    try:
        available, operator = FourierOperatorBase.interfaces[backend_name]
    except KeyError as exc:
        if not backend_name.startswith("stacked-"):
            raise ValueError(f"backend {backend_name} does not exist") from exc
        # try to get the backend with stacked
        # Dedicated registered stacked backend (like stacked-cufinufft)
        # have be found earlier.
        backend = backend_name.split("-")[1]
        operator = get_operator("stacked")
        operator = partial(operator, backend=backend)

    if not available:
        raise ValueError(f"backend {backend_name} found, but dependencies are not met.")

    if args or kwargs:
        operator = operator(*args, **kwargs)
    return operator


class FourierOperatorBase(ABC):
    """Base Fourier Operator class.

    Every (Linear) Fourier operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by ModOpt.
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

    def data_consistency(self, image, obs_data):
        """Compute the gradient data consistency.

        This is the naive implementation using adj_op(op(x)-y).
        Specific backend can (and should!) implement a more efficient version.
        """
        return self.adj_op(self.op(image) - obs_data)

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

    The NUFFT operation will be done sequentially and looped over coils and batches.

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
        n_batchs=1,
        n_trans=1,
        smaps=None,
        raw_op=None,
        squeeze_dims=True,
    ):
        super().__init__()

        self.shape = shape

        # we will access the samples by their coordinate first.
        self.samples = samples.reshape(-1, len(shape))
        self.dtype = self.samples.dtype

        # Density Compensation Setup
        if density is True:
            self.density = self.estimate_density(self.samples, shape)
        elif isinstance(density, np.ndarray):
            if len(density) != len(self.samples):
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
        self.smaps = smaps
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.squeeze_dims = squeeze_dims

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
            ret = self._op_calibless(data, ksp)
        ret /= self.norm_factor
        return self._safe_squeeze(ret)

    def _op_sense(self, data, ksp=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        dataf = data.reshape((B, *XYZ))
        if ksp is None:
            ksp = np.empty((B * C, K), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            coil_img = self.smaps[idx_coils].copy().reshape((T, *XYZ))
            coil_img *= dataf[idx_batch]
            self._op(coil_img, ksp[i * T : (i + 1) * T])
        ksp = ksp.reshape((B, C, K))
        return ksp

    def _op_calibless(self, data, ksp=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        if ksp is None:
            ksp = np.empty((B * C, K), dtype=self.cpx_dtype)
        dataf = np.reshape(data, (B * C, *XYZ))
        for i in range((B * C) // T):
            self._op(
                dataf[i * T : (i + 1) * T],
                ksp[i * T : (i + 1) * T],
            )
        ksp = ksp.reshape((B, C, K))
        return ksp

    def _op(self, image, coeffs):
        self.raw_op.op(coeffs, image)

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
        ret /= self.norm_factor
        return self._safe_squeeze(ret)

    def _adj_op_sense(self, coeffs, img=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        if img is None:
            img = np.zeros((B, *XYZ), dtype=self.cpx_dtype)
        coeffs_flat = coeffs.reshape((B * C, K))
        img_batched = np.zeros((T, *XYZ), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            self._adj_op(coeffs_flat[i * T : (i + 1) * T], img_batched)
            img_batched *= self.smaps[idx_coils].conj()
            for t, b in enumerate(idx_batch):
                img[b] += img_batched[t]
        img = img.reshape((B, 1, *XYZ))
        return img

    def _adj_op_calibless(self, coeffs, img=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        if img is None:
            img = np.empty((B * C, *XYZ), dtype=self.cpx_dtype)
        coeffs_f = np.reshape(coeffs, (B * C, K))
        for i in range((B * C) // T):
            self._adj_op(coeffs_f[i * T : (i + 1) * T], img[i * T : (i + 1) * T])

        img = img.reshape((B, C, *XYZ))
        return img

    def _adj_op(self, coeffs, image):
        if self.density is not None:
            coeffs2 = coeffs.copy()
            for i in range(self.n_trans):
                coeffs2[i * self.n_samples : (i + 1) * self.n_samples] *= self.density
        else:
            coeffs2 = coeffs
        self.raw_op.adj_op(coeffs2, image)

    def data_consistency(self, image_data, obs_data):
        """Compute the gradient data consistency.

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
            return self._safe_squeeze(self._grad_sense(image_data, obs_data))
        return self._safe_squeeze(self._grad_calibless(image_data, obs_data))

    def _grad_sense(self, image_data, obs_data):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        dataf = image_data.reshape((B, *XYZ))
        obs_dataf = obs_data.reshape((B * C, K))
        grad = np.empty_like(dataf)

        coil_img = np.empty((T, *XYZ), dtype=self.cpx_dtype)
        coil_ksp = np.empty((T, K), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            coil_img = self.smaps[idx_coils].copy().reshape((T, *XYZ))
            coil_img *= dataf[idx_batch]
            self._op(coil_img, coil_ksp)
            coil_ksp /= self.norm_factor
            coil_ksp -= obs_dataf[i * T : (i + 1) * T]
            self._adj_op(coil_ksp, coil_img)
            coil_img *= self.smaps[idx_coils].conj()
            for t, b in enumerate(idx_batch):
                grad[b] += coil_img[t]
        grad /= self.norm_factor
        return grad

    def _grad_calibless(self, image_data, obs_data):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        dataf = image_data.reshape((B * C, *XYZ))
        obs_dataf = obs_data.reshape((B * C, K))
        grad = np.empty_like(dataf)
        ksp = np.empty((T, K), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            self._op(dataf[i * T : (i + 1) * T], ksp)
            ksp /= self.norm_factor
            ksp -= obs_dataf[i * T : (i + 1) * T]
            if self.uses_density:
                ksp *= self.density
            self._adj_op(ksp, grad[i * T : (i + 1) * T])
        grad /= self.norm_factor
        return grad.reshape(B, C, *XYZ)

    def _safe_squeeze(self, arr):
        """Squeeze the first two dimensions of shape of the operator."""
        if self.squeeze_dims:
            try:
                arr = arr.squeeze(axis=1)
            except ValueError:
                pass
            try:
                arr = arr.squeeze(axis=0)
            except ValueError:
                pass
        return arr
