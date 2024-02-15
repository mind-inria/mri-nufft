"""
Base Fourier Operator interface.

from https://github.com/CEA-COSMIC/pysap-mri

:author: Pierre-Antoine Comby
"""

import warnings
from abc import ABC, abstractmethod
from functools import partial, wraps
import numpy as np
from mrinufft._utils import power_method, auto_cast, get_array_module
from mrinufft.operators.interfaces.utils import is_cuda_array

from mrinufft.density import get_density

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False

# Mapping between numpy float and complex types.
DTYPE_R2C = {"float32": "complex64", "float64": "complex128"}


def check_backend(backend_name: str):
    """Check if a specific backend is available."""
    backend_name = backend_name.lower()
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
    backend_name = backend_name.lower()
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


def with_numpy(fun):
    """Ensure the function works internally with numpy array."""

    @wraps(fun)
    def wrapper(self, data, *args, **kwargs):
        if hasattr(data, "__cuda_array_interface__"):
            warnings.warn("data is on gpu, it will be moved to CPU.")
        xp = get_array_module(data)
        if xp.__name__ == "torch":
            data_ = data.to("cpu").numpy()
        elif xp.__name__ == "cupy":
            data_ = data.get()
        elif xp.__name__ == "numpy":
            data_ = data
        else:
            raise ValueError(f"Array library {xp} not supported.")
        ret_ = fun(self, data_, *args, **kwargs)

        if xp.__name__ == "torch":
            if data.is_cpu:
                return xp.from_numpy(ret_)
            return xp.from_numpy(ret_).to(data.device)
        elif xp.__name__ == "cupy":
            return xp.array(ret_)
        else:
            return ret_

    return wrapper


def with_numpy_cupy(fun):
    """Ensure the function works internally with numpy or cupy array."""

    @wraps(fun)
    def wrapper(self, data, output=None, *args, **kwargs):
        xp = get_array_module(data)
        if xp.__name__ == "torch" and is_cuda_array(data):
            # Move them to cupy
            data_ = cp.from_dlpack(data)
            output_ = cp.from_dlpack(output) if output is not None else None
        elif xp.__name__ == "torch":
            # Move to numpy
            data_ = data.to("cpu").numpy()
            output_ = output.to("cpu").numpy() if output is not None else None
        else:
            data_ = data
            output_ = output

        ret_ = fun(self, data_, output_, *args, **kwargs)

        if xp.__name__ == "torch" and is_cuda_array(data):
            return xp.as_tensor(ret_, device=data.device)

        if xp.__name__ == "torch":
            if data.is_cpu:
                return xp.from_numpy(ret_)
            return xp.from_numpy(ret_).to(data.device)

        return ret_

    return wrapper


class FourierOperatorBase(ABC):
    """Base Fourier Operator class.

    Every (Linear) Fourier operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by ModOpt.
    """

    interfaces: dict[str, tuple] = {}

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

    def compute_density(self, method=None):
        """Compute the density compensation weights and set it.

        Parameters
        ----------
        method: str or callable or array or dict
            The method to use to compute the density compensation.
            If a string, the method should be registered in the density registry.
            If a callable, it should take the samples and the shape as input.
            If a dict, it should have a key 'name', to determine which method to use.
            other items will be used as kwargs.
            If an array, it should be of shape (Nsamples,) and will be used as is.
        """
        if isinstance(method, np.ndarray):
            self.density = method
            return None
        if not method:
            self.density = None
            return None

        kwargs = {}
        if isinstance(method, dict):
            kwargs = method.copy()
            method = kwargs.pop("name")  # must be a string !
        if method == "pipe" and "backend" not in kwargs:
            kwargs["backend"] = self.backend
        if isinstance(method, str):
            method = get_density(method)
        if not callable(method):
            raise ValueError(f"Unknown density method: {method}")

        self.density = method(self.samples, self.shape, **kwargs)

    def get_lipschitz_cst(self, max_iter=10, **kwargs):
        """Return the Lipschitz constant of the operator.

        Parameters
        ----------
        max_iter: int
            number of iteration to compute the lipschitz constant.
        **kwargs:
            Extra arguments givent

        Returns
        -------
        float
            Spectral Radius

        Notes
        -----
        This uses the Iterative Power Method to compute the largest singular value of a
        minified version of the nufft operator. No coil or B0 compensation is used,
        but includes any computed density.
        """
        tmp_op = self.__class__(
            self.samples, self.shape, density=self.density, n_coils=1, **kwargs
        )
        return power_method(max_iter, tmp_op)

    @property
    def uses_sense(self):
        """Return True if the operator uses sensitivity maps."""
        return self._smaps is not None

    @property
    def uses_density(self):
        """Return True if the operator uses density compensation."""
        return getattr(self, "density", None) is not None

    @property
    def ndim(self):
        """Number of dimensions in image space of the operator."""
        return len(self._shape)

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

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = np.dtype(dtype)

    @property
    def cpx_dtype(self):
        """Return complex floating precision of the operator."""
        return np.dtype(DTYPE_R2C[str(self.dtype)])

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
        self.compute_density(density)
        # Multi Coil Setup
        if n_coils < 1:
            raise ValueError("n_coils should be ≥ 1")
        self.n_coils = n_coils
        self.smaps = smaps
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.squeeze_dims = squeeze_dims

        self.raw_op = raw_op

    @with_numpy
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
        # sense
        data = auto_cast(data, self.cpx_dtype)

        if self.uses_sense:
            ret = self._op_sense(data, ksp)
        # calibrationless or monocoil.
        else:
            ret = self._op_calibless(data, ksp)
        ret /= self.norm_factor

        ret = self._safe_squeeze(ret)
        return ret

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

    @with_numpy
    def adj_op(self, coeffs, img=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        coeffs = auto_cast(coeffs, self.cpx_dtype)
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
