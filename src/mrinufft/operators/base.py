"""
Base Fourier Operator interface.

from https://github.com/CEA-COSMIC/pysap-mri

:author: Pierre-Antoine Comby
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import ClassVar, Callable
import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy, with_numpy_cupy, AUTOGRAD_AVAILABLE
from mrinufft._utils import auto_cast, power_method
from mrinufft.density import get_density
from mrinufft.extras import get_smaps
from mrinufft.operators.interfaces.utils import is_cuda_array, is_host_array

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


def get_operator(
    backend_name: str, wrt_data: bool = False, wrt_traj: bool = False, *args, **kwargs
):
    """Return an MRI Fourier operator interface using the correct backend.

    Parameters
    ----------
    backend_name: str
        Backend name
    wrt_data: bool, default False
        if set gradients wrt to data and images will be available.
    wrt_traj: bool, default False
        if set gradients wrt to trajectory will be available.
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
        raise ValueError(
            f"backend {backend_name} found, but dependencies are not met."
            f" ``pip install mri-nufft[{backend_name}]`` may solve the issue."
        )

    if args or kwargs:
        operator = operator(*args, **kwargs)

    # if autograd:
    if wrt_data or wrt_traj:
        if isinstance(operator, FourierOperatorBase):
            operator = operator.make_autograd(wrt_data, wrt_traj)
        else:
            # instance will be created later
            operator = partial(operator.with_autograd, wrt_data, wrt_traj)

    return operator


class FourierOperatorBase(ABC):
    """Base Fourier Operator class.

    Every (Linear) Fourier operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by ModOpt.
    """

    interfaces: dict[str, tuple] = {}
    autograd_available = False
    _density_method = None
    _grad_wrt_data = False
    _grad_wrt_traj = False

    backend: ClassVar[str]
    available: ClassVar[bool]

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

    def check_shape(self, *, image=None, ksp=None):
        """
        Validate the shapes of the image or k-space data against operator shapes.

        Parameters
        ----------
        image : np.ndarray, optional
            If passed, the shape of image data will be checked.

        ksp : np.ndarray or object, optional
            If passed, the shape of the k-space data will be checked.

        Raises
        ------
        ValueError
            If the shape of the provided image does not match the expected operator
            shape, or if the number of k-space samples does not match the expected
            number of samples.
        """
        if image is not None:
            image_shape = image.shape[-len(self.shape) :]
            if image_shape != self.shape:
                raise ValueError(
                    f"Image shape {image_shape} is not compatible "
                    f"with the operator shape {self.shape}"
                )

        if ksp is not None:
            kspace_shape = ksp.shape[-1]
            if kspace_shape != self.n_samples:
                raise ValueError(
                    f"Kspace samples {kspace_shape} is not compatible "
                    f"with the operator samples {self.n_samples}"
                )
        if image is None and ksp is None:
            raise ValueError("Nothing to check, provides image or ksp arguments")

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

    def data_consistency(self, image_data, obs_data):
        """Compute the gradient data consistency.

        This is the naive implementation using adj_op(op(x)-y).
        Specific backend can (and should!) implement a more efficient version.
        """
        return self.adj_op(self.op(image_data) - obs_data)

    def with_off_resonance_correction(self, B, C, indices):
        """Return a new operator with Off Resonnance Correction."""
        from .off_resonance import MRIFourierCorrected

        return MRIFourierCorrected(self, B, C, indices)

    def compute_smaps(self, method: NDArray | Callable | str | dict | None = None):
        """Compute the sensitivity maps and set it.

        Parameters
        ----------
        method: callable or dict or array
            The method to use to compute the sensitivity maps.
            If an array, it should be of shape (NCoils,XYZ) and will be used as is.
            If a dict, it should have a key 'name', to determine which method to use.
            other items will be used as kwargs.
            If a callable, it should take the samples and the shape as input.
            Note that this callable function should also hold the k-space data
            (use funtools.partial)
        """
        if is_host_array(method) or is_cuda_array(method):
            self.smaps = method
            return
        if not method:
            self.smaps = None
            return
        kwargs = {}
        if isinstance(method, dict):
            kwargs = method.copy()
            method = kwargs.pop("name")
        if isinstance(method, str):
            method = get_smaps(method)
        if not callable(method):
            raise ValueError(f"Unknown smaps method: {method}")
        self.smaps, self.SOS = method(
            self.samples,
            self.shape,
            density=self.density,
            backend=self.backend,
            **kwargs,
        )

    def make_autograd(self, wrt_data=True, wrt_traj=False):
        """Make a new Operator with autodiff support.

        Parameters
        ----------
        variable: , default data
            variable on which the gradient is computed with respect to.

        wrt_data : bool, optional
            If the gradient with respect to the data is computed, default is true

        wrt_traj : bool, optional
            If the gradient with respect to the trajectory is computed, default is false

        Returns
        -------
        torch.nn.module
            A NUFFT operator with autodiff capabilities.

        Raises
        ------
        ValueError
            If autograd is not available.
        """
        if not AUTOGRAD_AVAILABLE:
            raise ValueError("Autograd not available, ensure torch is installed.")
        if not self.autograd_available:
            raise ValueError("Backend does not support auto-differentiation.")

        from mrinufft.operators.autodiff import MRINufftAutoGrad

        return MRINufftAutoGrad(self, wrt_data=wrt_data, wrt_traj=wrt_traj)

    def compute_density(self, method=None):
        """Compute the density compensation weights and set it.

        Parameters
        ----------
        method: str or callable or array or dict or bool
            The method to use to compute the density compensation.
            If a string, the method should be registered in the density registry.
            If a callable, it should take the samples and the shape as input.
            If a dict, it should have a key 'name', to determine which method to use.
            other items will be used as kwargs.
            If an array, it should be of shape (Nsamples,) and will be used as is.
            If `True`, the method `pipe` is chosen as default estimation method,
            if `backend` is `tensorflow`, `gpunufft` or `torchkbnufft-cpu`
                or `torchkbnufft-gpu`.
        """
        if isinstance(method, np.ndarray):
            self._density = method
            return None
        if not method:
            self._density = None
            return None
        if method is True:
            method = "pipe"

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
        if self._density_method is None:
            self._density_method = lambda samples, shape: method(
                samples,
                shape,
                **kwargs,
            )
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
        if self.n_coils > 1:
            tmp_op = self.__class__(
                self.samples, self.shape, density=self.density, n_coils=1, **kwargs
            )
        else:
            tmp_op = self
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
    def smaps(self, new_smaps):
        self._check_smaps_shape(new_smaps)
        self._smaps = new_smaps

    def _check_smaps_shape(self, smaps):
        """Check the shape of the sensitivity maps."""
        if smaps is None:
            self._smaps = None
        elif smaps.shape != (self.n_coils, *self.shape):
            raise ValueError(
                f"smaps shape is {smaps.shape}, it should be"
                f"(n_coils, *shape): {(self.n_coils, *self.shape)}"
            )

    @property
    def density(self):
        """Density compensation of the operator."""
        return self._density

    @density.setter
    def density(self, new_density):
        if new_density is None:
            self._density = None
        elif len(new_density) != self.n_samples:
            raise ValueError("Density and samples should have the same length")
        else:
            self._density = new_density

    @property
    def dtype(self):
        """Return floating precision of the operator."""
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype):
        self._dtype = np.dtype(new_dtype)

    @property
    def cpx_dtype(self):
        """Return complex floating precision of the operator."""
        return np.dtype(DTYPE_R2C[str(self.dtype)])

    @property
    def samples(self):
        """Return the samples used by the operator."""
        return self._samples

    @samples.setter
    def samples(self, new_samples):
        self._samples = new_samples

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

    @classmethod
    def with_autograd(cls, wrt_data=True, wrt_traj=False, *args, **kwargs):
        """Return a Fourier operator with autograd capabilities."""
        return cls(*args, **kwargs).make_autograd(wrt_data, wrt_traj)


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
        self._samples = samples.reshape(-1, len(shape))
        self.dtype = self.samples.dtype
        if n_coils < 1:
            raise ValueError("n_coils should be ≥ 1")
        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.squeeze_dims = squeeze_dims

        # Density Compensation Setup
        self.compute_density(density)
        # Multi Coil Setup
        self.compute_smaps(smaps)

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
        self.check_shape(image=data, ksp=ksp)
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
        self.check_shape(image=img, ksp=coeffs)
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

    @with_numpy_cupy
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
        grad = np.zeros_like(dataf)

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
