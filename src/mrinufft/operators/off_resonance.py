"""Off Resonance correction Operator wrapper."""

from collections.abc import Callable
from scipy.ndimage import zoom
import warnings

from mrinufft._array_compat import with_numpy_cupy, CUPY_AVAILABLE
import numpy as np
from numpy.typing import NDArray

from .._array_compat import get_array_module, is_host_array, is_cuda_array
from ..extras.field_map import get_orc_factorization, get_complex_fieldmap_rad
from .base import FourierOperatorBase, power_method, get_operator

if CUPY_AVAILABLE:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as cp_zoom


class MRIFourierCorrected(FourierOperatorBase):
    """Fourier Operator with B0 Inhomogeneities compensation.

    This is a wrapper around the Fourier Operator to compensate for the
    B0 inhomogeneities in  the k-space.

    Parameters
    ----------
    fourier_op: FourierOperatorBase
        Existing NUFFT operator.
    b0_map : NDArray
        Static field inhomogeneities map.
        ``b0_map`` and ``readout_time`` should have reciprocal units. (e.g. [Hz]
        and [s]) Also supports Cupy arrays and Torch tensors.
    readout_time : NDArray
        Readout time in ``[s]`` of shape ``(n_shots, n_pts)`` or ``(n_shots *
        n_pts,)``. Also supports Cupy arrays and Torch tensors.
    mask : NDArray, optional
        Boolean mask of the region of interest
        (e.g., corresponding to the imaged object).
        This is used to exclude the background field-map values
        from histogram computation.
        The default is ``None`` (use the whole map).
        Also supports Cupy arrays and Torch tensors.
    r2star_map : NDArray, optional
        Effective transverse relaxation map (R2*). ``r2star_map`` and
        ``readout_time`` should have reciprocal units (e.g. [Hz] and [s]). Must
        have same shape as ``b0_map``. The default is ``None`` (purely imaginary
        field). Also supports Cupy arrays and Torch tensors.
    interpolators: str, dict, tuple[NDArray, NDArray]
        Determine how to decompose the field-map.

        - If ``str``, use an existing method in `extra/field_map.py` with
          default parameters
        - If ``{"name":name, **kwargs}`` use an existing
          method in `extra_field_map.py` parameterize by kwargs.
        - If``tuple[NDArray, NDArray]`` use this directly as the decomposition
          (B and C)

    Notes
    -----
    The total field map used to calculate the field coefficients is
    ``field_map = R2*_map + 1j * B0_map``. If R2* is not provided,
    the field is purely imaginary: ``field_map = 1j * B0_map``.

    You can also  use the method :py:func:`.with_off_resonance_correction
    <mrinufft.operators.base.FourierOperatorBase.with_off_resonance_correction>`
    to augment an existing operator with off-resonance correction capability.


    See Also
    --------
    :ref:`nufft-orc`

    """

    def __init__(
        self,
        fourier_op: FourierOperatorBase,
        b0_map: NDArray | None = None,
        readout_time: NDArray | None = None,
        r2star_map: NDArray | None = None,
        mask: NDArray | None = None,
        interpolator: str | dict | tuple[NDArray, NDArray] = "svd",
    ):
        self._fourier_op = fourier_op
        self.squeeze_dims = self._fourier_op.squeeze_dims
        self._fourier_op.squeeze_dims = False  # we will manage shapes here.

        if (
            b0_map is None
            and readout_time is None
            and not isinstance(interpolator, tuple)
        ):
            raise ValueError(
                "b0_map  and readout_time required for off-resonance correction."
            )

        complex_field_map = None
        if b0_map is not None:
            complex_field_map = get_complex_fieldmap_rad(b0_map, r2star_map)

        self.compute_interpolator(interpolator, complex_field_map, readout_time, mask)

    def compute_interpolator(
        self,
        interpolators: str | dict | tuple[NDArray, NDArray],
        field_map: NDArray | None,
        readout_time: NDArray | None,
        mask: NDArray | None,
    ):
        """Decompose the field-map in space and time-wise interpolators.

        Sets the B and C attributes.
        """
        if isinstance(interpolators, tuple):
            B, C = interpolators
            try:
                _ = get_array_module(B)
            except ValueError as e:
                raise ValueError(
                    "Provide a tuple of 2 array_like data"
                    " for space and time interpolators"
                ) from e

            if B.size != self.n_samples:
                n_shot, r = divmod(self.n_samples, B.shape[0])
                if r != 0:
                    raise ValueError(
                        "Time interpolator should divide or equal size of the samples."
                    )
                self.n_shots = n_shot
            self.B, self.C = B, C
            return

        readout_time = readout_time.ravel()
        self.n_shots = 1
        if readout_time.size != self.n_samples:
            n_shot, r = divmod(self.n_samples, readout_time.size)
            if r != 0:
                raise ValueError(
                    "readout_time should divide or equal the size of the samples."
                )
            self.n_shots = n_shot
        # Resize to match fourier shape
        if field_map.shape != self._fourier_op.shape:
            warnings.warn(
                "field_map and mask will be interpolated to match image shape."
            )
            xp = get_array_module(field_map)
            zoom_func = cp_zoom if xp.__name__ == "cupy" else zoom
            field_map = zoom_func(
                field_map,
                zoom=tuple(
                    np.array(self._fourier_op.shape) / np.array(field_map.shape)
                ),
                order=1,
            )
            if mask is not None:
                mask = zoom_func(
                    mask,
                    zoom=tuple(np.array(self._fourier_op.shape) / np.array(mask.shape)),
                    order=0,
                )

        kwargs = {}
        if isinstance(interpolators, dict):
            kwargs = interpolators.copy()
            interpolators = kwargs.pop("name")
        if isinstance(interpolators, str):
            interpolators = get_orc_factorization(interpolators)
        if not isinstance(interpolators, Callable):
            raise ValueError(f"Unknown off-resonance interpolator ``{interpolators}``")

        self.B, self.C, _ = interpolators(
            field_map=field_map, readout_time=readout_time, mask=mask, **kwargs
        )

    def __getattr__(self, name):
        """Delegate attribute to internal operator."""
        return getattr(self._fourier_op, name)

    @with_numpy_cupy
    def op(self, data, *args):
        """Compute Forward Operation with off-resonance effect.

        Parameters
        ----------
        data: NDArray
            N-D input image.

        Returns
        -------
        NDArray
            Masked distorted N-D k-space.
            Array module is the same as input data.
        """
        B, C, K = self.n_batchs, self.n_coils, self.n_samples
        NS, NK = self.n_shots, self.n_samples_per_shot

        on_gpu = is_cuda_array(data)
        if is_host_array(self.B) and on_gpu:
            warnings.warn("Interpolators are on CPU, moving GPU image data to CPU.")
            data = data.get()
        elif is_cuda_array(self.B) and not on_gpu:
            warnings.warn("Interpolators are on GPU, moving CPU image data to GPU")
            data = cp.array(data, copy=False)

        xp = get_array_module(self.B)
        y = xp.zeros((self.n_batchs, self.n_coils, self.n_samples), dtype=xp.complex64)

        data_d = xp.asarray(data)
        for ll in range(self.n_interpolators):
            ytmp = self._fourier_op.op(self.C[ll] * data_d, *args).reshape(B, C, NS, NK)
            y += (self.B[:, ll] * ytmp).reshape(B, C, K)

        if on_gpu:
            y = cp.array(y, copy=False)
        elif is_cuda_array(y):
            y = y.get()
        return self._safe_squeeze(y)

    @with_numpy_cupy
    def adj_op(self, coeffs, *args):
        """
        Compute Adjoint Operation with off-resonance effect.

        Parameters
        ----------
        coeffs: NDArray
            k-space data

        Returns
        -------
        NDArray
            Inverse Fourier transform of the distorted input k-space.
            Array module is the same as input coeffs.

        """
        xp = get_array_module(self.B)
        B, C = self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        NS, NK = self.n_shots, self.n_samples_per_shot
        ytmp = xp.zeros((B, C, K), dtype=xp.complex64)
        on_gpu = is_cuda_array(coeffs)
        if is_host_array(self.B) and on_gpu:
            coeffs = coeffs.get()
        elif is_cuda_array(self.B) and not on_gpu:
            coeffs = cp.array(coeffs, copy=False)

        if not self.uses_sense:
            img = xp.zeros((B, C, *XYZ), dtype=xp.complex64)
        else:
            img = xp.zeros((B, 1, *XYZ), dtype=xp.complex64)
        coeffs = coeffs.reshape(B, C, NS, NK)
        for ll in range(self.n_interpolators):
            Bconj = self.B[:, ll].conj()
            Cconj = self.C[ll].conj()
            ytmp = (Bconj * coeffs).reshape(B, C, K)
            img += Cconj * self._fourier_op.adj_op(ytmp)

        if on_gpu:
            img = cp.array(img, copy=False)
        elif is_cuda_array(img):
            img = img.get()
        return self._safe_squeeze(img)

    def get_lipschitz_cst(self, max_iter=10, **kwargs):
        """Return the Lipschitz constant of the operator.

        Parameters
        ----------
        max_iter: int
            Number of iteration to perform to estimate the Lipschitz constant.
        kwargs:
            Extra kwargs for the cufinufft operator.

        Returns
        -------
        float
            Lipschitz constant of the operator.
        """
        # Disable coil dimension for faster computation
        # For gpuNUFFT we have to create a new operator,
        # because coils are managed at C++ levels.
        if self._fourier_op.backend == "gpunufft":
            old_fourier_op = self._fourier_op
            self._fourier_op = get_operator("gpunufft")(
                samples=self.samples,
                shape=self.shape,
                density=self._fourier_op.density,
                n_coils=1,
            )
        else:
            n_coils = self._fourier_op.n_coils
            smaps = self._fourier_op.smaps

            self._fourier_op.smaps = None
            self._fourier_op.n_coils = 1
        x = 1j * np.random.random(self.shape).astype(self.cpx_dtype, copy=False)
        x += np.random.random(self.shape).astype(self.cpx_dtype, copy=False)

        x = np.asarray(x)
        lipschitz_cst, _ = power_method(
            max_iter, self, norm_func=lambda x: np.linalg.norm(x.flatten()), x=x
        )

        if self._fourier_op.backend == "gpunufft":
            self._fourier_op = old_fourier_op
        else:
            # restore coil setup
            self._fourier_op.n_coils = n_coils
            self._fourier_op.smaps = smaps

        return lipschitz_cst

    @property
    def n_interpolators(self):
        """Number of interpolators used to approximate the off-resonance effects."""
        return self.B.shape[1]

    @property
    def n_samples_per_shot(self):
        """Number of time points in a shot."""
        return self.B.shape[0]
