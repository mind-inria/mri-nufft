"""Off Resonance correction Operator wrapper."""

from collections.abc import Callable
from scipy.ndimage import zoom
import warnings

from mrinufft._array_compat import with_numpy_cupy, CUPY_AVAILABLE
import numpy as np
from numpy.typing import NDArray

from .._utils import get_array_module
from ..extras.field_map import get_orc_factorization, get_complex_fieldmap_rad
from .base import FourierOperatorBase, power_method

if CUPY_AVAILABLE:
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

    See Also
    --------
    :ref:`_nufft-orc`

    """

    def __init__(
        self,
        fourier_op,
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

        if readout_time.size != self.n_samples:
            n_shot, r = divmod(self.n_samples, readout_time.size)
            if r != 0:
                raise ValueError(
                    "readout_time should divide or equal the size of the samples."
                )
            self.n_shots = n_shot
        complex_field_map = None
        if b0_map is not None:
            complex_field_map = get_complex_fieldmap_rad(b0_map, r2star_map)

        self.B, self.C = self.compute_interpolator(
            interpolator, complex_field_map, readout_time.ravel(), mask
        )

    def compute_interpolator(
        self, interpolators, field_map, readout_time, mask
    ) -> tuple[NDArray, NDArray]:
        """Decompose the field-map in space and time-wise interpolators."""
        if isinstance(interpolators, tuple):
            B, C = interpolators
            try:
                _ = get_array_module(B)
            except ValueError as e:
                raise ValueError(
                    "Provide a tuple of 2 array_like data"
                    " for space and time interpolators"
                ) from e
            Bl, Bs = B.shape
            Cl, Cxyz = C.shape
            if Bl != Cl or self.n_samples % Bs or Cxyz != self.shape:
                raise ValueError("Interpolator shapes should be (k*Ns, L) and (L,*XYZ)")
            return B, C
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
            )
            if mask is not None:
                mask = zoom_func(
                    mask,
                    zoom=tuple(np.array(self._fourier_op.shape) / np.array(mask.shape)),
                )

        kwargs = {}
        if isinstance(interpolators, dict):
            kwargs = interpolators.copy()
            interpolators = kwargs.pop("name")
        if isinstance(interpolators, str):
            interpolators = get_orc_factorization(interpolators)
        if not isinstance(interpolators, Callable):
            raise ValueError(f"Unknown off-resonance interpolator ``{interpolators}``")
        B, C, _ = interpolators(
            field_map=field_map, readout_time=readout_time, mask=mask, **kwargs
        )
        return B, C

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
        xp = get_array_module(data)
        y = xp.zeros((self.n_batchs, self.n_coils, self.n_samples), dtype=xp.complex64)

        ns = self.n_samples_per_shot
        data_d = xp.asarray(data)
        for ll in range(self.n_interpolators):
            ytmp = self._fourier_op.op(self.C[ll] * data_d, *args)
            # repeat B over shots
            for s in range(self.n_shots):
                # TODO use reshape and multiply with broadcasting
                y[:, :, s * ns : (s + 1) * ns] += (
                    self.B[:, ll] * ytmp[:, :, s * ns : (s + 1) * ns]
                )
        return y

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
        xp = get_array_module(coeffs)
        B, C = self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        NS,NK = self.n_shots, self.n_samples_per_shot
        ytmp = xp.zeros((B, C, K), dtype=xp.complex64)
        img = xp.zeros((B, C, *XYZ), dtype=xp.complex64)

        coeffs = coeffs.reshape(B, C, NS, NK)
        ns = self.n_samples_per_shot
        for ll in range(self.n_interpolators):
            Bconj = self.B[:, ll].conj()
            Cconj = self.C[ll].conj()
            ytmp = (Bconj * coeffs).reshape(B, C, K)
            img += Cconj * self._fourier_op.adj_op(ytmp)
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
        n_coils = self._fourier_op.n_coils
        smaps = self._fourier_op.smaps
        squeeze_dims = self.squeeze_dims

        self._fourier_op.smaps = None
        self._fourier_op.n_coils = 1

        x = 1j * np.random.random(self.shape).astype(self.cpx_dtype, copy=False)
        x += np.random.random(self.shape).astype(self.cpx_dtype, copy=False)

        x = np.asarray(x)
        lipschitz_cst = power_method(
            max_iter, self, norm_func=lambda x: np.linalg.norm(x.flatten()), x=x
        )

        # restore coil setup
        self._fourier_op.n_coils = n_coils
        self._fourier_op.smaps = smaps

        return lipschitz_cst

        
    
    @property
    def n_interpolators(self):
        return self.B.shape[1]

    @property
    def n_samples_per_shot(self):
        return self.B.shape[0]

