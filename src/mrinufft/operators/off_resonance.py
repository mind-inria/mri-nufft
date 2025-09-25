"""Off Resonance correction Operator wrapper."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .._utils import get_array_module
from ..extras.field_map import get_orc_factorization, get_complex_fieldmap_rad
from .base import FourierOperatorBase


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
        if (
            b0_map is None
            and readout_time is None
            and not isinstance(interpolator, tuple)
        ):
            raise ValueError(
                "b0_map  and readout_time required for off-resonance correction."
            )

        if readout_time.size != self.n_samples:
            raise ValueError(
                "readout_time should match number of samples of the operator."
            )

        complex_field_map = get_complex_fieldmap_rad(b0_map, r2star_map)
        self.B, self.C = self.compute_interpolator(
            interpolator, complex_field_map, readout_time, mask
        )

    @staticmethod
    def compute_interpolator(
        interpolators, field_map, time_vec, mask
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
            return B, C
        kwargs = {}
        if isinstance(interpolators, dict):
            kwargs = interpolators.copy()
            interpolators = kwargs.pop("name")
        if isinstance(interpolators, str):
            interpolators = get_orc_factorization(interpolators)
        if not isinstance(interpolators, Callable):
            raise ValueError(f"Unknown off-resonance interpolator ``{interpolators}``")
        B, C = interpolators(
            field_map=field_map, timve_vec=time_vec, mask=mask, **kwargs
        )
        return B, C

    def __getattr__(self, name):
        """Delegate attribute to internal operator."""
        return getattr(self._fourier_op, name)

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
        y = 0.0
        data_d = self.xp.asarray(data)
        for idx in range(self.n_interpolators):
            y += self.B[idx] * self._fourier_op.op(self.C[idx] * data_d, *args)

        return y

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
        y = 0.0
        coeffs_d = self.xp.asarray(coeffs)
        for idx in range(self.n_interpolators):
            y += self.xp.conj(self.C[idx]) * self._fourier_op.adj_op(
                self.xp.conj(self.B[idx]) * coeffs_d, *args
            )

        return y
