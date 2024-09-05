"""Off Resonance correction Operator wrapper.

Based on the implementation of Guillaume Daval-Frérot in pysap-mri:
https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/operators/fourier/orc_wrapper.py
"""

import math
import numpy as np

from .._utils import get_array_module

from .base import (
    FourierOperatorBase,
    CUPY_AVAILABLE,
    AUTOGRAD_AVAILABLE,
    with_numpy_cupy,
)
from .interfaces.utils import is_cuda_array

if CUPY_AVAILABLE:
    import cupy as cp

if AUTOGRAD_AVAILABLE:
    import torch


@with_numpy_cupy
def get_interpolators_from_fieldmap(
    b0_map, readout_time, n_time_segments=6, n_bins=(40, 10), mask=None, r2star_map=None
):
    r"""Approximate ``exp(-2j*pi*fieldmap*readout_time) ≈ Σ B_n(t)C_n(r)``.

    Here, B_n(t) are n_time_segments temporal coefficients and C_n(r)
    are n_time_segments temporal spatial coefficients.

    The matrix B has shape ``(n_time_segments, len(readout_time))``
    and C has shape ``(n_time_segments, *b0_map.shape)``.

    From Sigpy: https://github.com/mikgroup/sigpy
    and MIRT (mri_exp_approx.m): https://web.eecs.umich.edu/~fessler/code/

    Parameters
    ----------
    b0_map : np.ndarray
        Static field inhomogeneities map.
        ``b0_map`` and ``readout_time`` should have reciprocal units.
        Also supports Cupy arrays and Torch tensors.
    readout_time : np.ndarray
        Readout time in ``[s]`` of shape ``(n_shots, n_pts)`` or ``(n_shots * n_pts,)``.
        Also supports Cupy arrays and Torch tensors.
    n_time_segments : int, optional
        Number of time segments. The default is ``6``.
    n_bins : int | Sequence[int] optional
        Number of histogram bins to use for ``(B0, T2*)``. The default is ``(40, 10)``
        If it is a scalar, assume ``n_bins = (n_bins, 10)``.
        For real fieldmap (B0 only), ``n_bins[1]`` is ignored.
    mask : np.ndarray, optional
        Boolean mask of the region of interest
        (e.g., corresponding to the imaged object).
        This is used to exclude the background fieldmap values
        from histogram computation. Must have same shape as ``b0_map``.
        The default is ``None`` (use the whole map).
        Also supports Cupy arrays and Torch tensors.
    r2star_map : np.ndarray, optional
        Effective transverse relaxation map (R2*).
        ``r2star_map`` and ``readout_time`` should have reciprocal units.
        Must have same shape as ``b0_map``.
        The default is ``None`` (purely imaginary field).
        Also supports Cupy arrays and Torch tensors.

    Notes
    -----
    The total field map used to calculate the field coefficients is
    ``field_map = R2*_map + 1j * B0_map``. If R2* is not provided,
    the field is purely immaginary: ``field_map = 1j * B0_map``.

    Returns
    -------
    B : np.ndarray
        Temporal interpolator of shape ``(n_time_segments, len(t))``.
        Array module is the same as input field_map.
    tl : np.ndarray
        Time segment centers of shape ``(n_time_segments,)``.
        Array module is the same as input field_map.

    """
    # default
    if isinstance(n_bins, (list, tuple)) is False:
        n_bins = (n_bins, 10)
    n_bins = list(n_bins)

    # get backend and device
    xp = get_array_module(b0_map)

    # enforce data types
    b0_map = xp.asarray(b0_map, dtype=xp.float32)
    readout_time = xp.asarray(readout_time, dtype=xp.float32).ravel()
    if mask is None:
        mask = xp.ones_like(b0_map, dtype=bool)
    else:
        mask = xp.asarray(mask, dtype=bool)
    if r2star_map is not None:
        r2star_map = xp.asarray(r2star_map, dtype=xp.float32)

    # Hz to radians / s
    field_map = _get_complex_fieldmap(b0_map, r2star_map)

    # create histograms
    z = field_map[mask].ravel()

    if r2star_map is not None:
        z = xp.stack((z.imag, z.real), axis=1)
        hk, ze = xp.histogramdd(z, bins=n_bins)
        ze = list(ze)

        # get bin centers
        zc = [e[1:] - (e[1] - e[0]) / 2 for e in ze]

        # complexify
        zk = _outer_sum(1j * zc[0], zc[1])  # [K1 K2]
        zk = zk.T
        hk = hk.T
    else:
        hk, ze = xp.histogram(z.imag, bins=n_bins[0])

        # get bin centers
        zc = ze[1:] - (ze[1] - ze[0]) / 2

        # complexify
        zk = 1j * zc  # [K 1]

    # flatten histogram values and centers
    hk = hk.ravel()
    zk = zk.ravel()

    # generate time for each segment
    tl = xp.linspace(
        readout_time.min(), readout_time.max(), n_time_segments, dtype=xp.float32
    )  # time seg centers in [s]

    # prepare for basis calculation
    ch = xp.exp(-tl[:, None, ...] @ zk[None, ...])
    w = xp.diag(hk**0.5)
    p = xp.linalg.pinv(w @ ch.T) @ w

    # actual temporal basis calculation
    B = p @ xp.exp(-zk[:, None, ...] * readout_time[None, ...])
    B = B.astype(xp.complex64)

    return B, tl


def _outer_sum(xx, yy):
    xx = xx[:, None, ...]  # add a singleton dimension at axis 1
    yy = yy[None, ...]  # add a singleton dimension at axis 0
    ss = xx + yy  # compute the outer sum
    return ss


class MRIFourierCorrected(FourierOperatorBase):
    """Fourier Operator with B0 Inhomogeneities compensation.

    This is a wrapper around the Fourier Operator to compensate for the
    B0 inhomogeneities in  the k-space.

    Parameters
    ----------
    b0_map : np.ndarray
        Static field inhomogeneities map.
        ``b0_map`` and ``readout_time`` should have reciprocal units.
        Also supports Cupy arrays and Torch tensors.
    readout_time : np.ndarray
        Readout time in ``[s]`` of shape ``(n_shots, n_pts)`` or ``(n_shots * n_pts,)``.
        Also supports Cupy arrays and Torch tensors.
    n_time_segments : int, optional
        Number of time segments. The default is ``6``.
    n_bins : int | Sequence[int] optional
        Number of histogram bins to use for ``(B0, T2*)``. The default is ``(40, 10)``
        If it is a scalar, assume ``n_bins = (n_bins, 10)``.
        For real fieldmap (B0 only), ``n_bins[1]`` is ignored.
    mask : np.ndarray, optional
        Boolean mask of the region of interest
        (e.g., corresponding to the imaged object).
        This is used to exclude the background fieldmap values
        from histogram computation.
        The default is ``None`` (use the whole map).
        Also supports Cupy arrays and Torch tensors.
    B : np.ndarray, optional
        Temporal interpolator of shape ``(n_time_segments, len(readout_time))``.
    tl : np.ndarray, optional
        Time segment centers of shape ``(n_time_segments,)``.
        Also supports Cupy arrays and Torch tensors.
    r2star_map : np.ndarray, optional
        Effective transverse relaxation map (R2*).
        ``r2star_map`` and ``readout_time`` should have reciprocal units.
        Must have same shape as ``b0_map``.
        The default is ``None`` (purely imaginary field).
        Also supports Cupy arrays and Torch tensors.
    backend: str, optional
        The backend to use for computations. Either 'cpu', 'gpu' or 'torch'.
        The default is `cpu`.

    Notes
    -----
    The total field map used to calculate the field coefficients is
    ``field_map = R2*_map + 1j * B0_map``. If R2* is not provided,
    the field is purely immaginary: ``field_map = 1j * B0_map``.

    """

    def __init__(
        self,
        fourier_op,
        b0_map=None,
        readout_time=None,
        n_time_segments=6,
        n_bins=(40, 10),
        mask=None,
        r2star_map=None,
        B=None,
        tl=None,
        backend="cpu",
    ):
        if backend == "gpu" and not CUPY_AVAILABLE:
            raise RuntimeError("Cupy is required for gpu computations.")
        if backend == "torch":
            self.xp = torch
        if backend == "gpu":
            self.xp = cp
        elif backend == "cpu":
            self.xp = np
        else:
            raise ValueError("Unsupported backend.")
        self._fourier_op = fourier_op

        self.n_coils = fourier_op.n_coils
        self.shape = fourier_op.shape
        self.smaps = fourier_op.smaps
        self.autograd_available = fourier_op.autograd_available

        if B is not None and tl is not None:
            self.B = self.xp.asarray(B)
            self.tl = self.xp.asarray(tl)
        else:
            if b0_map is None or readout_time is None:
                raise ValueError("Please either provide fieldmap and t or B and tl")
            b0_map = self.xp.asarray(b0_map)
            self.B, self.tl = get_interpolators_from_fieldmap(
                b0_map,
                readout_time,
                n_time_segments,
                n_bins,
                mask,
                r2star_map,
            )
        if self.B is None or self.tl is None:
            raise ValueError("Please either provide fieldmap and t or B and tl")
        self.n_interpolators = self.B.shape[0]

        # create spatial interpolator
        field_map = _get_complex_fieldmap(b0_map, r2star_map)
        if is_cuda_array(b0_map):
            self.C = None
            self.field_map = field_map
        else:
            self.C = _get_spatial_coefficients(field_map, self.tl)
            self.field_map = None

    def op(self, data, *args):
        """Compute Forward Operation with off-resonance effect.

        Parameters
        ----------
        x: numpy.ndarray
            N-D input image.
            Also supports Cupy arrays and Torch tensors.

        Returns
        -------
        numpy.ndarray
            Masked distorted N-D k-space.
            Array module is the same as input data.

        """
        y = 0.0
        data_d = self.xp.asarray(data)
        if self.C is not None:
            for idx in range(self.n_interpolators):
                y += self.B[idx] * self._fourier_op.op(self.C[idx] * data_d, *args)
        else:
            for idx in range(self.n_interpolators):
                C = self.xp.exp(-self.field_map * self.tl[idx].item())
                y += self.B[idx] * self._fourier_op.op(C * data_d, *args)

        return y

    def adj_op(self, coeffs, *args):
        """
        Compute Adjoint Operation with off-resonance effect.

        Parameters
        ----------
        x: numpy.ndarray
            Masked distorted N-D k-space.
            Also supports Cupy arrays and Torch tensors.


        Returns
        -------
        numpy.ndarray
            Inverse Fourier transform of the distorted input k-space.
            Array module is the same as input coeffs.

        """
        y = 0.0
        coeffs_d = self.xp.array(coeffs)
        if self.C is not None:
            for idx in range(self.n_interpolators):
                y += self.xp.conj(self.C[idx]) * self._fourier_op.adj_op(
                    self.xp.conj(self.B[idx]) * coeffs_d, *args
                )
        else:
            for idx in range(self.n_interpolators):
                C = self.xp.exp(-self.field_map * self.tl[idx].item())
                y += self.xp.conj(C) * self._fourier_op.adj_op(
                    self.xp.conj(self.B[idx]) * coeffs_d, *args
                )

        return y

    @staticmethod
    def get_spatial_coefficients(field_map, tl):
        """Compute spatial coefficients for field approximation.

        Parameters
        ----------
        field_map : np.ndarray
            Total field map used to calculate the field coefficients is
            ``field_map = R2*_map + 1j * B0_map``.
            Also supports Cupy arrays and Torch tensors.
        tl : np.ndarray
            Time segment centers of shape ``(n_time_segments,)``.
            Also supports Cupy arrays and Torch tensors.

        Returns
        -------
        C : np.ndarray
            Off-resonance phase map at each time segment center of shape
            ``(n_time_segments, *field_map.shape)``.
            Array module is the same as input field_map.

        """
        return _get_spatial_coefficients(field_map, tl)


def _get_complex_fieldmap(b0_map, r2star_map=None):
    xp = get_array_module(b0_map)

    if r2star_map is not None:
        r2star_map = xp.asarray(r2star_map, dtype=xp.float32)
        field_map = 2 * math.pi * (r2star_map + 1j * b0_map)
    else:
        field_map = 2 * math.pi * 1j * b0_map

    return field_map


def _get_spatial_coefficients(field_map, tl):
    xp = get_array_module(field_map)

    # get spatial coeffs
    C = xp.exp(-tl * field_map[..., None])
    C = C[None, ...].swapaxes(0, -1)[
        ..., 0
    ]  # (..., n_time_segments) -> (n_time_segments, ...)
    C = xp.asarray(C, dtype=xp.complex64)

    # clean-up of spatial coeffs
    C = xp.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    return C
