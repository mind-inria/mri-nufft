"""Off Resonance correction Operator wrapper.

Based on the implementation of Guillaume Daval-Frérot in pysap-mri:
https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/operators/fourier/orc_wrapper.py
"""

import math
import numpy as np

from .._utils import get_array_module

from .base import FourierOperatorBase
from .interfaces.utils import is_cuda_array

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False


TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False


def get_interpolators_from_fieldmap(
    fieldmap, t, n_time_segments=6, n_bins=(40, 10), mask=None
):
    r"""Create B and C matrices to approximate ``exp(-2j*pi*fieldmap*t)``.

    Here, B has shape ``(n_time_segments, len(t))``
    and C has shape ``(n_time_segments, *fieldmap.shape)``.

    From Sigpy: https://github.com/mikgroup/sigpy
    and MIRT (mri_exp_approx.m): https://web.eecs.umich.edu/~fessler/code/

    Parameters
    ----------
    fieldmap : np.ndarray or GPUarray
        Rate map defined as ``fieldmap = R2*_map + 1j * B0_map``.
        ``*_map`` and ``t`` should have reciprocal units.
        If ``zmap`` is real, assume ``zmap = B0_map``.
        Expected shape is ``(nz, ny, nx)``.
    t : np.ndarray or GPUarray
        Readout time in ``[s]`` of shape ``(nshots, npts)`` or ``(nshots * npts,)``.
    n_time_segments : int, optional
        Number of time segments. The default is ``6``.
    n_bins : int | Sequence[int] optional
        Number of histogram bins to use for ``(B0, T2*)``. The default is ``(40, 10)``
        If it is a scalar, assume ``n_bins = (n_bins, 10)``.
        For real fieldmap (B0 only), ``n_bins[1]`` is ignored.
    mask : np.ndarray or GPUarray, optional
        Boolean mask to avoid histogram of background values.
        The default is ``None`` (use the whole map).

    Returns
    -------
    B : np.ndarray or GPUarray
        Temporal interpolator of shape ``(n_time_segments, len(t))``.
    C : np.ndarray or GPUarray
        Off-resonance phase map at each time segment center of shape
        ``(n_time_segments, *fieldmap.shape)``.
    """
    # get backend and device
    xp = get_array_module(fieldmap)

    if xp.__name__ == "torch":
        is_torch = True
        if fieldmap.device.type == "cpu":
            xp = np
            fieldmap = fieldmap.numpy(force=True)
        else:
            assert CUPY_AVAILABLE, "GPU computation requires Cupy!"
            xp = cp
            fieldmap = cp.from_dlpack(fieldmap)
    else:
        is_torch = False

    # move t to backend
    fieldmap = xp.asarray(fieldmap, dtype=xp.complex64)
    t = xp.asarray(t, dtype=xp.float32).ravel()

    # default
    if isinstance(n_bins, (list, tuple)) is False:
        n_bins = (n_bins, 10)

    # transform to list
    n_bins = list(n_bins)

    # get field map
    if xp.isreal(fieldmap).all().item():
        r2star = None
        b0 = fieldmap
        fieldmap = 0.0 + 1j * b0
    else:
        r2star = fieldmap.real
        b0 = fieldmap.imag

    # default mask
    if mask is None:
        mask = xp.ones_like(fieldmap, dtype=bool)

    # Hz to radians / s
    fieldmap = 2 * math.pi * fieldmap

    # create histograms
    if r2star is not None:
        z = fieldmap[mask].ravel()
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
        z = fieldmap[mask].ravel()
        hk, ze = xp.histogram(z.imag, bins=n_bins[0])

        # get bin centers
        zc = ze[1:] - (ze[1] - ze[0]) / 2

        # complexify
        zk = 0 + 1j * zc  # [K 1]

    # flatten histogram values and centers
    hk = hk.ravel()
    zk = zk.ravel()

    # generate time for each segment
    tl = xp.linspace(
        t.min(), t.max(), n_time_segments, dtype=xp.float32
    )  # time seg centers in [s]

    # prepare for basis calculation
    ch = xp.exp(-tl[:, None, ...] @ zk[None, ...])
    w = xp.diag(hk**0.5)
    p = xp.linalg.pinv(w @ _transpose(ch)) @ w

    # actual temporal basis calculation
    B = p @ xp.exp(-zk[:, None, ...] * t[None, ...])
    B = B.astype(xp.complex64)

    # get spatial coeffs
    C = xp.exp(-tl * fieldmap[..., None])
    C = C[None, ...].swapaxes(0, -1)[
        ..., 0
    ]  # (..., n_time_segments) -> (n_time_segments, ...)

    # clean-up of spatial coeffs
    C = xp.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    # back to torch if required
    if is_torch:
        if xp.__name__ == "cupy":
            B = torch.from_dlpack(B)
            C = torch.from_dlpack(C)
        else:
            B = torch.from_numpy(B)
            C = torch.from_numpy(C)

    return B, C


def _outer_sum(xx, yy):
    xx = xx[:, None, ...]  # add a singleton dimension at axis 1
    yy = yy[None, ...]  # add a singleton dimension at axis 0
    ss = xx + yy  # compute the outer sum
    return ss


def _transpose(input):
    xp = get_array_module(input)
    if xp.__name__ == "torch":
        return input.mT
    else:
        return input.T


class MRIFourierCorrected(FourierOperatorBase):
    """Fourier Operator with B0 Inhomogeneities compensation.

    This is a wrapper around the Fourier Operator to compensate for the
    B0 inhomogeneities  in the  k-space.

    Parameters
    ----------
    fourier_op: object of class FourierBase
        the fourier operator to wrap
    fieldmap : np.ndarray or GPUarray, optional
        Rate map defined as ``fieldmap = R2*_map + 1j * B0_map``.
        ``*_map`` and ``t`` should have reciprocal units.
        If ``zmap`` is real, assume ``zmap = B0_map``.
        Expected shape is ``(nz, ny, nx)``.
    t : np.ndarray or GPUarray, optional
        Readout time in ``[s]`` of shape ``(nshots, npts)`` or ``(nshots * npts,)``.
    n_time_segments : int, optional
        Number of time segments. The default is ``6``.
    n_bins : int | Sequence[int] optional
        Number of histogram bins to use for ``(B0, T2*)``. The default is ``(40, 10)``
        If it is a scalar, assume ``n_bins = (n_bins, 10)``.
        For real fieldmap (B0 only), ``n_bins[1]`` is ignored.
    mask : np.ndarray or GPUarray, optional
        Boolean mask to avoid histogram of background values.
        The default is ``None`` (use the whole map).
    B : np.ndarray or GPUarray, optional
        Temporal interpolator of shape ``(n_time_segments, len(t))``.
    C : np.ndarray or GPUarray, optional
        Off-resonance phase map at each time segment center of shape
        ``(n_time_segments, *fieldmap.shape)``.
    backend: str, optional
        The backend to use for computations. Either 'cpu', 'gpu' or 'torch'.
        The default is `cpu`.
    """

    def __init__(
        self,
        fourier_op,
        fieldmap=None,
        t=None,
        n_time_segments=6,
        n_bins=(40, 10),
        mask=None,
        B=None,
        C=None,
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

        # if not fourier_op.uses_sense:
        # raise ValueError("please use smaps.")

        # self.n_samples = fourier_op.n_samples
        self.n_coils = fourier_op.n_coils
        self.shape = fourier_op.shape
        self.smaps = fourier_op.smaps

        if B is not None and C is not None:
            self.B = self.xp.asarray(B)
            self.C = self.xp.asarray(C)
        else:
            fieldmap = self.xp.asarray(fieldmap)
            self.B, self.C = get_interpolators_from_fieldmap(
                fieldmap, t, n_time_segments, n_bins, mask
            )
        if self.B is None or self.C is None:
            raise ValueError("Please either provide fieldmap and t or B and C")
        self.n_interpolators = len(self.C)

    def op(self, data, *args):
        """Compute Forward Operation with off-resonance effect.

        Parameters
        ----------
        x: numpy.ndarray or cupy.ndarray
            N-D input image

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            masked distorded N-D k-space
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
        x: numpy.ndarray or cupy.ndarray
            masked distorded N-D k-space

        Returns
        -------
            inverse Fourier transform of the distorded input k-space.
        """
        y = 0.0
        coeffs_d = self.xp.array(coeffs)
        for idx in range(self.n_interpolators):
            y += self.xp.conj(self.C[idx]) * self._fourier_op.adj_op(
                self.xp.conj(self.B[idx]) * coeffs_d, *args
            )

        return y

    def get_grad(self, image_data, obs_data):
        """Compute the data consistency error.

        Parameters
        ----------
        image_data: numpy.ndarray or cupy.ndarray
            N-D input image
        obs_data: numpy.ndarray or cupy.ndarray
            N-D observed k-space

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            data consistency error in image space.
        """
        return self.adj_op(self.op(image_data) - obs_data)
