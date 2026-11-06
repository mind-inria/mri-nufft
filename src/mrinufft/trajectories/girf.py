"""Gradient impulse response function (GIRF) model of the gradient chain."""

from __future__ import annotations

from math import comb, factorial, isqrt, prod

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

from .utils import (
    Acquisition,
    convert_gradients_to_trajectory,
    convert_trajectory_to_gradients,
)


def spherical_harmonics(positions: NDArray, n_terms: int = 16) -> NDArray:
    """Evaluate the spherical harmonics of the GIRF terms.

    The terms follow the order and the unnormalized polynomials of the Skope
    field cameras: ``(1, x, y, z, xy, zy, 3z^2 - r^2, xz, x^2 - y^2, ...)``, up to
    the third order. See :ref:`girf` for the full list.

    Parameters
    ----------
    positions : NDArray
        Positions in m, with shape ``(..., 3)``, or ``(..., 2)`` for ``z = 0``.
    n_terms : int, optional
        Number of terms, up to 16. The default is 16.

    Returns
    -------
    NDArray
        Spherical harmonics, with shape ``(..., n_terms)``.
    """
    if not 1 <= n_terms <= 16:
        raise ValueError("`n_terms` must be between 1 and 16.")
    positions = np.asarray(positions, dtype=float)
    x, y = positions[..., 0], positions[..., 1]
    z = positions[..., 2] if positions.shape[-1] == 3 else np.zeros_like(x)
    terms = [np.ones_like(x), x, y, z]
    if n_terms > 4:
        r2 = x**2 + y**2 + z**2
        terms += [x * y, z * y, 3 * z**2 - r2, x * z, x**2 - y**2]
    if n_terms > 9:
        terms += [
            3 * y * x**2 - y**3,
            x * y * z,
            5 * y * z**2 - y * r2,
            5 * z**3 - 3 * z * r2,
            5 * x * z**2 - x * r2,
            z * x**2 - z * y**2,
            x**3 - 3 * x * y**2,
        ]
    return np.stack(terms[:n_terms], axis=-1)


def _image_frame(
    acq: Acquisition, girf_affine: NDArray | None, dim: int
) -> tuple[NDArray, NDArray]:
    """Rotation and origin from image to GIRF coordinates.

    Image coordinates are in m, on the axes of the image, from its voxel
    ``img_size // 2``, where the phase of the NUFFT is zero.
    """
    scale = np.ones(3)
    scale[:dim] = acq.res[:dim]
    centre = np.zeros(4)
    centre[:dim] = np.asarray(acq.img_size[:dim]) // 2
    centre[3] = 1
    if acq.affine is None:
        affine = np.diag([*scale, 1.0])
        affine[:3, 3] = -scale * centre[:3]
    else:
        affine = np.asarray(acq.affine, dtype=float)
    if girf_affine is not None:
        affine = np.asarray(girf_affine, dtype=float) @ affine
    rotation = affine[:3, :3] / scale
    if dim == 2:
        rotation[:, 2] /= np.linalg.norm(rotation[:, 2])
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(
            "The affines must be rigid, with voxel sizes `acq.fov / acq.img_size`."
        )
    return rotation, (affine @ centre)[:3]


# Monomial coefficients {(px, py, pz): c} of the terms of `spherical_harmonics`.
_MONOMIALS = (
    {(0, 0, 0): 1},
    {(1, 0, 0): 1},
    {(0, 1, 0): 1},
    {(0, 0, 1): 1},
    {(1, 1, 0): 1},
    {(0, 1, 1): 1},
    {(0, 0, 2): 2, (2, 0, 0): -1, (0, 2, 0): -1},
    {(1, 0, 1): 1},
    {(2, 0, 0): 1, (0, 2, 0): -1},
    {(2, 1, 0): 3, (0, 3, 0): -1},
    {(1, 1, 1): 1},
    {(0, 1, 2): 4, (2, 1, 0): -1, (0, 3, 0): -1},
    {(0, 0, 3): 2, (2, 0, 1): -3, (0, 2, 1): -3},
    {(1, 0, 2): 4, (3, 0, 0): -1, (1, 2, 0): -1},
    {(2, 0, 1): 1, (0, 2, 1): -1},
    {(3, 0, 0): 1, (1, 2, 0): -3},
)


def _harmonic_tensors(order: int) -> tuple[NDArray, NDArray]:
    """Symmetric tensors ``T_j`` of the terms of an order, and their dual basis.

    ``P_j(r) = T_j[r, ..., r]``, and the dual basis gives the coefficients of a
    symmetric traceless tensor on the ``T_j``.
    """
    terms = _MONOMIALS[order**2 : (order + 1) ** 2]
    tensors = np.zeros((len(terms),) + (3,) * order)
    for index in np.ndindex(*tensors.shape[1:]):
        powers = tuple(index.count(axis) for axis in range(3))
        weight = prod(factorial(power) for power in powers) / factorial(order)
        for j, term in enumerate(terms):
            tensors[(j, *index)] = weight * term.get(powers, 0)
    flat = tensors.reshape(len(terms), -1)
    dual = np.linalg.solve(flat @ flat.T, flat).reshape(tensors.shape)
    return tensors, dual


_TENSORS = tuple(_harmonic_tensors(order) for order in range(4))


def _reexpansion(rotation: NDArray, origin: NDArray, n_terms: int) -> NDArray:
    """Matrix ``X`` such that ``P(rotation @ u + origin) = P(u) @ X`` for all ``u``.

    For a term of order ``l``, the part of degree ``k`` of
    ``T_j[R u + r0, ..., R u + r0]`` is ``comb(l, k) T_j[r0^(l - k), (R u)^k]``.
    It is computed at once in homogeneous coordinates, ``R u + r0 = A (u, 1)``
    with ``A = [R | r0]``, from the entries of ``T_j[A, ..., A]`` with
    ``l - k`` homogeneous indices. See :ref:`girf`.
    """
    affine = np.column_stack([rotation, origin])
    expansion = np.zeros((n_terms, n_terms))
    for order in range(isqrt(n_terms)):
        tensors = _TENSORS[order][0]
        for _ in range(order):
            tensors = np.einsum("ji...,ia->j...a", tensors, affine)
        for degree in range(order + 1):
            index = (slice(None), *(3,) * (order - degree), *(slice(3),) * degree)
            axes = "abc"[:degree]
            expansion[degree**2 : (degree + 1) ** 2, order**2 : (order + 1) ** 2] = (
                comb(order, degree)
                * np.einsum(f"i{axes},j{axes}->ij", _TENSORS[degree][1], tensors[index])
            )
    return expansion


def _predict_fields(
    gradients: NDArray,
    girf: NDArray,
    freqs: NDArray,
    dt: float,
    delay: float,
    t_start: float,
) -> NDArray:
    """Field terms played for ``gradients`` given on the GIRF axes."""
    n_samples = gradients.shape[1]
    # Sampling the impulse responses at dt directly on the frequency grid (no
    # interpolation when 1/(df dt) is an integer) and zero-padding them gives
    # their exact spectrum on the finer grid of the linear convolution.
    n_h = round(1 / ((freqs[1] - freqs[0]) * dt))
    n_pre = round(-t_start / dt)
    response = np.apply_along_axis(
        lambda v: np.interp(rfftfreq(n_h, dt), freqs, v, right=0), -1, girf
    )
    impulse = np.roll(irfft(response, n=n_h, axis=-1), n_pre, axis=-1)

    n_fft = next_fast_len(n_samples + n_h - 1, real=True)
    response = rfft(impulse, n=n_fft, axis=-1)
    response *= np.exp(-2j * np.pi * rfftfreq(n_fft, dt) * delay)
    spectrum = np.einsum("sfi,itf->sft", rfft(gradients, n=n_fft, axis=1), response)
    return irfft(spectrum, n=n_fft, axis=1)[:, n_pre : n_pre + n_samples]


def apply_girf(
    *,
    girf: NDArray,
    freqs: NDArray,
    gradients: NDArray | None = None,
    trajectory: NDArray | None = None,
    acq: Acquisition | None = None,
    girf_affine: NDArray | None = None,
    delay: float = 0.0,
    t_start: float = -1e-3,
) -> tuple[NDArray, NDArray]:
    """Predict the encoding actually played, using a GIRF.

    Exactly one of ``gradients`` or ``trajectory`` must be provided. The
    gradients or trajectory played are returned in the same form as the
    input, along with the deviations of all the field terms from the nominal
    encoding. These are expanded on spherical harmonics in the frame of the
    image: on its axes, in m, from its voxel ``acq.img_size // 2``. See
    :ref:`girf` for the model and its conventions.

    Parameters
    ----------
    girf : NDArray
        Gradient impulse response function, with shape ``(3, n_terms, n_freqs)``:
        ``girf[i, j]`` is the transfer function from the gradient axis ``i`` to
        the field term ``j``, given by :func:`spherical_harmonics`. ``n_terms``
        is 4, 9 or 16, for complete orders up to 1, 2 or 3.
    freqs : NDArray
        Frequencies of ``girf`` in Hz, uniformly spaced from 0. The response is
        taken as zero above ``freqs[-1]``.
    gradients : NDArray, optional
        Nominal gradients in T/m on the axes of the image, with shape
        ``(n_shots, n_samples, dim)``, ``dim`` being 2 or 3.
    trajectory : NDArray, optional
        Nominal normalized trajectory, with shape ``(n_shots, n_samples, dim)``.
    acq : Acquisition, optional
        Acquisition configuration, providing the sampling interval of the inputs
        (``acq.raster_time``), the gyromagnetic ratio, the normalization of
        ``trajectory``, and the position and orientation of the image
        (``acq.affine``). If `None`, the default acquisition is used.
    girf_affine : NDArray, optional
        Affine ``(4, 4)`` from the physical coordinates of ``acq.affine`` to the
        coordinates of the GIRF: the axes of the gradient coils, from the origin
        of the field measurement. If `None`, they are the same.
    delay : float, optional
        Extra delay applied to the responses, in s. The default is 0.
    t_start : float, optional
        Time in s at which the impulse responses start, in ``(-1/df, 0]`` with
        ``df`` the step of ``freqs``. They are taken as zero outside of
        ``[t_start, t_start + 1/df)``. The default is -1 ms.

    Returns
    -------
    gradients : NDArray
        If ``gradients`` is provided, the gradients played at the centre of the
        image in T/m, with the same shape and axes as the input.
    trajectory : NDArray
        If ``trajectory`` is provided, the normalized trajectory played, wrapped
        into ``[-acq.norm_factor, acq.norm_factor)``, with the same shape and axes
        as the input.
    fields : NDArray
        Deviations of the field terms from the nominal encoding, in T/m^l for a
        term of order l, at the gradient raster: shape
        ``(n_shots, n_samples, n_terms)`` for ``gradients``, and
        ``(n_shots, n_samples - 1, n_terms)``, between the samples, for
        ``trajectory``.

    Raises
    ------
    ValueError
        If none or both of ``gradients`` and ``trajectory`` are provided, if
        ``girf`` and ``freqs`` are inconsistent, if ``t_start`` is out of range,
        or if the affines are not rigid.

    Notes
    -----
    The data are compensated for the phase accrued by the zeroth order, and the
    higher orders are included in the reconstruction with
    :func:`mrinufft.extras.compute_girf_interpolators`.
    Gradients are taken as zero outside of the input: as the response to the
    pre-phasers leaks into the readout, they should be included in the input
    and cropped from the output.

    See Also
    --------
    mrinufft.io.read_girf
    mrinufft.extras.compute_girf_interpolators
    """
    if (gradients is None) == (trajectory is None):
        raise ValueError("Provide exactly one of `gradients` or `trajectory`.")
    acq = acq or Acquisition.default

    if trajectory is not None:
        gradients, initial_positions = convert_trajectory_to_gradients(trajectory, acq)
        gradients, fields = apply_girf(
            girf=girf,
            freqs=freqs,
            gradients=gradients,
            acq=acq,
            girf_affine=girf_affine,
            delay=delay,
            t_start=t_start,
        )
        trajectory = convert_gradients_to_trajectory(gradients, initial_positions, acq)
        bound = acq.norm_factor
        trajectory = (trajectory + bound) % (2 * bound) - bound
        return trajectory, fields

    freqs = np.asarray(freqs, dtype=float)
    girf = np.asarray(girf)
    if freqs[0] != 0:
        raise ValueError("`freqs` must start at 0 Hz.")
    if girf.ndim != 3 or girf.shape[0] != 3 or girf.shape[1] not in (4, 9, 16):
        raise ValueError(
            "`girf` must have a shape (3, n_terms in (4, 9, 16), n_freqs)."
        )
    if girf.shape[2] != len(freqs):
        raise ValueError("`girf` and `freqs` have different numbers of frequencies.")
    if not -1 / (freqs[1] - freqs[0]) < t_start <= 0:
        raise ValueError("`t_start` must be in (-1/df, 0].")

    gradients = np.asarray(gradients, dtype=float)
    dim = gradients.shape[-1]
    rotation, origin = _image_frame(acq, girf_affine, dim)
    nominal = np.zeros((*gradients.shape[:-1], 3))
    nominal[..., :dim] = gradients
    nominal = nominal @ rotation.T

    fields = _predict_fields(nominal, girf, freqs, acq.raster_time, delay, t_start)
    fields[..., 1:4] -= nominal
    fields = fields @ _reexpansion(rotation, origin, girf.shape[1]).T
    return gradients + fields[..., 1 : dim + 1], fields
