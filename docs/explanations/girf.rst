.. _girf:

=========================================
Gradient Impulse Response Function (GIRF)
=========================================

In MR System, the gradient system has imperfections, that can make the effective k-space trajectories different from the one that were prescribed.

The fields played by the gradient chain differ from the nominal gradients: the
amplifiers and coils have a limited bandwidth, eddy currents oppose fast
changes, and each gradient coil also produces B0 and higher-order spatial terms.
For non-Cartesian trajectories, reconstructing with the nominal k-space
positions then causes blurring, shading or ghosting.

These imperfection are typically modeled as Linear Time Invariant (LTI) system, and we 
model them using the gradient impulse
response function (GIRF) [Van13]_ [Add12]_.

.. note::

   In mri-nufft, the GIRF can be read and apply with the following functions:

   - :func:`mrinufft.trajectories.apply_girf` to correct for these imperfections.
   - :func:`mrinufft.io.read_girf` reads the GIRF (tailored to the Skope system in used at Neurospin)
   - :func:`mrinufft.extras.compute_girf_interpolators` includes the higher-order
       terms, and the static off-resonance, in :class:`~mrinufft.operators.off_resonance.MRIFourierCorrected`.

.. tip::

   See the example :ref:`sphx_glr_generated_autoexamples_operators_example_girf.py` for a
   complete reconstruction.


Model
=====

Field terms
-----------

During the acquisition, the longitudinal field departs from the main field
:math:`B_0` by

.. math::

   B(\boldsymbol{r}, t) = \Delta B_0(\boldsymbol{r})
   + \sum_j b_j(t) P_j(\boldsymbol{r}),

where :math:`\boldsymbol{r} = (x, y, z)` is the position on the axes of the GIRF
(the axes of the gradient coils), relative to its origin (usually the
isocentre).

- :math:`\Delta B_0(\boldsymbol{r})` is the static off-resonance, due to the
  magnetic susceptibility of the object and to the shim. It depends on the
  object, is measured with a field map, and is not described by the GIRF.
- :math:`b_j(t) P_j(\boldsymbol{r})` are the dynamic terms produced by the
  gradient chain, expanded on real spherical harmonics. They include the
  nominal gradients, and the GIRF predicts them from the gradients played. The
  B0 term :math:`j = 0` is the dynamic part only, e.g. from eddy currents.

The terms follow the order and the unnormalized polynomials of the Skope field
cameras (:func:`~mrinufft.trajectories.spherical_harmonics`), with
:math:`r^2 = x^2 + y^2 + z^2`:

+---------+-------+-------------------------------------------------------------------+----------------------+
| Terms   | Order | :math:`P_j(\boldsymbol{r})`                                       | Unit of ``girf``     |
+=========+=======+===================================================================+======================+
| 0       | 0     | :math:`1`                                                         | m                    |
+---------+-------+-------------------------------------------------------------------+----------------------+
| 1 - 3   | 1     | :math:`x,\ y,\ z`                                                 | 1                    |
+---------+-------+-------------------------------------------------------------------+----------------------+
| 4 - 8   | 2     | :math:`xy,\ zy,\ 3z^2 - r^2,\ xz,\ x^2 - y^2`                     | 1/m                  |
+---------+-------+-------------------------------------------------------------------+----------------------+
| 9 - 15  | 3     | :math:`3yx^2 - y^3,\ xyz,\ 5yz^2 - yr^2,\ 5z^3 - 3zr^2,`          | 1/m\ :sup:`2`        |
|         |       | :math:`5xz^2 - xr^2,\ zx^2 - zy^2,\ x^3 - 3xy^2`                  |                      |
+---------+-------+-------------------------------------------------------------------+----------------------+

These polynomials are harmonic, and those up to an order span all the harmonic polynomials up to that degree.
:func:`~mrinufft.trajectories.apply_girf` handles complete orders: 4, 9 or 16
terms.

Transfer functions
------------------

Each term responds linearly to the nominal gradients :math:`g_x, g_y, g_z` on
the axes of the GIRF,

.. math::

   b_j(t) = \sum_{i \in \{x, y, z\}} (h_{ij} * g_i)(t),
   \quad \text{with} \quad
   \mathrm{girf}[i, j](f) = \int h_{ij}(t)\, e^{-2\imath\pi f t}\, dt.

The self terms :math:`h_{xx}, h_{yy}, h_{zz}` are close to a low-pass filter
with a unit gain at 0 Hz, and the other terms are small. As the impulse
responses are real, ``girf`` is stored for positive frequencies only.


Frames and affines
==================

GIRF coefficients are expressed in the Scanner gradient axis. However, the reconstruction may have a different position and orientation.  The change of coordinates between the two frames is handled by the affine transform matrix. In MRI-NUFFT, this is handled in the :class:`~mrinufft.trajectories.Acquisition` class:
``acq.affine`` (:class:`~mrinufft.trajectories.Acquisition`) maps the voxel indices to physical coordinates as given by the raw data or a NIfTI header,  ``acq.fov / acq.img_size``, holds the size and resolution of the image.

For applying the GIRF this is taken into account with the argument ``girf_affine`` in :func:`mrinufft.trajectories.apply_girf`.

.. note::

    The change of coordinate is a rigid transform: :math:`\boldsymbol{r} = R\boldsymbol{u} + \boldsymbol{r}_0`, with :math:`\boldsymbol{r}_0` the centre of the image in the frame of the GIRF.

    The nominal gradients, given on the image axes, are rotated by :math:`R` to predict the fields, which are then expanded again in the frame of the image,

    .. math::

        \sum_j b_j(t) P_j(R\boldsymbol{u} + \boldsymbol{r}_0)
        = \sum_j b'_j(t) P_j(\boldsymbol{u}).


.. dropdown:: Computing :math:`b_j'` in closed form

    A term of order :math:`l` is a symmetric tensor :math:`T_j` of rank :math:`l`
    applied to the position,

    .. math::

        P_j(\boldsymbol{r}) = T_j[\boldsymbol{r}, \dots, \boldsymbol{r}]
        = \sum_{i_1, \dots, i_l} T_{j, i_1 \dots i_l}\, r_{i_1} \cdots r_{i_l},

    e.g. :math:`xz = \boldsymbol{r}^\top T \boldsymbol{r}` with
    :math:`T_{xz} = T_{zx} = 1/2` and zeros elsewhere. The polynomial is harmonic
    if and only if the tensor is traceless. As :math:`T_j` is symmetric, the
    binomial theorem gives

    .. math::

        P_j(R\boldsymbol{u} + \boldsymbol{r}_0) = \sum_{k=0}^{l} \binom{l}{k}\,
        T_j[\underbrace{\boldsymbol{r}_0, \dots, \boldsymbol{r}_0}_{l - k},
        R\boldsymbol{u}, \dots, R\boldsymbol{u}].

    The part of degree :math:`k` in :math:`\boldsymbol{u}` is the tensor :math:`T_j` contracted :math:`l - k` times with :math:`\boldsymbol{r}_0`, with its remaining indices contracted with :math:`R`.
    Both operations keep the tensors symmetric and traceless, so that each degree remains harmonic, and the
    :math:`b'_j` are the coefficients of these tensors on the :math:`T_j` of their order.
    Up to the second order, with :math:`\boldsymbol{g} = (b_x, b_y, b_z)` and

    .. math::

        Q = \begin{pmatrix}
        b_{x^2 - y^2} - b_{3z^2 - r^2} & b_{xy}/2 & b_{xz}/2 \\
        b_{xy}/2 & -b_{x^2 - y^2} - b_{3z^2 - r^2} & b_{zy}/2 \\
        b_{xz}/2 & b_{zy}/2 & 2 b_{3z^2 - r^2}
        \end{pmatrix},

    and :math:`\boldsymbol{g}', Q'` built in the same way from the :math:`b'_j`,

    .. math::

        b'_0 = b_0 + \boldsymbol{g} \cdot \boldsymbol{r}_0
        + \boldsymbol{r}_0^\top Q\, \boldsymbol{r}_0,
        \qquad
        \boldsymbol{g}' = R^\top (\boldsymbol{g} + 2 Q\, \boldsymbol{r}_0),
        \qquad
        Q' = R^\top Q R.

    The rotation keeps the orders, and the translation moves each order to the lower ones, with a factor :math:`r_0^{l - k}`. 
    For instance, the :math:`xz` term of a slice at :math:`z_0` adds :math:`z_0 x` to the gradient along :math:`x`, and :math:`3z^2 - r^2` a phase in :math:`z_0^2`. 
    This holds for reflections too.

    In homogeneous coordinates, :math:`R\boldsymbol{u} + \boldsymbol{r}_0 =
    A\tilde{\boldsymbol{u}}`, with :math:`A = [R \mid \boldsymbol{r}_0]` the first three rows of the affine and :math:`\tilde{\boldsymbol{u}} =
    (\boldsymbol{u}, 1)`.
    All the degrees then follow from :math:`\tilde{T}_j = T_j[A, \dots, A]`, which is symmetric: the part of degree :math:`k` is :math:`\binom{l}{k}` times its entries whose first :math:`l - k`
    indices are the homogeneous one.

.. dropdown:: Computing :math:`b_j'` with the linear regression

    The relation holds for all :math:`\boldsymbol{u}`. Evaluating the terms at
    :math:`N \geq n_\mathrm{terms}` random points :math:`\boldsymbol{u}_n` gives
    the linear system

    .. math::

        P(R\boldsymbol{u}_n + \boldsymbol{r}_0) = P(\boldsymbol{u}_n)\, X,
        \qquad \boldsymbol{b}' = X \boldsymbol{b},

    with :math:`P(\boldsymbol{u}_n)` the row of the terms at
    :math:`\boldsymbol{u}_n`. Its least-squares solution is exact up to the
    rounding errors, as the points are generic. It needs no tensor, and the tests
    use it to check the closed form.


In MRI-NUFFT we use the closed form expression, in homogeneous coordinates
(``_reexpansion`` in ``mrinufft/trajectories/girf.py``, used by
:func:`~mrinufft.trajectories.apply_girf`).

    
Deviations from the nominal encoding
------------------------------------

:func:`~mrinufft.trajectories.apply_girf` returns the trajectory played, and the
fields of the deviations of the terms from the nominal encoding,
:math:`b'_j - b'_{j,\mathrm{nominal}}`, in T/m\ :sup:`l` for a term of order
:math:`l`, at the gradient raster: the nominal gradients are removed from the
first-order terms before expanding them in the frame of the image. The
reconstruction uses the phases they accrue since the start of the waveforms,

.. math::

   \phi_j(t) = 2\pi\gamma \int_0^t (b'_j - b'_{j,\mathrm{nominal}})(\tau)\, d\tau,

in rad/m\ :sup:`l`. At the samples of the trajectory, between which the fields
are given,

.. code-block:: python

    phase = 2 * np.pi * acq.gamma * acq.raster_time * np.cumsum(fields, axis=1)
    phase = np.pad(phase, ((0, 0), (1, 0), (0, 0)))

Off-centre, the zeroth order
thus includes :math:`2\pi(\boldsymbol{k} - \boldsymbol{k}_\mathrm{nominal})
\cdot \boldsymbol{r}_0`. Being relative to the nominal encoding, these
corrections apply on top of however the shift of the field of view is handled
for the nominal trajectory, by the sequence or by the reconstruction.


Reconstruction
==============

Signal model
------------

With the NUFFT convention of MRI-NUFFT (:ref:`NUFFT`), where a field offset
:math:`b` adds :math:`e^{-2\imath\pi\gamma \int b}`, the data are

.. math::

   y(t) = e^{-\imath \phi_0(t)} \int x(\boldsymbol{u})\,
   e^{-2\imath\pi \boldsymbol{k}(t) \cdot \boldsymbol{u}}\,
   e^{-\imath \psi(\boldsymbol{u}, t)}\, d\boldsymbol{u},
   \quad \text{with} \quad
   \psi(\boldsymbol{u}, t) = \sum_{j \geq 4} \phi_j(t) P_j(\boldsymbol{u})
   + 2\pi\gamma \Delta B_0(\boldsymbol{u})\, t,

:math:`\boldsymbol{k}` being the trajectory played, and :math:`t` the time
from the excitation, or from the reference echo time of the field map.

Order by order
--------------

Zeroth order: a phase of the data
    :math:`\phi_0` is common to all positions: multiply the data, of all the
    coils, by ``np.exp(1j * phase[..., 0])``. This is exact, and needs no change
    of the operator. As the phase accrued during the pre-phasers differs between
    shots, it must be integrated from the start of the waveforms, before
    cropping them.

First order: the trajectory
    Use the trajectory played in the NUFFT [Van16]_, and recompute what derives from it:
    density compensation, sensitivity maps (from the compensated data), and the
    off-resonance interpolators. The trajectory is wrapped into
    ``[-acq.norm_factor, acq.norm_factor)``, which is exact for a discrete image,
    as its spectrum is periodic. Otherwise the gain above 1 of the gradients
    would push samples past the normalization, and the operators would then
    misread their units.

Higher orders and static off-resonance: an extended model
    :math:`e^{-\imath\psi}` depends on both time and position: the model is not
    a Fourier transform anymore [Wil11]_. As for the off-resonance alone
    (:ref:`nufft-orc`), it is approximated by
    :math:`\sum_\ell B_\ell(t) C_\ell(\boldsymbol{u})`, given to
    :class:`~mrinufft.operators.off_resonance.MRIFourierCorrected` with
    ``interpolator=(B, C)``: each application then costs :math:`L` NUFFTs. Both
    terms are approximated together by
    :func:`~mrinufft.extras.compute_girf_interpolators`, which integrates the
    fields as above, from the factorization of the static off-resonance given by
    :func:`~mrinufft.extras.get_orc_factorization`.

.. note::

   :class:`~mrinufft.operators.off_resonance.MRIFourierCorrected` models the
   off-resonance as :math:`e^{2\imath\pi \Delta f(\boldsymbol{u}) t}`,
   :math:`\Delta f` being its ``b0_map``: with the sign convention of the
   trajectory, a field offset :math:`\Delta B_0` corresponds to
   :math:`\Delta f = -\gamma\Delta B_0`.

.. _girf-factorization:

Factorization of the higher orders
----------------------------------

The higher-order terms are polynomials of the position, so that
:math:`e^{-\imath\psi}` has a Taylor series in closed form:

.. math::

   e^{-\imath \sum_j \phi_j(t) P_j(\boldsymbol{u})}
   = \sum_{\boldsymbol{\alpha}} \frac{(-\imath)^{|\boldsymbol{\alpha}|}}
   {\boldsymbol{\alpha}!}\, \boldsymbol{\phi}(t)^{\boldsymbol{\alpha}}\,
   \boldsymbol{P}(\boldsymbol{u})^{\boldsymbol{\alpha}},

a factorization whose time factors are products of the phase coefficients, and
whose space factors are products of spherical harmonics. Truncated to an order
:math:`n`, its error is at most :math:`|\psi|^{n+1} / (n + 1)!`, which is small
as the expansion in the frame of the image leaves only the phase that is
neither constant nor linear. Its rank, the number of products, is then reduced
to :math:`L` by an SVD of the small matrices relating the two sets of factors,
which gives the best rank :math:`L` approximation of the series, without ever
forming the :math:`M \times N` matrix of the exponential. A factorization of the
static off-resonance is included by taking the products of its factors with
those of the compressed series, and compressing these products in the same way.

:func:`~mrinufft.extras.compute_girf_interpolators` chooses the order from the
range of :math:`\psi`, and the rank from the target error ``tol``. For a
:math:`\psi` of a few tenths of a radian, a rank of 6 to 8 reaches
:math:`10^{-3}`, as the optimal SVD of the full matrix does.


Implementation
==============

Sampling
--------

Gradient sample :math:`n` is held during :math:`[n\Delta t, (n + 1)\Delta t)`,
with :math:`\Delta t` = ``acq.raster_time``. Gradients are taken as zero before
the first sample and after the last one, and the fields are given at the same
samples as the gradients. To predict the fields at the ADC samples, use the ADC
dwell time as ``raster_time`` and repeat each gradient sample accordingly.

Time window and causality
-------------------------

A GIRF sampled with a frequency step :math:`\delta f` only determines the
impulse responses modulo a period of :math:`1/\delta f`. They are therefore
taken as zero outside of :math:`[t_\mathrm{start}, t_\mathrm{start} + 1/\delta f)`,
with :math:`t_\mathrm{start}` = ``t_start``.

The gradient chain is causal, but the measured responses start slightly before
:math:`t = 0`: the band limit of the measurement rings on both sides of
:math:`t = 0`, and the time reference of the measurement may be offset by a few
microseconds. By default, :math:`t_\mathrm{start} = -1` ms, as in the MATLAB
GIRF toolbox [GIRFtb]_.

- With :math:`t_\mathrm{start} = 0`, the ringing before :math:`t = 0` is moved to
  the end of the period.
- With a centred window, :math:`t_\mathrm{start} = -1/(2\delta f)`, half of any
  response slower than the window precedes its cause.

Responses slower than :math:`1/\delta f` cannot be represented. In particular, a
gain at 0 Hz that departs from those of the neighbouring frequencies, e.g.
because of how drifts were removed during the measurement, is spread as a
constant over the whole window: it adds a ramp lasting :math:`1/\delta f` after
each change of the gradients. The low frequencies of a GIRF are worth checking
before use.

The impulse responses are sampled exactly from the frequencies of the GIRF,
without interpolation when :math:`1/(\delta f \Delta t)` is an integer, then
zero-padded for the linear convolution. Interpolating the GIRF linearly onto a
finer frequency grid instead would attenuate the slow parts of the responses.

``delay`` adds a delay, possibly fractional, to all the responses.


Practical aspects
=================

Pre-phasers
    The responses to the pre-phasers, and to anything played before the ADC,
    leak into the readout. They should be included in the input, and cropped
    from the output. :func:`mrinufft.io.read_gradients` returns the full
    waveforms, with the number of samples before the ADC.

Noise
    Above the bandwidth of the measurement, the GIRF is only noise, which makes
    the impulse responses long. ``fmax`` in :func:`mrinufft.io.read_girf` sets
    it to zero; it should stay above the bandwidth of the gradient waveforms.
    The higher-order terms are measured with less sensitivity, and usually need
    a lower cut, e.g. ``girf[:, 4:, freqs > 10e3] = 0``.


Example
=======

.. code-block:: python

    import numpy as np

    from mrinufft.io import read_girf, read_gradients
    from mrinufft.trajectories import (
        Acquisition,
        Hardware,
        apply_girf,
        convert_gradients_to_trajectory,
    )

    freqs, girf = read_girf("girf.mat", fmax=30e3)
    gradients, initial_positions, params = read_gradients("trajectory.bin")
    acq = Acquisition(
        fov=tuple(params["FOV"]),
        img_size=tuple(params["img_size"]),
        gamma=params["gamma"] * 1e3,  # kHz/T to Hz/T
        hardware=Hardware(grad_raster_time=10e-6),
        affine=affine,  # voxel indices to physical coordinates, in m
    )

    nominal = convert_gradients_to_trajectory(gradients, initial_positions, acq)
    trajectory, fields = apply_girf(
        girf=girf, freqs=freqs, trajectory=nominal, acq=acq, girf_affine=girf_affine
    )
    # Phase of the zeroth order, accrued since the start of the waveforms.
    phase_0 = 2 * np.pi * acq.gamma * acq.raster_time * np.cumsum(fields[..., 0], axis=1)
    phase_0 = np.pad(phase_0, ((0, 0), (1, 0)))
    # Drop the pre-phasers, before the ADC.
    skip = params["start_skip_samples"]
    trajectory, phase_0 = trajectory[:, skip:], phase_0[:, skip:]

    # Compensate the zeroth order, with kspace_data of shape (n_coils, n_samples)
    # sampled at the same times as the trajectory.
    kspace_data = kspace_data * np.exp(1j * phase_0.ravel())

See :ref:`sphx_glr_generated_autoexamples_operators_example_girf.py` for the
higher orders and the static off-resonance.


References
==========

.. [Van13] Vannesjo, S. J., Haeberlin, M., Kasper, L., Pavan, M., Wilm, B. J.,
   Barmet, C., Pruessmann, K. P. "Gradient system characterization by impulse
   response measurements with a dynamic field camera." Magnetic Resonance in
   Medicine 69.2 (2013): 583-593. https://doi.org/10.1002/mrm.24263

.. [Add12] Addy, N. O., Wu, H. H., Nishimura, D. G. "Simple method for MR
   gradient system characterization and k-space trajectory estimation."
   Magnetic Resonance in Medicine 68.1 (2012): 120-129.
   https://doi.org/10.1002/mrm.23217

.. [Van16] Vannesjo, S. J., Graedel, N. N., Kasper, L., Gross, S., Busch, J.,
   Haeberlin, M., Barmet, C., Pruessmann, K. P. "Image reconstruction using a
   gradient impulse response model for trajectory prediction." Magnetic
   Resonance in Medicine 76.1 (2016): 45-58. https://doi.org/10.1002/mrm.25841

.. [Wil11] Wilm, B. J., Barmet, C., Pavan, M., Pruessmann, K. P. "Higher order
   reconstruction for MRI in the presence of spatiotemporal field
   perturbations." Magnetic Resonance in Medicine 65.6 (2011): 1690-1701.
   https://doi.org/10.1002/mrm.22767

.. [GIRFtb] MATLAB GIRF toolbox, ``GirfApplier.PredictGrad``.
   https://github.com/MRI-gradient/GIRF
