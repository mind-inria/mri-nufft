Changelog
=========

Breaking changes and deprecations of the public API, following the
:ref:`deprecation policy <api-stability>`. All the changes of a release are
listed in its `GitHub release notes <https://github.com/mind-inria/mri-nufft/releases>`_.

Releases before 1.7.0 had no deprecation period: their changes broke the previous
API directly.

Unreleased
----------

Breaking changes
^^^^^^^^^^^^^^^^

- The default ``n_bins`` of :func:`~mrinufft.extras.compute_svd_coefficients`,
  :func:`~mrinufft.extras.compute_mti_coefficients` and
  :func:`~mrinufft.extras.compute_mfi_coefficients` changed from ``1024`` to
  ``-1``: the number of bins is now estimated from the field map and the readout
  time (:gh:`427`).
- The default ``power_iter`` of :func:`~mrinufft.extras.espirit` and
  :func:`~mrinufft.extras.cartesian_espirit` changed from ``30`` to ``10``, as the
  power iterations are now warm-started (:gh:`424`).

Deprecations
^^^^^^^^^^^^

- Importing ``DeepInvPhyNufft`` from ``mrinufft.operators.autodiff`` is
  deprecated and will fail from 1.9.0. Import it from :mod:`mrinufft.operators`
  instead (:gh:`417`).

1.6.0 (2026-10-29)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- The top-level ``mrinufft`` namespace no longer re-exports the content of its
  submodules (:gh:`386`, :gh:`391`). Import from the submodules instead:

  - :mod:`mrinufft.trajectories`: the trajectory initializations
    (``initialize_2D_radial``...), the trajectory tools (``stack``, ``rotate``,
    ``precess``, ``conify``, ``shellify``, ``stack_spherically``,
    ``duplicate_along_axes``, ``radialize_center``, ``oversample``), the sampling
    densities (``create_*_density``, ``sample_from_density``), ``Acquisition``
    and ``Hardware``;
  - :mod:`mrinufft.display`: ``display_2D_trajectory``, ``display_3D_trajectory``,
    ``display_gradients``, ``display_gradients_simply`` and ``displayConfig``;
  - :mod:`mrinufft.density`: ``voronoi``, ``cell_count``, ``pipe`` and
    ``get_density``;
  - :mod:`mrinufft.operators`: ``check_backend`` and ``list_backends``.

- The display functions moved from ``mrinufft.trajectories`` to
  :mod:`mrinufft.display`. ``mrinufft.trajectories.inits`` and
  ``mrinufft.trajectories.tools`` no longer re-export their content, and the
  ``mrinufft.trajectories.trajectory2D``, ``trajectory3D`` and ``display3D``
  compatibility modules are removed: import from :mod:`mrinufft.trajectories`
  (:gh:`386`).
- The Siemens hardware presets ``SIEMENS`` are renamed ``SiemensGradient``
  (:gh:`386`).
- Several helpers became private: ``voronoi_unique``, ``normalize_weights``,
  ``scaled_dcp``, ``acq2opts``, ``siemens_quat_to_rot_mat`` and ``C_lazy``
  (:gh:`386`).
- :func:`~mrinufft.extras.espirit` and :func:`~mrinufft.extras.cartesian_espirit`
  (:gh:`395`): ``kernel_width`` (default ``6``) is replaced by ``kernel_size``
  (default ``7``) and ``kernel_shape`` (default ``"ellipsoid"``); the default
  ``thresh`` changed from ``0.02`` to ``0.05``; ``crop`` now bounds the gap
  between the eigenvalue and its ideal value (default ``0.08``), instead of
  thresholding the eigenvalue (default ``0.95``).
- The ``dims`` argument of ``mrinufft.extras.cartesian.fft`` and ``ifft`` is now
  required (it was ``3``) (:gh:`395`).

1.5.1 (2026-05-28)
------------------

No breaking change.

1.5.0 (2026-02-28)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- ``readout_time`` is now the first, required, argument of
  ``with_off_resonance_correction`` (:gh:`337`).
- ``read_siemens_rawdat`` no longer returns the twix object by default
  (``return_twix=False``), and its new ``reshape`` argument comes before
  ``return_twix``, ``slice_num`` and ``contrast_num`` (:gh:`349`).

1.4.0 (2025-11-25)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- ``get_gradient_times_to_travel`` and
  ``get_gradient_amplitudes_to_travel_for_set_time`` are removed from
  ``mrinufft.trajectories.tools``, replaced by the ``mrinufft.trajectories.gradients``
  module (``connect_gradient``, ``get_prephasors_and_spoilers``) (:gh:`283`).
- ``write_trajectory``: the new ``grad_method`` argument comes before ``version``
  (:gh:`283`).

1.3.1 (2025-11-14)
------------------

No breaking change.

1.3.0 (2025-11-14)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- Python 3.10 or newer is required (:gh:`295`).
- The ``pynfft`` backend uses the ``pynfft3`` package (:gh:`274`).
- The hardware and acquisition parameters are grouped in an ``Acquisition``
  object (:gh:`285`): the ``norm_factor``, ``resolution``, ``raster_time``,
  ``gamma``, ``gmax`` and ``smax`` arguments of ``normalize_trajectory``,
  ``unnormalize_trajectory``, ``convert_trajectory_to_gradients``,
  ``convert_gradients_to_trajectory``, ``convert_gradients_to_slew_rates``,
  ``convert_slew_rates_to_gradients``, ``compute_gradients_and_slew_rates``,
  ``check_hardware_constraints`` and of the display functions are replaced by a
  single ``acq`` argument (``Acquisition.default`` when not given). The same goes
  for the ``shape`` and ``traj_params`` arguments of ``get_gridded_trajectory``.
- The values of ``Gammas`` are now in Hz/T instead of kHz/T, e.g.
  ``Gammas.HYDROGEN`` is ``42_576_000`` instead of ``42576`` (:gh:`285`).
- The off-resonance correction is reworked (:gh:`300`):
  ``MRIFourierCorrected(fourier_op, b0_map, readout_time, r2star_map, mask, interpolator="svd")``
  computes its interpolators, and ``with_off_resonance_correction`` takes the same
  arguments instead of ``B, C, indices``. The ``n_time_segments``, ``n_bins``,
  ``B``, ``tl`` and ``backend`` arguments of ``MRIFourierCorrected`` are removed,
  and ``get_interpolators_from_fieldmap`` is replaced by the
  ``mrinufft.extras.field_map`` functions (``compute_svd_coefficients``...).
- The ``cg`` method of the operators is removed: use ``pinv_solver`` (with
  ``optim="cg"``, ``"lsqr"`` or ``"lsmr"``) or ``mrinufft.extras.optim.cg``,
  which replaces ``mrinufft.extras.gradient.cg`` (:gh:`316`).
- ``get_smaps`` moved from ``mrinufft.extras.utils`` to ``mrinufft.extras.smaps``,
  and ``fse_simulation`` from ``mrinufft.extras.sim`` to ``mrinufft.extras.data``
  (:gh:`300`).
- The ``num_iterations`` argument of the ``pipe`` density compensation of the
  gpuNUFFT, torchkbnufft and tensorflow backends is renamed ``max_iter``
  (:gh:`316`).
- ``wrt_data`` and ``wrt_traj`` of ``make_autograd`` are keyword-only (:gh:`316`).
- ``initialize_3D_wave_caipi``: ``Nc`` is renamed ``Nc_or_R`` (:gh:`308`).
- ``display_gradients`` and ``display_gradients_simply`` no longer take
  ``show_signal`` (:gh:`285`).
- ``write_trajectory``: the new ``TE_pos``, ``pregrad`` and ``postgrad`` arguments
  come before ``gmax``, ``smax`` and ``version`` (:gh:`276`).

1.2.1 (2025-04-03)
------------------

No breaking change.

1.2.0 (2025-03-04)
------------------

No breaking change.

1.1.1 (2025-02-10)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- The ``image`` argument of ``data_consistency`` is renamed ``image_data``.

1.1.0 (2025-02-02)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- ``hard_bounded`` of ``stack`` and ``reverse_odd_shots`` of ``epify`` are
  keyword-only (:gh:`228`).

1.0.0 (2024-11-01)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- ``mrinufft.operators.off_resonnance`` is renamed
  ``mrinufft.operators.off_resonance``, and the ``with_off_resonnance_correction``
  method of the operators ``with_off_resonance_correction`` (:gh:`180`).
- ``MRIFourierCorrected`` computes its interpolators from ``b0_map`` and
  ``readout_time``: its ``C`` and ``indices`` arguments are removed (:gh:`180`).
- ``initialize_3D_TURBINE`` and ``initialize_3D_REPI`` are renamed
  ``initialize_3D_turbine`` and ``initialize_3D_repi`` (:gh:`186`).
- ``epify``: ``Nc_trains`` is replaced by ``nb_trains``, which comes after
  ``Ns_transitions`` (:gh:`186`).
- ``with_torch`` and ``with_tensorflow`` are no longer importable from
  ``mrinufft.operators.base`` (:gh:`194`).
