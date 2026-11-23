Changelog
=========

Notable changes of each release. Breaking changes and deprecations follow the
:ref:`deprecation policy <api-stability>`. Changes limited to the CI, the tests or
the code style are left out: the `GitHub release notes
<https://github.com/mind-inria/mri-nufft/releases>`_ list every pull request.

Releases before 1.7.0 had no deprecation period: their breaking changes applied
directly.

Unreleased
----------

Breaking changes
^^^^^^^^^^^^^^^^

- The default ``n_bins`` of :func:`~mrinufft.extras.compute_svd_coefficients`,
  :func:`~mrinufft.extras.compute_mti_coefficients` and
  :func:`~mrinufft.extras.compute_mfi_coefficients` changed from ``1024`` to
  ``-1``: the number of bins is now estimated from the field map and the readout
  time, and empty bins are dropped (:gh:`427`).
- The default ``power_iter`` of :func:`~mrinufft.extras.espirit` and
  :func:`~mrinufft.extras.cartesian_espirit` changed from ``30`` to ``10``, as the
  power iterations are now warm-started (:gh:`424`).
- The `GroupL2SoftThresholding` and `GradientLinearProjection` pyproximal operators are made private and
   refactored in :mod:`mrinufft.trajectories._proximal` (:gh:`434`).
- MRI-nufft requires Python 3.12 or later (:gh:`435`).
- ``MRIStackedNUFFTGPU`` takes an ``MRICufiNUFFT`` instance as ``backend``, or
  ``None`` to create one. A backend name raises a ``ValueError`` (:gh:`435`).

Deprecations
^^^^^^^^^^^^

- Importing ``DeepInvPhyNufft`` from ``mrinufft.operators.autodiff`` is
  deprecated and will fail from 1.9.0. Import it from :mod:`mrinufft.operators`
  instead (:gh:`417`).

New features
^^^^^^^^^^^^

- Gradient impulse response function (GIRF) correction: ``read_girf``,
  ``read_gradients``, ``compute_girf_interpolators`` and ``apply_girf``, and
  ``read_trajectory(read_pre_post_grad=True)`` to read the gradients outside of
  the ADC window (:gh:`414`).
- ``mlx-nufft`` backend for Apple Silicon, installed with the ``mlx-nufft`` extra
  (:gh:`413`).
- MRPro interface: ``make_mrpro`` exposes an operator as an MRPro linear operator
  (:gh:`417`).
- Analytical density compensation for radial trajectories,
  ``mrinufft.density.radial`` (:gh:`408`).
- ``write_trajectory(pad_to_multiple_of_4=True)`` pads the gradient waveforms to a
  multiple of four samples (:gh:`423`).
- The API stability and deprecation policy, and this changelog.

Improvements
^^^^^^^^^^^^

- The trajectory projection supports pyproximal 0.13 (:gh:`410`).
- Importing :mod:`mrinufft.trajectories`, :mod:`mrinufft.io` or
  :mod:`mrinufft.display` no longer imports pyproximal, pylops and torch, or
  pypulseq and matplotlib: the functions that need them import them. Importing
  :mod:`mrinufft.trajectories` takes about a third of the time it did.

Fixes
^^^^^

- ESPIRiT gives accurate sensitivity maps again after the PISCO refactor of 1.6.0,
  and PISCO is warm-started and keeps single precision (:gh:`424`).
- With a complex (B0 + R2*) field map, the off-resonance histogram keeps the B0
  resolution (:gh:`419`).
- The Twix affine origin is correct with slice oversampling (:gh:`429`).
- cufinufft waits for pending host-to-device copies before overwriting pinned
  sensitivity maps (:gh:`416`).
- The phase accrual of prephasors, and direct off-resonance interpolators, are
  handled correctly (:gh:`414`).
- ``mrinufft.__version__`` returns the installed version; it was not set.
- ``from mrinufft.trajectories import *`` and ``from mrinufft.io import *`` work:
  their ``__all__`` listed names the modules did not define. ``KMAX`` and
  ``compute_coprime_factors`` can now be imported from
  :mod:`mrinufft.trajectories`, and ``twix2nifti_affine`` is exported by
  :mod:`mrinufft.io`.

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
- ``espirit`` and ``cartesian_espirit`` (:gh:`395`): ``kernel_width`` (default
  ``6``) is replaced by ``kernel_size`` (default ``7``) and ``kernel_shape``
  (default ``"ellipsoid"``); the default ``thresh`` changed from ``0.02`` to
  ``0.05``; ``crop`` now bounds the gap between the eigenvalue and its ideal value
  (default ``0.08``), instead of thresholding the eigenvalue (default ``0.95``).
- The ``dims`` argument of ``mrinufft.extras.cartesian.fft`` and ``ifft`` is now
  required (it was ``3``) (:gh:`395`).
- The R2* map is subtracted from the complex field map, so that a positive R2*
  gives a decay; it used to be added (:gh:`404`).
- The slice position shifts read from Siemens data follow the X, Y, Z
  (sagittal, coronal, transverse) order (:gh:`404`).
- Siemens data are read with ``turbotwix`` instead of ``pymapvbvd`` (``io``
  extra) (:gh:`400`).

New features
^^^^^^^^^^^^

- Gram operator ``gram_op``, with a Toeplitz embedding on CPU and for the
  cufinufft, gpuNUFFT and torchkbnufft backends (``compute_toeplitz_kernel``,
  ``apply_toeplitz_kernel``) (:gh:`348`, :gh:`393`).
- PISCO estimation of sensitivity maps, ``pisco`` and ``cartesian_pisco``
  (:gh:`395`).
- Off-resonance correction for stacked operators (:gh:`404`).
- Logging throughout the package, configured with ``mrinufft.set_log_level``
  (:gh:`390`).
- Complex-to-real helpers for the autodiff operators: ``kspace_as_real``,
  ``kspace_as_cpx``, ``image_as_real`` and ``image_as_cpx`` (:gh:`370`).
- I/O: ``read_nsp_pair`` for NeuroSpin data, ``discard_frequency_outliers``, the
  noise scan in the Siemens header, and the typed ``TwixHeaderDict``
  (:gh:`400`, :gh:`404`).

Improvements
^^^^^^^^^^^^

- The submodules and the heavy optional dependencies are imported lazily
  (:gh:`388`).
- cufinufft uses CUDA streams for host-device transfers, also for stacked
  operators (:gh:`392`, :gh:`404`).
- Fewer array copies and a smaller memory footprint throughout (:gh:`393`).
- finufft uses ``execute_adjoint`` with finufft >= 2.5.0.
- A warning is emitted when the Lipschitz constant does not converge (:gh:`404`).
- The trajectory and k-space data dtypes are matched when reading data
  (:gh:`404`).

Fixes
^^^^^

- The 3D golden means radial trajectories no longer repeat every 10000 shots
  (:gh:`402`).
- The coil compression matrix had the wrong orientation (:gh:`406`).
- The density compensation is used correctly by LSQR and LSMR, and updated with
  the off-resonance operators (:gh:`404`).
- Stacked operators round the z positions instead of truncating them (:gh:`404`).
- Siemens reader: extraction of the slices and of the ACS lines, and batched
  shifts.
- The ``pipe`` density compensation is real valued with finufft, and uses the
  backend of the operator by default.
- The power method works with operators other than NUFFT operators.

Documentation
^^^^^^^^^^^^^

- Flat API documentation (:gh:`386`, :gh:`391`), more explanations on the NUFFT
  (:gh:`389`) and on the Toeplitz embedding (:gh:`348`).
- TV reconstruction with DeepInverse in the model-based reconstruction example
  (:gh:`370`).

1.5.1 (2026-05-28)
------------------

Fixes
^^^^^

- Recursion error with DeepInverse.
- The initial guess of ``cg`` is checked properly.
- Minimal ``torch`` version for the ``autodiff`` extra.

1.5.0 (2026-02-28)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- ``readout_time`` is now the first, required, argument of
  ``with_off_resonance_correction`` (:gh:`337`).
- ``read_siemens_rawdat`` no longer returns the twix object by default
  (``return_twix=False``), and its new ``reshape`` argument comes before
  ``return_twix``, ``slice_num`` and ``contrast_num`` (:gh:`349`).
- ``cg`` no longer uses the density compensation, which is mathematically wrong
  for this solver (:gh:`360`).

New features
^^^^^^^^^^^^

- DUCC0 backend, ``ducc0`` (:gh:`345`).
- Cartesian operator backend, ``cartesian`` (:gh:`355`).
- Variable density spiral, ``initialize_2D_vds_spiral``, compliant with the
  hardware constraints (:gh:`336`, :gh:`359`).
- Projection of trajectories on the hardware constraints,
  ``project_trajectory`` (:gh:`356`).
- Autograd for the off-resonance corrected operators (:gh:`337`).
- Trajectories can be initialized and updated as torch tensors, and the main
  backends implement ``update_samples`` (:gh:`338`).

Improvements
^^^^^^^^^^^^

- Trajectories outside of [-0.5, 0.5) are clipped, with a warning (:gh:`343`).
- The trajectory module is split into ``inits``, ``tools``, ``maths`` and
  ``display`` submodules, with the former modules kept for compatibility
  (:gh:`361`).
- Packaging for CUDA 13, and support for cupy 14.

Fixes
^^^^^

- I/O: shifts are read correctly and Cartesian data are reshaped properly
  (:gh:`349`), and version checks of old trajectory files avoid float rounding
  issues (:gh:`353`).
- Memory is released after estimating the sensitivity maps (:gh:`350`).
- Rounding of the decimated shape in ESPIRiT (:gh:`363`).

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

New features
^^^^^^^^^^^^

- Pulseq support: ``read_pulseq_traj`` extracts the trajectory of a Pulseq
  sequence file, and ``pulseq_gre`` creates a 3D GRE sequence for arbitrary
  trajectories (:gh:`283`).
- Gradient waveform optimization to connect shots and compute prephasors and
  spoilers, ``connect_gradient`` and ``get_prephasors_and_spoilers`` (:gh:`283`).
- DeepInverse support: ``make_deepinv_phy`` converts an operator to a DeepInverse
  physics, ``DeepInvPhyNufft`` (:gh:`328`).
- An existing progress bar can be passed to the solvers, e.g. ``pinv_solver``
  (:gh:`329`).

1.3.1 (2025-11-14)
------------------

Fixes
^^^^^

- README rendering on PyPI.

1.3.0 (2025-11-14)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- Python 3.10 or newer is required, with new minimal versions of the dependencies
  (:gh:`295`).
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

New features
^^^^^^^^^^^^

- ``Acquisition`` and ``Hardware`` configurations, with presets for Siemens
  scanners, to handle the physical units of trajectories (:gh:`285`).
- Pseudo-inverse solvers: ``pinv_solver`` with LSQR, LSMR and CG, Tikhonov
  regularization, and conversion of operators to SciPy ``LinearOperator``
  (:gh:`316`).
- Off-resonance correction with SVD, MTI and MFI interpolators
  (``compute_svd_coefficients``, ``compute_mti_coefficients``,
  ``compute_mfi_coefficients``), with lazily evaluated spatial coefficients
  (:gh:`300`).
- ESPIRiT sensitivity maps, ``espirit`` and ``cartesian_espirit`` (:gh:`324`), and
  coil compression, ``coil_compression`` (:gh:`293`).
- ECCENTRIC trajectories, ``initialize_2D_eccentric`` and
  ``initialize_3D_eccentric`` (:gh:`238`).
- Arc length parameterization of trajectories, ``parameterize_by_arc_length``
  (:gh:`236`).
- Wave-CAIPI defined from the acquisition and the GRAPPA / CAIPI factors
  (:gh:`308`).
- Prephasors and rewinders of trajectories (:gh:`276`).
- Density compensation on CPU with finufft's spread and interpolate (:gh:`224`).
- GPU selection with ``gpu_device_id`` for the cufinufft and gpuNUFFT backends
  (:gh:`290`).
- Batched sensitivity maps for the autograd operators (:gh:`263`).
- I/O: skipping samples when reading trajectories (:gh:`281`), the NIfTI affine
  matrix of Siemens data, ``nifti_affine`` (:gh:`266`), and the extraction of
  the sequence parameters (TE, TR...) from Siemens data (:gh:`315`).

Improvements
^^^^^^^^^^^^

- The B0 map is upsampled when its shape does not match the operator
  (:gh:`259`).
- CG with cufinufft no longer creates a second operator to compute the Lipschitz
  constant (:gh:`294`).
- More robust sensitivity maps estimation (:gh:`314`, :gh:`317`), with support of
  array masks.

Fixes
^^^^^

- Gridding of trajectories for display (:gh:`279`).
- GPU memory when updating the samples of cufinufft operators (:gh:`278`).
- Shape of the operators and of the initial guess of CG (:gh:`284`, :gh:`287`).
- Default acquisition when displaying gradients.

Documentation
^^^^^^^^^^^^^

- New documentation theme, improved README and docs (:gh:`275`, :gh:`321`,
  :gh:`325`), and an explanation of the NUFFT with off-resonance correction.

1.2.1 (2025-04-03)
------------------

Improvements
^^^^^^^^^^^^

- Update of the conjugate gradient and of its example (:gh:`252`).
- Clearer signal display of the gradient diagrams (:gh:`246`).

Fixes
^^^^^

- The coordinates fitted by the Lloyd sampling were shifted by half a voxel
  (:gh:`247`).

1.2.0 (2025-03-04)
------------------

New features
^^^^^^^^^^^^

- Display of 3D trajectories on a grid, ``get_gridded_trajectory`` (:gh:`193`).
- Siemens reader: reading of the ACS region, and ``remove_extra_kspace_samples``
  (:gh:`233`).

1.1.1 (2025-02-10)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- The ``image`` argument of ``data_consistency`` is renamed ``image_data``.

Fixes
^^^^^

- The computed density is set through the ``density`` setter.
- Missing default argument for stacked operators.
- ``is_cuda_tensor`` failed without torch.

1.1.0 (2025-02-02)
------------------

Breaking changes
^^^^^^^^^^^^^^^^

- Python 3.9 or newer is required, and numpy 2 is supported.
- ``hard_bounded`` of ``stack`` and ``reverse_odd_shots`` of ``epify`` are
  keyword-only (:gh:`228`).

New features
^^^^^^^^^^^^

- Sampling densities (``create_cutoff_decay_density``,
  ``create_polynomial_density``, ``create_energy_density``,
  ``create_chauffert_density``, ``create_fast_chauffert_density``) and sampling
  from them, ``sample_from_density`` (:gh:`206`).
- Random walk and travelling salesman trajectories in 2D and 3D (:gh:`206`).
- Random stacking of 2D trajectories, ``stack_random`` (:gh:`220`).
- Orientation of Siemens raw data, ``siemens_quat_to_rot_mat`` (:gh:`230`).
- ``pipe`` density compensation with cufinufft (:gh:`214`).

Fixes
^^^^^

- Normalization of the density compensation (:gh:`214`).
- Array backend checks of the off-resonance operator (:gh:`212`).
- The ``pynfft`` backend with Python 3.12.

Documentation
^^^^^^^^^^^^^

- U-Net training example (:gh:`199`, :gh:`209`), updated examples (:gh:`219`),
  type hints (:gh:`228`) and a contributing guide.

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

New features
^^^^^^^^^^^^

- Subspace projection operator, ``MRISubspace`` (:gh:`201`).
- Off-resonance correction from B0 and complex (R2* + B0) field maps,
  ``get_interpolators_from_fieldmap``, and field map simulations, ``make_b0map``
  and ``make_t2smap`` (:gh:`180`).
- Conjugate gradient solver, ``cg`` (:gh:`185`).
- Version 5 of the trajectory file format (:gh:`188`).
- ``skip_factor`` for TURBINE trajectories (:gh:`186`).

Fixes
^^^^^

- The density update uses the rescaled trajectory (:gh:`204`), and works with
  gpuNUFFT (:gh:`190`).
- Sensitivity maps can be cupy arrays (:gh:`198`).

Documentation
^^^^^^^^^^^^^

- TURBINE, REPI, off-resonance and CG examples (:gh:`180`, :gh:`186`), and the
  JOSS paper (:gh:`200`).
