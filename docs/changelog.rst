Changelog
=========

Breaking changes and deprecations of the public API, following the
:ref:`deprecation policy <api-stability>`. All the changes of a release are
listed in its `GitHub release notes <https://github.com/mind-inria/mri-nufft/releases>`_.

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
