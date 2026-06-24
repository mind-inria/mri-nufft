"""
Extra utilities for non Cartesian MRI.

This modules notably provides way to estimate sensitivity maps and off-resonance
correction.

Most of this module is accessible through the following registry :

- coil sensitivity maps: :py:func:`~mrinufft.extras.get_smaps`:

  .. autoregistry:: smaps

- off-resonance correction: :py:func:`~mrinufft.extras.get_orc_factorization`:

  .. autoregistry:: orc_factorization

- least-square optimization methods: :py:func:`~mrinufft.extras.get_optimizer`:

  .. autoregistry:: optimizer

.. tip::
    You can register your own methods to the registries using
    the following **decorators**:

    - :py:func:`~mrinufft.extras.register_smaps`,
    - :py:func:`~mrinufft.extras.register_orc`,
    - :py:func:`~mrinufft.extras.register_optim`.


This registry system is also available when using a non-Cartesian Fourier operator

For example:

.. code-block:: Python

    from mrinufft import get_operator
    from mrinufft.extras import register_smaps

    # get a non-Cartesian Fourier operator
    # with espirit sensitivity maps and off-resonance correction
    fourier_op = get_operator("nufft", trajectory=trajectory, smaps="espirit")
    fourier_op_orc = fourier_op.with_off_resonance_correction(
        interpolators={"name":"svd", "L":10}
    )
    # select the least-square solver to use for pseudo-inverse computation.
    img = fourier_op_orc.pinv_solver(kspace_data, solver="lsqr")

Custom registered functions can be used as well:

.. code-block:: Python

    @register_smaps("awesome")
    def awesome_smaps(kspace_data, trajectory, **kwargs):
        ...

    fourier_op = get_operator("nufft", trajectory=trajectory, smaps="awesome")
    # using the function is equivalent:
    fourier_op = get_operator("nufft", trajectory=trajectory, smaps=awesome_smaps)
"""

from .cartesian import fft, ifft
from .data import fse_simulation, get_brainweb_map
from .field_map import (
    get_complex_fieldmap_rad,
    get_orc_factorization,
    make_b0map,
    make_t2smap,
    register_orc,
    compute_mfi_coefficients,
    compute_mti_coefficients,
    compute_svd_coefficients,
)
from .optim import (
    get_optimizer,
    loss_l2_AHreg,
    loss_l2_reg,
    register_optim,
    lsmr,
    lsqr,
    cg,
)
from .smaps import (
    cartesian_espirit,
    coil_compression,
    espirit,
    get_smaps,
    low_frequency,
    register_smaps,
)

__all__ = [
    "cartesian_espirit",
    "cg",
    "coil_compression",
    "compute_mfi_coefficients",
    "compute_mti_coefficients",
    "compute_svd_coefficients",
    "espirit",
    "fft",
    "fse_simulation",
    "get_brainweb_map",
    "get_complex_fieldmap_rad",
    "get_optimizer",
    "get_orc_factorization",
    "get_smaps",
    "ifft",
    "loss_l2_AHreg",
    "loss_l2_reg",
    "low_frequency",
    "lsmr",
    "lsqr",
    "make_b0map",
    "make_t2smap",
    "register_optim",
    "register_orc",
    "register_smaps",
]
