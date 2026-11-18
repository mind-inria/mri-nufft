"""Adapters exposing MRI-NUFFT operators to external libraries."""

import numpy as np

from mrinufft._array_compat import CUPY_AVAILABLE


def make_scipy_linop(nufft, *, cupy=False):
    """Create a SciPy or CuPy linear operator wrapping a NUFFT operator."""
    if cupy and not CUPY_AVAILABLE:
        raise ValueError("cupy is not available")
    elif cupy:
        from cupyx.scipy.sparse.linalg import LinearOperator
    else:
        from scipy.sparse.linalg import LinearOperator

    linop = LinearOperator(
        shape=(
            nufft.n_batchs * nufft.n_coils * nufft.n_samples,
            nufft.n_batchs
            * (1 if nufft.uses_sense else nufft.n_coils)
            * np.prod(nufft.shape),
        ),
        matvec=lambda x: nufft.op(x.reshape(nufft.img_full_shape)).ravel(),
        rmatvec=lambda x: nufft.adj_op(x.reshape(nufft.ksp_full_shape)).ravel(),
        dtype=nufft.cpx_dtype,
    )
    linop._nufft = nufft
    return linop


def __getattr__(name):
    """Load optional framework adapters only when requested."""
    if name == "DeepInvPhyNufft":
        from .deepinv import DeepInvPhyNufft

        return DeepInvPhyNufft
    if name == "MRProNUFFT":
        from .mrpro import MRProNUFFT

        return MRProNUFFT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
