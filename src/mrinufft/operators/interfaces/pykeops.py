"""Non Uniform Fourier transform with PyKeOps and torch.

TODO: Use Keops for the non fourier stuff (e.g. smaps and multi coil )
"""

import numpy as np
from .base import FourierOperatorCPU, proper_trajectory

PYKEOPS_AVAILABLE = True
try:
    from pykeops.numpy import ComplexLazyTensor as NumpyComplexLazyTensor
    from pykeops.numpy import LazyTensor as NumpyLazyTensor
    from pykeops.numpy import Genred as NumpyGenred
except ImportError:
    PYKEOPS_AVAILABLE = False

PYTORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    PYTORCH_AVAILABLE = False

if PYTORCH_AVAILABLE:
    from pykeops.torch import ComplexLazyTensor as TorchComplexLazyTensor
    from pykeops.torch import LazyTensor as TorchLazyTensor
    from pykeops.torch import Genred as TorchGenred


class KeopsNDFT:
    """Non Uniform Fourier transform with PyKeOps and torch.

    Parameters
    ----------
    samples : torch.Tensor
        Samples of the function to be transformed.
    shape : tuple
        Shape of the output array.

    """

    def __init__(self, samples, shape, keops_backend="auto"):
        self.samples = proper_trajectory(samples, "pi").astype(np.float32)
        self.shape = shape
        self.dim = samples.shape[-1]
        self.dtype = samples.dtype

        if isinstance(samples, np.ndarray):
            self._LazyTensor = NumpyLazyTensor
            self._cLazyTensor = NumpyComplexLazyTensor
            self._Genred = NumpyGenred
        elif PYTORCH_AVAILABLE and isinstance(samples, torch.Tensor):
            self._LazyTensor = TorchLazyTensor
            self._cLazyTensor = TorchComplexLazyTensor
            self._Genred = TorchGenred
        else:
            raise ValueError("Samples must be a numpy or torch array")

        # location in the image domain.
        self._locs = (
            np.array(np.meshgrid(*[np.arange(s) for s in self.shape], indexing="ij"))
            .reshape(self.dim, -1)
            .T.astype(np.float32)
        )

        # Fourier Transform image -> kspace
        variables = ["x_j = Vj(1,{dim})", "nu_i = Vi(0,{dim})", "b_j = Vi(2,2)"]
        aliases = [s.format(dim=self.dim) for s in variables]
        self._op = self._Genred(
            "ComplexMult(ComplexExp1j( - nu_i | x_j),  b_j)",
            aliases,
            reduction_op="Sum",
            axis=1,
            backend=keops_backend,
        )

        # Adjoint Fourier Transform kspace -> image
        variables = ["x_i = Vi(0,{dim})", "nu_j = Vj(1,{dim})", "c_j = Vj(2,2)"]
        aliases = [s.format(dim=self.dim) for s in variables]
        self._adj_op = self._Genred(
            "ComplexMult(ComplexExp1j(nu_j | x_i), c_j)",
            aliases,
            reduction_op="Sum",
            axis=1,
            backend=keops_backend,
        )

    def op(self, coeffs, image):
        """Apply the Fourier transform."""
        image_ = image.astype(np.complex64).reshape(-1, 1)
        # Compute the Fourier transform
        coeffs = self._op(self.samples, self._locs, image_).view(np.complex64)
        return coeffs

    def adj_op(self, coeffs, image):
        """Apply the adjoint Fourier transform."""
        coeffs_ = coeffs.astype(np.complex64).reshape(-1, 1)

        # Compute the adjoint Fourier transform
        image = self._adj_op(self._locs, self.samples, coeffs_).view(np.complex64)

        return image.reshape(self.shape)


class MRIKeops(FourierOperatorCPU):
    """MRI Fourier operator using Keops."""

    backend = "keops"

    def __init__(self, samples, shape, n_coils, smaps=None, **kwargs):
        super().__init__(
            samples,
            shape,
            density=False,
            n_coils=n_coils,
            smaps=smaps,
            raw_op=KeopsNDFT(samples, shape, **kwargs),
        )
