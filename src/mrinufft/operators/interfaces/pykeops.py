"""Non Uniform Fourier transform with PyKeOps and torch.

TODO: Use Keops for the non fourier stuff (e.g. smaps and multi coil )
"""

import numpy as np
from .base import FourierOperatorCPU, proper_trajectory

PYKEOPS_AVAILABLE = True
try:
    from pykeops.numpy import Genred as NumpyGenred
except ImportError:
    PYKEOPS_AVAILABLE = False

PYTORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    PYTORCH_AVAILABLE = False

if PYTORCH_AVAILABLE:
    from pykeops.torch import Genred as TorchGenred

PYKEOPS_GPU_AVAILABLE = PYTORCH_AVAILABLE and torch.cuda.is_available()


class KeopsRawNDFT:
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
        self.n_samples = len(samples)
        self.ndim = len(shape)
        self.dtype = samples.dtype

        if isinstance(samples, np.ndarray):
            self._Genred = NumpyGenred
        elif PYTORCH_AVAILABLE and isinstance(samples, torch.Tensor):
            self._Genred = TorchGenred
        else:
            raise ValueError("Samples must be a numpy or torch array")

        self._keops_backend = keops_backend
        # location in the image domain.
        self._locs = np.ascontiguousarray(
            np.reshape(
                np.meshgrid(*[np.arange(s) for s in self.shape], indexing="ij"),
                (self.ndim, -1),
            ).T.astype(np.float32)
        )
        self._keops_backend = keops_backend
        # Fourier Transform image -> kspace
        variables = ["x_j = Vj(1,{dim})", "nu_i = Vi(0,{dim})", "b_j = Vj(2,2)"]
        aliases = [s.format(dim=self.ndim) for s in variables]
        self._op = self._Genred(
            "ComplexMult(ComplexExp1j( - nu_i | x_j),  b_j)",
            aliases,
            reduction_op="Sum",
            axis=1,
        )

        # Adjoint Fourier Transform kspace -> image
        variables = ["x_j = Vj(1,{dim})", "nu_i = Vi(0,{dim})", "c_i = Vi(2,2)"]
        aliases = [s.format(dim=self.ndim) for s in variables]
        self._adj_op = self._Genred(
            "ComplexMult(c_i, ComplexExp1j(nu_i | x_j))",
            aliases,
            reduction_op="Sum",
            axis=0,
        )

    def op(self, coeffs, image):
        """Apply the Fourier transform."""
        image_ = image.astype(np.complex64).reshape(-1, 1)
        # Compute the Fourier transform
        np.copyto(
            coeffs,
            self._op(
                self.samples,
                self._locs,
                image_,
                backend=self._keops_backend,
            )
            .view(np.complex64)
            .reshape(-1),
        )
        return coeffs

    def adj_op(self, coeffs, image):
        """Apply the adjoint Fourier transform."""
        coeffs_ = coeffs.astype(np.complex64).reshape(-1, 1)

        # Compute the adjoint Fourier transform
        np.copyto(
            image,
            self._adj_op(
                self.samples,
                self._locs,
                coeffs_,
                backend=self._keops_backend,
            )
            .view(np.complex64)
            .reshape(self.shape),
        )
        return image


class MRIKeops(FourierOperatorCPU):
    """MRI Fourier operator using Keops."""

    backend = "pykeops"

    def __init__(self, samples, shape, n_coils, smaps=None, **kwargs):
        super().__init__(
            samples,
            shape,
            density=False,
            n_coils=n_coils,
            smaps=smaps,
        )

        self.raw_op = KeopsRawNDFT(self.samples, self.shape, **kwargs)


class MRIKeopsGPU(FourierOperatorCPU):
    """MRI Fourier operator using Keops."""

    backend = "pykeops-gpu"

    def __init__(self, samples, shape, n_coils, smaps=None, **kwargs):
        super().__init__(
            samples,
            shape,
            density=False,
            n_coils=n_coils,
            smaps=smaps,
        )

        self.raw_op = KeopsRawNDFT(
            self.samples, self.shape, keops_backend="GPU", **kwargs
        )
