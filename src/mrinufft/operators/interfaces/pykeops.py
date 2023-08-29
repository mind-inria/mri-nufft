"""Non Uniform Fourier transform with PyKeOps and torch."""

import numpy as np
from .base import FourierOperatorCPU, proper_trajectory

PYKEOPS_AVAILABLE = True
try:
    from pykeops.numpy import ComplexLazyTensor as NumpyComplexLazyTensor
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

    def __init__(self, samples, shape):
        self.samples = proper_trajectory(samples, "unit")
        self.shape = shape
        self.dim = samples.shape[1]
        self.dtype = samples.dtype

        if isinstance(samples, np.ndarray):
            self._Genred = NumpyGenred
            self._LazyTensor = NumpyComplexLazyTensor
        elif PYTORCH_AVAILABLE and isinstance(samples, torch.Tensor):
            self._Genred = TorchGenred
            self._LazyTensor = TorchComplexLazyTensor
        else:
            raise ValueError("Samples must be a numpy or torch array")

        self._precomp_samples = self._LazyTensor(2 * np.pi * 1j * self.samples, axis=0)

        # location in the image domain.
        self._image_loc = self._LazyTensor(
            np.array(np.meshgrid(*[np.arange(s) for s in self.shape]))
            .reshape(self.dim, -1)
            .T
        )

        self._op = self._Genred(
            "Sum_Reduction(x * Exp( - nu | px ))",
            variable=["x = Vi(1,1)", "nu = Vj(1,2)" "px = Vi(1,2)"],
            axis=0,
        )

        self._adj_op = self._Genred(
            "Sum_Reduction(x * Exp( + nu | px ))",
            variable=["x = Vi(1,1)", "nu = Vj(1,2)" "px = Vi(1,2)"],
            axis=1,
        )

    def op(self, coeffs, image):
        """Apply the Fourier transform."""
        image_ = self._LazyTensor(image)
        # Compute the Fourier transform
        coeffs = self._op(image_.flatten(), self._precomp_samples, self._image_loc)
        return coeffs

    def adj_op(self, coeffs, image):
        """Apply the adjoint Fourier transform."""
        coeffs_ = self._LazyTensor(coeffs)
        # Compute the adjoint Fourier transform
        image = self._adj_op(coeffs_, self._precomp_samples, self._image_loc)
        return image.reshape(self.shape)


class MRIKeops(FourierOperatorCPU):
    backend = "keops"

    def __init__(self, samples, shape, n_coils, smaps=None):
        super().__init__(
            samples,
            shape,
            density=False,
            n_coils=n_coils,
            smaps=smaps,
            raw_op=KeopsNDFT(samples, shape),
        )
