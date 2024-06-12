"""Pytorch MRI Nufft Operators."""

from ..base import FourierOperatorBase, with_torch
from mrinufft._utils import proper_trajectory, power_method
import numpy as np
import cupy as cp

TORCH_AVAILABLE = True

try:
    import torchkbnufft as torchnufft
    import torch

except ImportError:
    TORCH_AVAILABLE = False


class MRITorchKbNufft(FourierOperatorBase):
    """MRI Transform Operator using Torch NUFFT.

    Parameters
    ----------
    samples: Tensor
        The samples location of shape ``Nsamples x N_dimensions``.
        It should be C-contiguous.
    shape: tuple
        Shape of the image space.
    n_coils: int
        Number of coils.
    density: bool or Tensor
       Density compensation support.
        - If a Tensor, it will be used for the density.
        - If True, the density compensation will be automatically estimated,
          using the fixed point method.
        - If False, density compensation will not be used.
    smaps: Tensor
    squeeze_dims: bool, default True
        If True, will try to remove the singleton dimension for batch and coils.
    """

    backend = "torchkbnufft"
    available = TORCH_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        n_trans=1,
        smaps=None,
        eps=1e-6,
        squeeze_dims=True,
    ):
        super().__init__()

        self.shape = shape
        self.n_coils = n_coils
        self.eps = eps
        self.squeeze_dims = squeeze_dims
        self.n_batchs = n_batchs
        self.dtype = None

        self._tkb_op = torchnufft.KbNufft(im_size=self.shape)
        self._tkb_adj_op = torchnufft.KbNufftAdjoint(im_size=self.shape)

        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
        samples = proper_trajectory(
            samples.astype(np.float32, copy=False), normalize="pi"
        )
        self.samples = samples.transpose(1, 0)
        self.samples = torch.tensor(samples)

        if density is True:
            self.density = torchnufft.calc_density_compensation_function(
                ktraj=self.samples, im_size=shape, num_iterations=15
            )
        elif density is False:
            self.density = None
        elif (
            torch.is_tensor(density)
            or isinstance(density, np.ndarray)
            or isinstance(density, cp.ndarray)
        ):
            self.density = density
        else:
            raise ValueError(
                "argument `density` of type" f"{type(density)} is invalid."
            )

        if smaps is None:
            self.smaps = None
        elif isinstance(smaps, torch.Tensor):
            self.smaps = smaps
        elif isinstance(smaps, np.ndarray) or isinstance(smaps, cp.ndarray):
            self.smaps = torch.tensor(smaps)
        else:
            raise ValueError("argument `smaps` of type" f"{type(smaps)} is invalid")

    @with_torch
    def op(self, data, ksp=None):
        """Forward operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor: Non-uniform Fourier transform of the input image.
        """
        ktraj = self.samples
        smaps = self.smaps

        B, C, XYZ = self.n_batchs, self.n_coils, self.shape
        if not B:
            B = 1
        if not C:
            C = 1
        data = data.reshape((B, 1 if self.uses_sense else C, *XYZ))

        if ktraj.shape[0] != data.shape[0]:
            ktraj = ktraj.permute(1, 0)
        if smaps is not None:
            smaps = smaps.to(data.dtype)
        kdata = self._tkb_op.forward(image=data, omega=ktraj, smaps=smaps)
        kdata /= self.norm_factor

        return self._safe_squeeze(kdata)

    @with_torch
    def adj_op(self, data, coeffs=None):
        """Backward Operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        ktraj = self.samples
        smaps = self.smaps
        B, C, K, XYZ = self.n_batchs, self.n_coils, self.n_samples, self.shape
        if not B:
            B = 1
        if not C:
            C = 1
        data = data.reshape((B, C, K))

        if ktraj.shape[0] != data.shape[0]:
            ktraj = ktraj.permute(1, 0)
        if self.density:
            data = data * self.density

        if smaps is not None:
            smaps = smaps.to(data.dtype)

        img = self._tkb_adj_op.forward(data=data, omega=ktraj, smaps=smaps)
        img = img.reshape((B, 1 if self.uses_sense else C, *XYZ))

        return self._safe_squeeze(img)

    @with_torch
    def data_consistency(self, data, obs_data):
        """Compute the data consistency.

        Parameters
        ----------
        data: Tensor
            Image data
        obs_data: Tensor
            Observed data

        Returns
        -------
        Tensor
            The data consistency error in image space.
        """
        return self.adj_op(self.op(data) - obs_data)

    def _safe_squeeze(self, arr):
        """Squeeze the first two dimensions of shape of the operator."""
        if self.squeeze_dims:
            try:
                arr = arr.squeeze(axis=1)
            except ValueError:
                pass
            try:
                arr = arr.squeeze(axis=0)
            except ValueError:
                pass
        return arr
