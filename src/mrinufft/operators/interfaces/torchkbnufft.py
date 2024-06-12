"""Pytorch MRI Nufft Operators."""

from ..base import FourierOperatorBase, with_torch
from mrinufft._utils import proper_trajectory, power_method
import numpy as np

TORCH_AVAILABLE = True
try:
    import torchkbnufft as tkbn
    import torch

except ImportError:
    TORCH_AVAILABLE = False

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False


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
    autograd_available = False

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

        self._tkb_op = tkbn.KbNufft(im_size=self.shape)
        self._tkb_adj_op = tkbn.KbNufftAdjoint(im_size=self.shape)

        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
        samples = proper_trajectory(
            samples.astype(np.float32, copy=False), normalize="pi"
        )
        self.samples = samples.transpose(1, 0)
        self.samples = torch.tensor(samples)

        self.compute_density(density)

        self.compute_smaps(smaps)
        if isinstance(smaps, np.ndarray) or isinstance(smaps, cp.ndarray):
            self.smaps = torch.tensor(smaps)

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
        samples = self.samples
        smaps = self.smaps

        B, C, XYZ = self.n_batchs, self.n_coils, self.shape
        data = data.reshape((B, 1 if self.uses_sense else C, *XYZ))

        if samples.shape[0] != data.shape[0]:
            samples = samples.permute(1, 0)
        if smaps is not None:
            smaps = smaps.to(data.dtype)
        kdata = self._tkb_op.forward(image=data, omega=samples, smaps=smaps)
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
        samples = self.samples
        smaps = self.smaps
        B, C, K, XYZ = self.n_batchs, self.n_coils, self.n_samples, self.shape
        data = data.reshape((B, C, K))

        if samples.shape[0] != data.shape[0]:
            samples = samples.permute(1, 0)
        if self.density:
            data = data * self.density

        if smaps is not None:
            smaps = smaps.to(data.dtype)

        img = self._tkb_adj_op.forward(data=data, omega=samples, smaps=smaps)
        img = img.reshape((B, 1 if self.uses_sense else C, *XYZ))
        img /= self.norm_factor

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

    @classmethod
    @with_torch
    def pipe(
        cls,
        kspace_loc,
        volume_shape,
        num_iterations=10,
        osf=2,
        normalize=True,
        **kwargs,
    ):
        """Compute the density compensation weights for a given set of kspace locations.

        Parameters
        ----------
        kspace_loc: Tensor
            the kspace locations
        volume_shape: tuple
            the volume shape
        num_iterations: int default 10
            the number of iterations for density estimation
        osf: float or int
            The oversampling factor the volume shape
        normalize: bool
            Whether to normalize the density compensation.
            We normalize such that the energy of PSF = 1
        """
        volume_shape = (np.array(volume_shape) * osf).astype(int)
        grid_op = MRITorchKbNufft(
            samples=kspace_loc,
            shape=volume_shape,
            osf=1,
            **kwargs,
        )
        density_comp = tkbn.calc_density_compensation_function(
                ktraj=kspace_loc, im_size=volume_shape, num_iterations=num_iterations
            )
        if normalize:
            spike = torch.zeros(volume_shape, dtype=torch.float32)
            mid_loc = tuple(v // 2 for v in volume_shape)
            spike[mid_loc] = 1
            psf = grid_op.adj_op(grid_op.op(spike))
            density_comp /= torch.norm(psf)
        
        return density_comp.squeeze()