"""Pytorch MRI Nufft Operators."""

from mrinufft.operators.base import FourierOperatorBase
from mrinufft._array_compat import with_torch
from mrinufft._utils import proper_trajectory
from mrinufft.operators.interfaces.utils import (
    is_cuda_tensor,
)
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
    """
    MRI Transform Operator using Torch NUFFT.

    This class provides a Non-Uniform Fast Fourier Transform (NUFFT) operator
    for MRI data, utilizing the torchkbnufft library for performing the
    computations. It supports both CPU and GPU computations.

    Parameters
    ----------
    samples : Tensor
        The sample locations of shape ``Nsamples x N_dimensions``.
        It should be C-contiguous.
    shape : tuple
        Shape of the image space.
    density : bool or Tensor, optional
        Density compensation support. Default is False.
        - If a Tensor, it will be used for density.
        - If True, the density compensation will be automatically estimated
            using the fixed point method.
        - If False, density compensation will not be used.
    n_coils : int, optional
        Number of coils. Default is 1.
    n_batchs : int, optional
        Number of batches. Default is 1.
    smaps : Tensor, optional
        Sensitivity maps. Default is None.
    eps : float, optional
        A small epsilon value for numerical stability. Default is 1e-6.
    squeeze_dims : bool, optional
        If True, tries to remove singleton dimensions for batch and coils.
        Default is True.
    use_gpu : bool, optional
        Whether to use the GPU. Default is False.
    osf : int, optional
        Oversampling factor. Default is 2.
    **kwargs : dict
        Additional keyword arguments.

    """

    available = TORCH_AVAILABLE
    autograd_available = False

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        smaps=None,
        eps=1e-6,
        squeeze_dims=True,
        use_gpu=False,
        osf=2,
        **kwargs,
    ):
        super().__init__()

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"
        if isinstance(samples, torch.Tensor):
            if is_cuda_tensor(samples):
                samples = samples.cpu()
            samples = samples.numpy()
        samples = proper_trajectory(
            samples.astype(np.float32, copy=False), normalize="pi"
        )
        self.samples = torch.tensor(samples).to(self.device)

        self.dtype = None
        # self.dtype = self.samples.dtype
        self.shape = shape
        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.squeeze_dims = squeeze_dims
        # self.eps = eps
        self.compute_density(density)

        if isinstance(smaps, torch.Tensor):
            self.smaps = smaps
        else:
            self.compute_smaps(smaps)
            if self.smaps is not None:
                self.smaps = torch.tensor(self.smaps).to(self.device)

        self._tkb_op = tkbn.KbNufft(im_size=self.shape).to(self.device)
        self._tkb_adj_op = tkbn.KbNufftAdjoint(im_size=self.shape).to(self.device)

    @with_torch
    def op(self, data, out=None):
        """Forward operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor: Non-uniform Fourier transform of the input image.
        """
        self.check_shape(image=data, ksp=out)
        B, C, XYZ = self.n_batchs, self.n_coils, self.shape
        data = data.reshape((B, 1 if self.uses_sense else C, *XYZ))
        data = data.to(self.device, copy=False)

        if self.smaps is not None:
            self.smaps = self.smaps.to(data.dtype, copy=False)

        kdata = self._tkb_op.forward(
            image=data, omega=self.samples.t(), smaps=self.smaps
        )
        kdata /= self.norm_factor
        return self._safe_squeeze(kdata)

    @with_torch
    def adj_op(self, coeffs, out=None):
        """Backward Operation.

        Parameters
        ----------
        coeffs: Tensor

        Returns
        -------
        Tensor
        """
        self.check_shape(image=out, ksp=coeffs)
        B, C, K, XYZ = self.n_batchs, self.n_coils, self.n_samples, self.shape
        coeffs = coeffs.reshape((B, C, K))
        coeffs = coeffs.to(self.device, copy=False)

        if self.smaps is not None:
            self.smaps = self.smaps.to(coeffs.dtype, copy=False)
        if self.density:
            coeffs = coeffs * self.density

        img = self._tkb_adj_op.forward(
            data=coeffs, omega=self.samples.t(), smaps=self.smaps
        )
        img = img.reshape((B, 1 if self.uses_sense else C, *XYZ))
        img /= self.norm_factor
        return self._safe_squeeze(img)

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
        obs_data = obs_data.to(self.device, copy=False)
        ret = self.adj_op(self.op(data) - obs_data)
        return ret

    @classmethod
    @with_torch
    def pipe(
        cls,
        kspace_loc,
        volume_shape,
        num_iterations=10,
        osf=2,
        normalize=True,
        use_gpu=False,
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
        use_gpu: bool, default False
            Whether to use the GPU
        """
        volume_shape = (np.array(volume_shape) * osf).astype(int)
        grid_op = MRITorchKbNufft(
            samples=kspace_loc,
            shape=volume_shape,
            osf=1,
            use_gpu=use_gpu,
            **kwargs,
        )
        density_comp = tkbn.calc_density_compensation_function(
            ktraj=kspace_loc, im_size=volume_shape, num_iterations=num_iterations
        )
        if normalize:
            spike = torch.zeros(volume_shape, dtype=torch.float32).to(grid_op.device)
            mid_loc = tuple(v // 2 for v in volume_shape)
            spike[mid_loc] = 1
            psf = grid_op.adj_op(grid_op.op(spike))
            density_comp /= torch.norm(psf)

        return density_comp.squeeze()


class TorchKbNUFFTcpu(MRITorchKbNufft):
    """
    MRI Transform Operator using Torch NUFFT for CPU.

    This class provides a Non-Uniform Fast Fourier Transform (NUFFT) operator
    specifically optimized for CPU using the torchkbnufft library. It inherits
    from the MRITorchKbNufft class and sets the use_gpu parameter to False.

    """

    backend = "torchkbnufft-cpu"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_gpu=False)


class TorchKbNUFFTgpu(MRITorchKbNufft):
    """
    MRI Transform Operator using Torch NUFFT for GPU.

    This class provides a Non-Uniform Fast Fourier Transform (NUFFT) operator
    specifically optimized for GPU using the torchkbnufft library. It inherits
    from the MRITorchKbNufft class and sets the use_gpu parameter to True.

    """

    backend = "torchkbnufft-gpu"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_gpu=True)
