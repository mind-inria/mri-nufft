"""MRI-NUFFT interface for the GROG gridding backend."""

import numpy as np

from mrinufft._array_compat import _array_to_numpy, _module_available, with_torch
from mrinufft._utils import proper_trajectory
from mrinufft.operators.base import FourierOperatorBase

GROG_AVAILABLE = _module_available("pygrog")


class MRIGrog(FourierOperatorBase):
    """MRI operator using PyGROG's geometric sparse Fourier transform.

    Parameters
    ----------
    samples : array
        Non-Cartesian trajectory in MRI-NUFFT coordinates.
    shape : tuple[int, ...]
        Image-space shape. GROG supports 2D and 3D images.
    density : array, optional
        Additional density compensation weights.
    n_coils : int, optional
        Number of coils. Default is 1.
    n_batchs : int, optional
        Number of batches. Default is 1.
    smaps : array, optional
        Sensitivity maps with shape ``(n_coils, *shape)``.
    oversamp : float, optional
        Cartesian grid oversampling factor. Default is 2.
    kernel_width : int, optional
        GROG interpolation kernel width. Default is 2.
    kernel_shape : str, optional
        GROG interpolation kernel shape. Default is ``"circle"``.
    device : str, optional
        Device used by the GROG operator.
    time_map : array, optional
        Sampling time for each trajectory point.
    squeeze_dims : bool, optional
        If True, squeeze singleton batch and coil dimensions from outputs.
    """

    backend = "grog"
    available = GROG_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        smaps=None,
        oversamp=2,
        kernel_width=2,
        kernel_shape="circle",
        device=None,
        time_map=None,
        squeeze_dims=True,
    ):
        super().__init__()

        from pygrog.calib import GrogInterpolator
        from pygrog.operator import SparseFFT

        self.shape = tuple(shape)
        if len(self.shape) not in (2, 3):
            raise ValueError("GROG supports only 2D and 3D image shapes.")
        if any(size < 1 for size in self.shape):
            raise ValueError("Image dimensions must be positive.")

        samples = np.asarray(_array_to_numpy(samples), dtype=np.float32)
        if samples.size == 0:
            raise ValueError("The trajectory must contain at least one sample.")
        samples = proper_trajectory(samples, normalize="pi")
        if samples.shape[-1] != len(self.shape):
            raise ValueError(
                f"Trajectory dimension {samples.shape[-1]} does not match "
                f"image dimension {len(self.shape)}."
            )
        self._samples = samples
        self.dtype = samples.dtype
        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.squeeze_dims = squeeze_dims
        self.compute_density(density)
        if (
            smaps is not None
            and not isinstance(smaps, (str, dict))
            and not callable(smaps)
        ):
            smaps = np.asarray(_array_to_numpy(smaps), dtype=np.complex64)
        self.compute_smaps(smaps)

        self._grog_interpolator_class = GrogInterpolator
        time_map = None if time_map is None else np.asarray(_array_to_numpy(time_map))
        if time_map is not None:
            time_map = time_map.reshape(-1)
            if len(time_map) != len(samples):
                raise ValueError("`time_map` must have one value per trajectory point.")
        self._grog_options = {
            "oversamp": oversamp,
            "kernel_width": kernel_width,
            "kernel_shape": kernel_shape,
            "time_map": time_map,
        }
        self._sparse_fft_class = SparseFFT
        self.device = device
        self._build_grog()
        self._make_raw_op()

    def _build_grog(self):
        """Create a GROG plan for the current trajectory."""
        coords = proper_trajectory(self.samples, normalize="unit") * np.asarray(
            self.shape, dtype=self.samples.dtype
        )
        self.grog = self._grog_interpolator_class(
            self.shape, coords, **self._grog_options
        )

    def _make_raw_op(self):
        """Create the low-level GROG FFT operator from the calibrated plan."""
        self.raw_op = self._sparse_fft_class(
            plan=self.grog.fft_plan(), smaps=self.smaps, device=self.device
        )
        self._op_scale = np.sqrt(np.prod(self.raw_op.grid_shape)) / self.norm_factor

    def update_samples(self, new_samples, *, unsafe=False):
        """Update the trajectory and rebuild its GROG plan."""
        samples = np.asarray(_array_to_numpy(new_samples), dtype=np.float32)
        if samples.size == 0:
            raise ValueError("The trajectory must contain at least one sample.")
        if unsafe:
            samples = samples.reshape(-1, len(self.shape))
        else:
            samples = proper_trajectory(samples, normalize="pi")
        if samples.shape[-1] != len(self.shape):
            raise ValueError(
                f"Trajectory dimension {samples.shape[-1]} does not match "
                f"image dimension {len(self.shape)}."
            )
        time_map = self._grog_options["time_map"]
        if time_map is not None and len(time_map) != len(samples):
            raise ValueError(
                "Updating the trajectory requires a matching `time_map`."
            )
        self._samples = samples
        self.compute_density(self._density_method)
        self._build_grog()
        self._make_raw_op()

    @property
    def smaps(self):
        """Sensitivity maps of the operator."""
        return self._smaps

    @smaps.setter
    def smaps(self, new_smaps):
        if (
            new_smaps is not None
            and not isinstance(new_smaps, (str, dict))
            and not callable(new_smaps)
        ):
            new_smaps = np.asarray(_array_to_numpy(new_smaps), dtype=np.complex64)
        FourierOperatorBase.smaps.fset(self, new_smaps)
        if hasattr(self, "grog"):
            self._make_raw_op()

    @with_torch
    def op(self, image):
        """Apply the forward transform from image space to non-Cartesian k-space."""
        import torch

        self.check_shape(image=image)
        image = image.to(torch.complex64)
        if self.uses_sense:
            image = image.reshape(self.n_batchs, *self.shape)
        else:
            image = image.reshape(self.n_batchs, self.n_coils, *self.shape)
        return self._safe_squeeze(self.raw_op.forward(image)) * self._op_scale

    @with_torch
    def adj_op(self, coeffs):
        """Apply the adjoint transform from non-Cartesian k-space to image space."""
        import torch

        self.check_shape(ksp=coeffs)
        coeffs = coeffs.to(torch.complex64).reshape(
            self.n_batchs, self.n_coils, self.n_samples
        )
        if self.density is not None:
            density = torch.as_tensor(
                np.asarray(_array_to_numpy(self.density)).reshape(-1),
                device=coeffs.device,
                dtype=coeffs.real.dtype,
            )
            coeffs = coeffs * density.reshape((1,) * (coeffs.ndim - 1) + (-1,))
        image = self.raw_op.adjoint(coeffs)
        return self._safe_squeeze(image) * self._op_scale
