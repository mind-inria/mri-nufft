"""Tensorflow MRI Nufft Operators."""

import numpy as np

from ..base import FourierOperatorBase
from mrinufft._array_compat import with_tensorflow, TENSORFLOW_AVAILABLE
from mrinufft._utils import proper_trajectory

if TENSORFLOW_AVAILABLE:
    import tensorflow as tf

try:
    import tensorflow_nufft as tfnufft
    import tensorflow_mri as tfmri
except ImportError:
    TENSORFLOW_AVAILABLE = False


class MRITensorflowNUFFT(FourierOperatorBase):
    """MRI Transform Operator using Tensorflow NUFFT.

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
    """

    backend = "tensorflow"
    available = TENSORFLOW_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        n_coils=1,
        density=False,
        smaps=None,
        eps=1e-6,
    ):
        super().__init__()

        self.shape = shape
        self.n_coils = n_coils
        self.eps = eps

        self.compute_density(density)

        if isinstance(samples, tf.Tensor):
            samples = samples.numpy()
        samples = proper_trajectory(
            samples.astype(np.float32, copy=False), normalize="pi"
        )
        self.samples = tf.convert_to_tensor(samples)
        self.dtype = samples.dtype
        self.compute_smaps(smaps)
        if self.smaps is not None and not isinstance(self.smaps, tf.Tensor):
            self.smaps = tf.convert_to_tensor(self.smaps)

    @with_tensorflow
    def op(self, data):
        """Forward operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        self.check_shape(image=data)
        if self.uses_sense:
            data_d = data * self.smaps
        else:
            data_d = data
        coeff = tfnufft.nufft(
            data_d,
            self.samples,
            self.shape,
            transform_type="type_2",
            fft_direction="forward",
            tol=self.eps,
        )
        coeff /= self.norm_factor
        return coeff

    @with_tensorflow
    def adj_op(self, coeffs):
        """
        Backward Operation.

        Parameters
        ----------
        coeffs: Tensor

        Returns
        -------
        Tensor
        """
        self.check_shape(ksp=coeffs)
        if self.uses_density:
            coeffs_d = coeffs * self.density
        else:
            coeffs_d = coeffs
        img = tfnufft.nufft(
            coeffs_d,
            self.samples,
            self.shape,
            transform_type="type_1",
            fft_direction="backward",
            tol=self.eps,
        )
        img /= self.norm_factor
        if self.uses_sense:
            return tf.math.reduce_sum(img * tf.math.conj(self.smaps), axis=0)
        else:
            return img

    @property
    def norm_factor(self):
        """Norm factor of the operator."""
        return np.sqrt(np.prod(self.shape) * 2 ** len(self.shape))

    @with_tensorflow
    def data_consistency(self, image_data, obs_data):
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
        return self.adj_op(self.op(image_data) - obs_data)

    @classmethod
    def pipe(
        cls,
        samples,
        shape,
        num_iterations=10,
        osf=2,
        normalize=True,
    ):
        """Estimate the density compensation using the pipe method.

        Parameters
        ----------
        samples: Tensor
            The samples location of shape ``Nsamples x N_dimensions``.
            It should be C-contiguous.
        shape: tuple
            Shape of the image space.
        n_iter: int
            Number of iterations.
        osf: int, default 2
            Currently, we support only OSF=2 and this value cannot be changed.
            Changing will raise an error.

        Returns
        -------
        Tensor
            The estimated density compensation.
        """
        if TENSORFLOW_AVAILABLE is False:
            raise ValueError(
                "tensorflow is not available, cannot estimate the density compensation"
            )
        if osf != 2:
            raise ValueError("Tensorflow does not support OSF != 2")

        density_comp = tf.math.reciprocal_no_nan(
            tfmri.estimate_density(
                samples.astype(np.float32),
                shape,
                method="pipe",
                max_iter=num_iterations,
            )
        )

        if normalize:
            fourier_op = MRITensorflowNUFFT(samples, shape)
            spike = np.zeros(shape)
            mid_loc = tuple(v // 2 for v in shape)
            spike[mid_loc] = 1
            psf = fourier_op.adj_op(fourier_op.op(spike.astype(np.complex64)))
            density_comp /= np.linalg.norm(psf)

        return np.squeeze(density_comp)
