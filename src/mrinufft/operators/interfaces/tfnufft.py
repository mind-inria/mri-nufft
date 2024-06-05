"""Tensorflow MRI Nufft Operators."""

import numpy as np
from ..base import FourierOperatorBase

TENSORFLOW_AVAILABLE = True

try:
    import tensorflow_nufft as tfnufft
    import tensorflow_mri as tfmri
    import tensorflow as tf

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

        self.samples = samples
        self.shape = shape
        self.n_coils = n_coils
        self.eps = eps

        self.compute_density(density)

        if smaps is None:
            self.uses_sense = False
        elif tf.is_tensor(smaps):
            self.uses_sense = True
            self.smaps = smaps
        else:
            raise ValueError("argument `smaps` of type" f"{type(smaps)} is invalid")

    def op(self, data):
        """Forward operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        if self.uses_sense:
            data_d = data * self.smaps
        else:
            data_d = data
        return tfnufft.nufft(
            data_d,
            self.samples,
            self.shape,
            transform_type="type_2",
            fft_direction="backward",
            tol=self.eps,
        )

    def adj_op(self, data):
        """
        Backward Operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        if self.uses_density:
            data_d = data * self.density
        else:
            data_d = data
        img = tfnufft.nufft(
            data_d,
            self.samples,
            self.shape,
            transform_type="type_1",
            fft_direction="forward",
            tol=self.eps,
        )
        return tf.math.reduce_sum(img * tf.math.conj(self.smaps), axis=0)

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

    @classmethod
    def pipe(samples, shape, n_iter=15, normalize=True):
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

        Returns
        -------
        Tensor
            The estimated density compensation.
        """
        if TENSORFLOW_AVAILABLE is False:
            raise ValueError(
                "tensorflow is not available, cannot estimate the density compensation"
            )

        density_comp = tf.math.reciprocal_no_nan(
            tfmri.estimate_density(samples, shape, method="pipe", max_iter=15)
        )

        grid_op = MRITensorflowNUFFT(samples, shape, num_iter=n_iter)
        if normalize:
            spike = np.zeros(shape)
            mid_loc = tuple(v // 2 for v in shape)
            spike[mid_loc] = 1
            psf = grid_op.adj_op(grid_op.op(spike))
            density_comp /= np.linalg.norm(psf)

        return density_comp.squeeze()
