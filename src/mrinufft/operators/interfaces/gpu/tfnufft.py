"""Tensorflow MRI Nufft Operators."""

from .base import FourierOperatorBase

try:
    import tensorflow_nufft as tfnufft
    import tensorflow_mri as tfmri
    import tensorflow as tf

except ImportError as exc:
    TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = True

class MRITensorflowNUFFT(FourierOperatorBase):
    """MRI Transform Operator using Tensorflow NUFFT.

    Parameters
    ----------
    samples: np.array
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
    def __init__(self, samples, shape, n_coils=1, density=False, smaps=None, eps=1e-6):
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow NUFFT is not available.")

        self.samples = samples
        self.shape = shape
        self.n_coils = n_coils
        self.eps = eps

        if density is True:
            self.density = tfmri.estimate_density
            self.uses_density = True
        elif density is False:
            self.density = None
            self.uses_density = False
        elif tf.is_tensor(density):
            self.density = density
            self.uses_density = True
        else:
            raise ValueError("argument `density` of type"
                             f"{type(density)} is invalid.")
        if smaps is None:
            self.uses_sense = False
        elif tf.is_tensor(smaps):
            self.uses_sense = True
            self.smaps = smaps
        else:
            raise ValueError("argument `smaps` of type"
                             f"{type(smaps)} is invalid")

    def op(self, data):
        """Forward operation. """
        if self.uses_sense:
            data_d = data * self.smaps
        else:
            data_d = data
        return tfnufft.nufft(data_d, self.samples, self.shape,
                        transform_type="type_2",
                        fft_direction="backward",
                        tol=self.eps)

    def adj_op(self, data):
        if self.uses_density:
            data_d = data * self.density
        else:
            data_d = data
        img =  tfnufft.nufft(data_d, self.samples, self.shape,
                        transform_type="type_1",
                        fft_direction="forward",
                        tol=self.eps)
        return tf.math.reduce_sum(img * tf.math.conj(self.smaps), axis=0)

    def data_consistency(self, data, obs_data):
        return self.adj_op(self.op(data)-obs_data)


    @classmethod
    def estimate_density(cls, samples, shape, n_iter=10):
        return tfmri.estimate_density(samples, shape, method="pipe", max_iter=n_iter)
