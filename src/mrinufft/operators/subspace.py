"""Subspace NUFFT Operator wrapper."""

from mrinufft._array_compat import _get_device, _to_interface
from mrinufft._utils import ARRAY_LIBS, get_array_module


from .base import FourierOperatorBase


class MRISubspace(FourierOperatorBase):
    """Fourier Operator with subspace projection.

    This is a wrapper around the Fourier Operator to project
    data on a low-rank subspace (e.g., dynamic and multi-contrast MRI).

    Parameters
    ----------
    fourier_op: object of class FourierBase
        the fourier operator to wrap
    subspace_basis : np.ndarray
        Low rank subspace basis of shape (K, T),
        where K is the rank of the subspace and T is the number
        of time frames or contrasts in the original image series.
        Also supports Cupy arrays and Torch tensors.
    use_gpu : bool, optional
        Whether to use the GPU. Default is False.
        Ignored if the fourier operator internally use only GPU (e.g., cupy)
        or CPU (e.g., numpy)

    Notes
    -----
    This extension add an axis on the leftmost position for both
    image and k-space data:

    * Image: ``(B, C, XYZ)`` -> ``(T, B, C, XYZ)``
    * K-Space: ``(B, C, K)`` -> ``(T, B, C, K)``

    with ``T`` representing time domain or contrast space (for dynamic and m
    multi-contrast MRI, respectively).

    Similarly, k-space trajectory is expected to have the following shape:
    ``(<N_frames or N_contrasts>, N_shots, N_samples, dim)``. The flatten
    version is also accepted: ``(<N_frames or N_contrasts>, N_shots * N_samples, dim)``

    """

    def __init__(self, fourier_op, subspace_basis, use_gpu=False):
        self._fourier_op = fourier_op
        self.backend = fourier_op.backend

        self.n_batchs = fourier_op.n_batchs
        self.n_coils = fourier_op.n_coils
        self.shape = fourier_op.shape
        self.smaps = fourier_op.smaps
        self.autograd_available = fourier_op.autograd_available

        self.subspace_basis = subspace_basis
        self.n_coeffs, self.n_frames = self.subspace_basis.shape

    def op(self, data, *args):
        """
        Compute Forward Operation time/contrast-domain backprojection.

        Parameters
        ----------
        data: numpy.ndarray
            N-D subspace-projected image.
            Also supports Cupy arrays and Torch tensors.

        Returns
        -------
        numpy.ndarray
            Time/contrast-domain k-space data.
        """
        xp = get_array_module(data)
        device = _get_device(data)
        subspace_basis = _to_interface(self.subspace_basis, xp, device)

        # perform computation
        y = 0.0
        for idx in range(self.n_coeffs):

            # select basis element
            basis_element = subspace_basis[idx]

            # actual transform
            _y = self._fourier_op.op(data[idx], *args)
            _y = _y.reshape(*_y.shape[:-1], self.n_frames, -1)

            # back-project on time domain
            y += basis_element.conj() * _y.swapaxes(-2, -1)

        return y[None, ...].swapaxes(0, -1)[..., 0]  # bring back time domain in front

    def adj_op(self, coeffs, *args):
        """
        Compute Adjoint Operation with subspace projection.

        Parameters
        ----------
        coeffs: numpy.ndarray
            Time/contrast-domain k-space data.
            Also supports Cupy arrays and Torch tensors.

        Returns
        -------
        numpy.ndarray
            Inverse Fourier transform of the subspace-projected k-space.
        """
        xp = get_array_module(coeffs)
        device = _get_device(coeffs)
        subspace_basis = _to_interface(self.subspace_basis, xp, device)
        coeffs_d = coeffs[..., None].swapaxes(0, -1)[0, ...]

        # perform computation
        y = []
        for idx in range(self.n_coeffs):

            # select basis element
            basis_element = subspace_basis[idx]

            # project on current subspace basis element
            _coeffs_d = basis_element * coeffs_d
            _coeffs_d = _coeffs_d.swapaxes(-2, -1).reshape(*coeffs_d.shape[:-2], -1)

            # actual transform
            y.append(self._fourier_op.adj_op(_coeffs_d, *args))

        # stack coefficients
        y = xp.stack(y, axis=0)

        return y


def _get_arraylib_from_operator(
    fourier_op, use_gpu
):  # maybe that is usefull for MRIFourierCorrected constructor?
    LUT = {
        "MRIBartNUFFT": ("numpy", "numpy"),
        "MRICufiNUFFT": ("cupy", "cupy"),
        "MRIfinufft": ("numpy", "numpy"),
        "MRIGpuNUFFT": ("cupy", "cupy"),
        "MRInfft": ("numpy", "numpy"),
        "MRIPynufft": ("numpy", "numpy"),
        "MRISigpyNUFFT": ("numpy", "cupy"),
        "MRITensorflowNUFFT": ("tensorflow", "tensorflow"),
        "MRITorchKbNufft": ("torch", "torch"),
        "TorchKbNUFFTcpu": ("torch", "torch"),
        "TorchKbNUFFTgpu": ("torch", "torch"),
    }
    return ARRAY_LIBS[LUT[fourier_op.__class__.__name__][use_gpu]][0]
