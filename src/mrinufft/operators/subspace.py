"""Subspace NUFFT Operator wrapper."""

from mrinufft._array_compat import _get_device, _to_interface
from mrinufft._utils import ARRAY_LIBS, get_array_module


from .base import FourierOperatorBase


class MRISubspace(FourierOperatorBase):
    """Fourier Operator with subspace projection.

    This is a wrapper around the Fourier Operator to project
    data onto a low-rank subspace (e.g., dynamic and multi-contrast MRI).

    Parameters
    ----------
    fourier_op: object of class FourierBase
        the fourier operator to wrap
    subspace_basis : np.ndarray
        Low rank subspace basis of shape ``(K, T)``,
        where K is the rank of the subspace and T is the number
        of time frames or contrasts in the original image series.
        Also supports Cupy arrays and Torch tensors.
    use_gpu : bool, optional
        Whether to use the GPU. Default is False.
        Ignored if the Fourier operator internally use only GPU (e.g., Cupy)
        or CPU (e.g., Numpy)

    Notes
    -----
    This extension adds a new axis for both image and k-space data:

    * Image: ``(B, C, XYZ)`` -> ``(B, S, C, XYZ)``
    * K-Space: ``(B, C, K)`` -> ``(B, T, C, K)``

    with ``S`` representing the subspace index and ``T`` representing time
    domain or contrast space (for dynamic and multi-contrast MR, respectively).

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

        # if required, move subspace index axis to leftmost position
        if self.n_batchs != 1 or data.shape[0] == 1:  # non-squeezed data
            data_d = data.swapaxes(0, 1)
        else:
            data_d = data

        # enforce data contiguity
        if xp.__name__ == "torch":
            data_d = data_d.contiguous()
        else:
            data_d = xp.ascontiguousarray(data_d)

        # perform computation
        y = 0.0
        for idx in range(self.n_coeffs):

            # select basis element
            basis_element = subspace_basis[idx]

            # actual transform
            _y = self._fourier_op.op(data_d[idx], *args)
            _y = _y.reshape(*_y.shape[:-1], self.n_frames, -1)

            # back-project on time domain
            y += basis_element.conj() * _y.swapaxes(-2, -1)

        y = y[None, ...].swapaxes(0, -1)[..., 0]

        # bring back time/contrast axis to original position (B, T, ...)
        if self.n_batchs != 1 or data.shape[0] == 1:  # non-squeezed data
            y = y.swapaxes(0, 1)

        return y

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

        # if required, move time/contrast axis to leftmost position
        if self.n_batchs != 1 or coeffs.shape[0] == 1:  # non-squeezed data
            coeffs_d = coeffs.swapaxes(0, 1)
        else:
            coeffs_d = coeffs

        coeffs_d = coeffs_d[..., None].swapaxes(0, -1)[0, ...]

        # perform computation
        y = []
        for idx in range(self.n_coeffs):

            # select basis element
            basis_element = subspace_basis[idx]

            # project on current subspace basis element
            _coeffs_d = basis_element * coeffs_d
            _coeffs_d = _coeffs_d.swapaxes(-2, -1).reshape(*coeffs_d.shape[:-2], -1)

            # enforce data contiguity
            if xp.__name__ == "torch":
                _coeffs_d = _coeffs_d.contiguous()
            else:
                _coeffs_d = xp.ascontiguousarray(_coeffs_d)

            # actual transform
            y.append(self._fourier_op.adj_op(_coeffs_d, *args))

        # stack coefficients
        y = xp.stack(y, axis=0)

        # bring back subspace index to original position (B, S, ...)
        if self.n_batchs != 1 or coeffs.shape[0] == 1:
            y = y.swapaxes(0, 1)

        return y

    @property
    def n_samples(self):
        """Return the number of samples used by the operator."""
        return self._fourier_op.n_samples


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
