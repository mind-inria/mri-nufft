"""Stacked Operator for NUFFT."""

import warnings

import numpy as np
import scipy as sp

from mrinufft._utils import proper_trajectory, power_method, get_array_module, auto_cast
from mrinufft.operators.base import (
    FourierOperatorBase,
    check_backend,
    get_operator,
    with_numpy_cupy,
)
from mrinufft.operators.interfaces.utils import (
    is_cuda_array,
    is_host_array,
    pin_memory,
    sizeof_fmt,
)

CUPY_AVAILABLE = True
try:
    import cupy as cp
    from cupyx.scipy import fft as cpfft
except ImportError:
    CUPY_AVAILABLE = False


class MRIStackedNUFFT(FourierOperatorBase):
    """Stacked NUFFT Operator for MRI.

    The dimension of stacking is always the last one.

    Parameters
    ----------
    samples : array-like
        Sample locations in a 2D kspace
    shape: tuple
        Shape of the image.
    z_index: array-like
        Cartesian z index of masked plan. if "auto" the z_index is computed from the
        samples, if they are 3D, using the last coordinate.
    backend: str or FourierOperatorBase
        Backend to use.
        If str, a NUFFT operator is initialized with str being a registered backend.
        If FourierOperatorBase, operator is checked for compatibility and used as is
        notably one should have:
        ``n_coils = self.n_coils*len(z_index), squeeze_dims=True, smaps=None``

    smaps: array-like
        Sensitivity maps.
    n_coils: int
        Number of coils.
    n_batchs: int
        Number of batchs.
    **kwargs: dict
        Additional arguments to pass to the backend.
    """

    # Developer Notes:
    # Internally the stacked NUFFT operator (self) uses a backend MRI aware NUFFT
    # operator(op), configured as such:
    # - op.smaps=None
    # - op.n_coils self.n_coils * len(self.z_index) ; op.n_batch= 1.
    # The kspace is organized as a 2D array of shape
    # (self.n_batchs, self.n_coils, self.n_samples) Note that the stack dimension is
    # fused with the samples

    backend = "stacked"
    available = True  # the true availabily will be check at runtime.

    def __init__(
        self,
        samples,
        shape,
        backend,
        smaps,
        z_index="auto",
        n_coils=1,
        n_batchs=1,
        squeeze_dims=False,
        **kwargs,
    ):
        super().__init__()
        self.shape = shape
        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.squeeze_dims = squeeze_dims
        self.smaps = smaps
        if isinstance(backend, str):
            samples2d, z_index_ = self._init_samples(samples, z_index, shape)
            self._samples2d = samples2d.reshape(-1, 2)
            self.z_index = z_index_
            self.operator = get_operator(backend)(
                self._samples2d,
                shape[:-1],
                n_coils=self.n_coils * len(self.z_index),
                smaps=None,
                squeeze_dims=True,
                **kwargs,
            )
        elif isinstance(backend, FourierOperatorBase):
            # get all the interesting values from the operator
            if backend.shape != shape[:-1]:
                raise ValueError("Backend operator should have compatible shape")

            samples2d, z_index_ = self._init_samples(backend.samples, z_index, shape)
            self._samples2d = samples2d.reshape(-1, 2)
            self.z_index = z_index_

            if backend.n_coils != self.n_coils * (len(z_index_)):
                raise ValueError(
                    "The backend operator should have ``n_coils * len(z_index)``"
                    " specified for its coil dimension."
                )
            if backend.uses_sense:
                raise ValueError("Backend operator should not uses smaps.")
            if not backend.squeeze_dims:
                raise ValueError("Backend operator should have ``squeeze_dims=True``")
            self.operator = backend

        else:
            raise ValueError(
                "backend should either be a 2D nufft operator,"
                " or a str specifying which nufft library to use."
            )

    @staticmethod
    def _init_samples(samples, z_index, shape):
        samples_dim = samples.shape[-1]
        auto_z = isinstance(z_index, str) and z_index == "auto"
        if samples_dim == len(shape) and auto_z:
            # samples describes a 3D trajectory,
            # we  convert it to a 2D + index.
            samples2d, z_index_ = traj3d2stacked(samples, shape[-1])

        elif samples_dim == (len(shape) - 1) and not auto_z:
            # samples describes a 2D trajectory
            samples2d = samples
            if z_index is None:
                z_index_ = np.ones(shape[-1], dtype=bool)
            try:
                z_index_ = np.arange(shape[-1])[z_index]
            except IndexError as e:
                raise ValueError(
                    "z-index should be a boolean array of length shape[-1], "
                    "or an array of integer."
                ) from e
        else:
            raise ValueError("Invalid samples or z-index")
        return samples2d, z_index_

    @property
    def dtype(self):
        """Return dtype."""
        return self.operator.dtype

    @dtype.setter
    def dtype(self, dtype):
        self.operator.dtype = dtype

    @property
    def n_samples(self):
        """Return number of samples."""
        return len(self._samples2d) * len(self.z_index)

    @staticmethod
    def _fftz(data):
        """Apply FFT on z-axis."""
        xp = get_array_module(data)
        # sqrt(2) required for normalization
        return xp.fft.fftshift(
            xp.fft.fft(xp.fft.ifftshift(data, axes=-1), axis=-1, norm="ortho"), axes=-1
        ) / np.sqrt(2)

    @staticmethod
    def _ifftz(data):
        """Apply IFFT on z-axis."""
        # sqrt(2) required for normalization
        xp = get_array_module(data)
        return xp.fft.fftshift(
            xp.fft.ifft(xp.fft.ifftshift(data, axes=-1), axis=-1, norm="ortho"), axes=-1
        ) / np.sqrt(2)

    @with_numpy_cupy
    def op(self, data, ksp=None):
        """Forward operator."""
        if self.uses_sense:
            return self._safe_squeeze(self._op_sense(data, ksp))
        return self._safe_squeeze(self._op_calibless(data, ksp))

    def _op_sense(self, data, ksp=None):
        """Apply SENSE operator."""
        B, C, XYZ = self.n_batchs, self.n_coils, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)

        xp = get_array_module(data)
        if ksp is None:
            ksp = xp.empty((B, C, NZ, NS), dtype=self.cpx_dtype)
        ksp = ksp.reshape((B, C * NZ, NS))
        data_ = data.reshape(B, *XYZ)
        for b in range(B):
            data_c = data_[b] * self.smaps
            data_c = self._fftz(data_c)
            data_c = data_c.reshape(C, *XYZ)
            tmp = xp.ascontiguousarray(data_c[..., self.z_index])
            tmp = xp.moveaxis(tmp, -1, 1)
            tmp = tmp.reshape(C * NZ, *XYZ[:2])
            ksp[b, ...] = self.operator.op(xp.ascontiguousarray(tmp))
        ksp = ksp.reshape((B, C, NZ * NS))
        return ksp

    def _op_calibless(self, data, ksp=None):
        B, C, XYZ = self.n_batchs, self.n_coils, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)
        xp = get_array_module(data)
        if ksp is None:
            ksp = xp.empty((B, C, NZ, NS), dtype=self.cpx_dtype)
        ksp = ksp.reshape((B, C * NZ, NS))
        data_ = data.reshape(B, C, *XYZ)
        ksp_z = self._fftz(data_)
        ksp_z = ksp_z.reshape((B, C, *XYZ))
        for b in range(B):
            tmp = ksp_z[b][..., self.z_index]
            tmp = xp.moveaxis(tmp, -1, 1)
            tmp = tmp.reshape(C * NZ, *XYZ[:2])
            ksp[b, ...] = self.operator.op(xp.ascontiguousarray(tmp))
        ksp = ksp.reshape((B, C, NZ, NS))
        ksp = ksp.reshape((B, C, NZ * NS))
        return ksp

    @with_numpy_cupy
    def adj_op(self, coeffs, img=None):
        """Adjoint operator."""
        if self.uses_sense:
            return self._safe_squeeze(self._adj_op_sense(coeffs, img))
        return self._safe_squeeze(self._adj_op_calibless(coeffs, img))

    def _adj_op_sense(self, coeffs, img):
        B, C, XYZ = self.n_batchs, self.n_coils, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)

        xp = get_array_module(coeffs)
        imgz = xp.zeros((B, C, *XYZ), dtype=self.cpx_dtype)
        coeffs_ = coeffs.reshape((B, C * NZ, NS))
        for b in range(B):
            tmp = xp.ascontiguousarray(coeffs_[b, ...])
            tmp_adj = self.operator.adj_op(tmp)
            # move the z axis back
            tmp_adj = tmp_adj.reshape(C, NZ, *XYZ[:2])
            tmp_adj = xp.moveaxis(tmp_adj, 1, -1)
            imgz[b][..., self.z_index] = tmp_adj
        imgc = self._ifftz(imgz)
        img = img or xp.empty((B, *XYZ), dtype=self.cpx_dtype)
        for b in range(B):
            img[b] = xp.sum(imgc[b] * self.smaps.conj(), axis=0)
        return img

    def _adj_op_calibless(self, coeffs, img):
        B, C, XYZ = self.n_batchs, self.n_coils, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)

        xp = get_array_module(coeffs)
        imgz = xp.zeros((B, C, *XYZ), dtype=self.cpx_dtype)
        coeffs_ = coeffs.reshape((B, C, NZ, NS))
        coeffs_ = coeffs.reshape((B, C * NZ, NS))
        for b in range(B):
            t = xp.ascontiguousarray(coeffs_[b, ...])
            adj = self.operator.adj_op(t)
            # move the z axis back
            adj = adj.reshape(C, NZ, *XYZ[:2])
            adj = xp.moveaxis(adj, 1, -1)
            imgz[b][..., self.z_index] = xp.ascontiguousarray(adj)
        imgz = xp.reshape(imgz, (B, C, *XYZ))
        img = self._ifftz(imgz)
        return img

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

    def get_lipschitz_cst(self, max_iter=10):
        """Return the Lipschitz constant of the operator.

        Parameters
        ----------
        max_iter: int
            number of iteration to compute the lipschitz constant.
        **kwargs:
            Extra arguments givent

        Returns
        -------
        float
            Spectral Radius

        Notes
        -----
        This uses the Iterative Power Method to compute the largest singular value of a
        minified version of the nufft operator. No coil or B0 compensation is used,
        but includes any computed density.
        """
        return self.operator.get_lipschitz_cst(max_iter)

    @property
    def samples(self):
        """Return samples as a N_slice x N_samples x 3 array.

        Built from the 2D samples and the z_index normalized to [-0.5, 0.5).
        """
        samples = np.zeros(
            (len(self.z_index), len(self._samples2d), 3), dtype=self._samples2d.dtype
        )

        for i, idx in enumerate(self.z_index):
            z_coord = idx / self.shape[-1] - 0.5
            samples[i] = np.concatenate(
                [
                    self._samples2d,
                    z_coord
                    * np.ones((len(self._samples2d), 1), dtype=self._samples2d.dtype),
                ],
                axis=1,
            )

    @samples.setter
    def samples(self, samples):
        """Set samples."""
        self._samples2d, self.z_index = self._init_samples(samples, "auto", self.shape)
        self.operator.samples = self._samples2d


class MRIStackedNUFFTGPU(MRIStackedNUFFT):
    """
    Stacked NUFFT Operator for MRI using GPU only backend.

    This requires cufinufft to be installed.

    Parameters
    ----------
    samples : array-like
        Sample locations in a 2D kspace
    shape: tuple
        Shape of the image.
    z_index: array-like
        Cartesian z index of masked plan. if "auto" the z_index is computed from the
        samples, if they are 3D, using the last coordinate.
    smaps: array-like
        Sensitivity maps.
    n_coils: int
        Number of coils.
    n_batchs: int
        Number of batchs.
    **kwargs: dict
        Additional arguments to pass to the backend.
    """

    backend = "stacked-cufinufft"
    available = True  # the true availabily will be check at runtime.

    def __init__(
        self,
        samples,
        shape,
        smaps,
        n_coils=1,
        n_batchs=1,
        n_trans=1,
        z_index="auto",
        squeeze_dims=False,
        smaps_cached=False,
        density=False,
        backend="cufinufft",
        **kwargs,
    ):
        if not (CUPY_AVAILABLE and check_backend("cufinufft")):
            raise RuntimeError("Cupy and cufinufft are required for this backend.")

        if (n_batchs * n_coils) % n_trans != 0:
            raise ValueError("n_batchs * n_coils should be a multiple of n_transf")

        self.shape = shape
        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.squeeze_dims = squeeze_dims

        if isinstance(backend, str):
            samples2d, z_index_ = self._init_samples(samples, z_index, shape)
            self._samples2d = samples2d.reshape(-1, 2)
            self.z_index = z_index_
            self.operator = get_operator(backend)(
                self._samples2d,
                shape[:-1],
                n_coils=self.n_trans * len(self.z_index),
                n_trans=len(self.z_index),
                smaps=None,
                squeeze_dims=True,
                density=density,
                **kwargs,
            )
        elif isinstance(backend, FourierOperatorBase):
            # get all the interesting values from the operator
            if backend.shape != shape[:-1]:
                raise ValueError("Backend operator should have compatible shape")

            samples2d, z_index_ = self._init_samples(backend.samples, z_index, shape)
            self._samples2d = samples2d.reshape(-1, 2)
            self.z_index = z_index_

            if backend.n_coils != self.n_trans * len(z_index_):
                raise ValueError(
                    "The backend operator should have ``n_coils * len(z_index)``"
                    " specified for its coil dimension."
                )
            if backend.uses_sense:
                raise ValueError("Backend operator should not uses smaps.")
            if not backend.squeeze_dims:
                raise ValueError("Backend operator should have ``squeeze_dims=True``")
            self.operator = backend
        else:
            raise ValueError(
                "backend should either be a 2D nufft operator,"
                " or a str specifying which nufft library to use."
            )

        # Smaps support
        self.smaps = smaps
        self.smaps_cached = False
        if smaps is not None:
            if not (is_host_array(smaps) or is_cuda_array(smaps)):
                raise ValueError(
                    "Smaps should be either a C-ordered ndarray, " "or a GPUArray."
                )
            if smaps_cached:
                warnings.warn(
                    f"{sizeof_fmt(smaps.size * np.dtype(self.cpx_dtype).itemsize)}"
                    "used on gpu for smaps."
                )
                self.smaps = cp.array(
                    smaps, order="C", copy=False, dtype=self.cpx_dtype
                )
                self.smaps_cached = True
            else:
                self.smaps = pin_memory(smaps.astype(self.cpx_dtype))
                self._smap_d = cp.empty(self.shape, dtype=self.cpx_dtype)

    @property
    def norm_factor(self):
        """Norm factor of the operator."""
        return self.operator.norm_factor * np.sqrt(2)

    @staticmethod
    def _fftz(data):
        """Apply FFT on z-axis."""
        # sqrt(2) required for normalization
        return cpfft.fftshift(
            cpfft.fft(
                cpfft.ifftshift(data, axes=-1),
                axis=-1,
                norm="ortho",
                overwrite_x=True,
            ),
            axes=-1,
        )

    @staticmethod
    def _ifftz(data):
        """Apply IFFT on z-axis."""
        # sqrt(2) required for normalization
        return cpfft.fftshift(
            cpfft.ifft(
                cpfft.ifftshift(data, axes=-1),
                axis=-1,
                norm="ortho",
                overwrite_x=False,
            ),
            axes=-1,
        )

    @with_numpy_cupy
    def op(self, data, ksp=None):
        """Forward operator."""
        self.check_shape(image=data, ksp=ksp)
        # Dispatch to special case.
        data = auto_cast(data, self.cpx_dtype)

        if self.uses_sense and is_cuda_array(data):
            op_func = self._op_sense_device
        elif self.uses_sense:
            op_func = self._op_sense_host
        elif is_cuda_array(data):
            op_func = self._op_calibless_device
        else:
            op_func = self._op_calibless_host
        ret = op_func(data, ksp)

        return self._safe_squeeze(ret)

    def _op_sense_host(self, data, ksp=None):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)

        dataf = data.reshape((B, *XYZ))
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)

        if ksp is None:
            ksp = np.empty((B, C, NZ, NS), dtype=self.cpx_dtype)
        ksp = ksp.reshape((B * C, NZ * NS))
        ksp_batched = cp.empty((T * NZ, NS), dtype=self.cpx_dtype)
        for i in range((B * C) // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            # Send the n_trans coils to gpu
            data_batched.set(dataf[idx_batch].reshape((T, *XYZ)))
            # Apply Smaps
            if not self.smaps_cached:
                coil_img_d.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                cp.copyto(coil_img_d, self.smaps[idx_coils])
            coil_img_d *= data_batched
            # FFT along Z axis (last)
            coil_img_d = self._fftz(coil_img_d)
            coil_img_d = coil_img_d.reshape((T, *XYZ))
            tmp = coil_img_d[..., self.z_index]
            tmp = cp.moveaxis(tmp, -1, 1)
            tmp = tmp.reshape(T * NZ, *XYZ[:2])
            # After reordering, apply 2D NUFFT
            ksp_batched = self.operator._op_calibless_device(cp.ascontiguousarray(tmp))
            ksp_batched /= self.norm_factor
            ksp_batched = ksp_batched.reshape(T, NZ, NS)
            ksp_batched = ksp_batched.reshape(T, NZ * NS)
            ksp[i * T : (i + 1) * T] = ksp_batched.get()
        ksp = ksp.reshape((B, C, NZ * NS))
        return ksp

    def _op_sense_device(self, data, ksp):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)
        data = cp.asarray(data)
        dataf = data.reshape((B, *XYZ))
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)

        if ksp is None:
            ksp = cp.empty((B, C, NZ, NS), dtype=self.cpx_dtype)

        ksp = ksp.reshape((B * C, NZ * NS))
        ksp_batched = cp.empty((T * NZ, NS), dtype=self.cpx_dtype)
        for i in range((B * C) // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C

            data_batched = dataf[idx_batch].reshape((T, *XYZ))
            # Apply Smaps
            if not self.smaps_cached:
                coil_img_d.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                cp.copyto(coil_img_d, self.smaps[idx_coils])
            coil_img_d *= data_batched
            # FFT along Z axis (last)
            coil_img_d = self._fftz(coil_img_d)
            coil_img_d = coil_img_d.reshape((T, *XYZ))
            tmp = coil_img_d[..., self.z_index]
            tmp = cp.moveaxis(tmp, -1, 1)
            tmp = tmp.reshape(T * NZ, *XYZ[:2])
            # After reordering, apply 2D NUFFT
            ksp_batched = self.operator._op_calibless_device(cp.ascontiguousarray(tmp))
            ksp_batched /= self.norm_factor
            ksp_batched = ksp_batched.reshape(T, NZ, NS)
            ksp_batched = ksp_batched.reshape(T, NZ * NS)
            ksp[i * T : (i + 1) * T] = ksp_batched
        ksp = ksp.reshape((B, C, NZ * NS))
        return ksp

    def _op_calibless_host(self, data, ksp=None):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)

        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, NZ * NS), dtype=self.dtype)
        if ksp is None:
            ksp = np.zeros((B, C, NZ, NS), dtype=self.cpx_dtype)
        ksp = ksp.reshape((B * C, NZ * NS))

        dataf = data.reshape(B * C, *XYZ)

        for i in range((B * C) // T):
            coil_img_d.set(dataf[i * T : (i + 1) * T])
            coil_img_d = self._fftz(coil_img_d)
            coil_img_d = coil_img_d.reshape((T, *XYZ))
            tmp = coil_img_d[..., self.z_index]
            tmp = cp.moveaxis(tmp, -1, 1)
            tmp = tmp.reshape(T * NZ, *XYZ[:2])
            # After reordering, apply 2D NUFFT
            ksp_batched = self.operator._op_calibless_device(cp.ascontiguousarray(tmp))
            ksp_batched /= self.norm_factor
            ksp_batched = ksp_batched.reshape(T, NZ, NS)
            ksp_batched = ksp_batched.reshape(T, NZ * NS)
            ksp[i * T : (i + 1) * T] = ksp_batched.get()

        ksp = ksp.reshape((B, C, NZ * NS))
        return ksp

    def _op_calibless_device(self, data, ksp=None):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)
        data = cp.asarray(data)

        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, NZ * NS), dtype=self.dtype)
        if ksp is None:
            ksp = cp.zeros((B, C, NZ, NS), dtype=self.cpx_dtype)
        ksp = ksp.reshape((B * C, NZ * NS))

        dataf = data.reshape(B * C, *XYZ)

        for i in range((B * C) // T):
            coil_img_d = dataf[i * T : (i + 1) * T]
            coil_img_d = self._fftz(coil_img_d)
            coil_img_d = coil_img_d.reshape((T, *XYZ))
            tmp = coil_img_d[..., self.z_index]
            tmp = cp.moveaxis(tmp, -1, 1)
            tmp = tmp.reshape(T * NZ, *XYZ[:2])
            # After reordering, apply 2D NUFFT
            ksp_batched = self.operator._op_calibless_device(cp.ascontiguousarray(tmp))
            ksp_batched /= self.norm_factor
            ksp_batched = ksp_batched.reshape(T, NZ, NS)
            ksp_batched = ksp_batched.reshape(T, NZ * NS)
            ksp[i * T : (i + 1) * T] = ksp_batched

        ksp = ksp.reshape((B, C, NZ * NS))
        return ksp

    @with_numpy_cupy
    def adj_op(self, coeffs, img=None):
        """Adjoint operator."""
        if img is not None:
            self.check_shape(image=img, ksp=coeffs)
        # Dispatch to special case.
        coeffs = auto_cast(coeffs, self.cpx_dtype)

        if self.uses_sense and is_cuda_array(coeffs):
            adj_op_func = self._adj_op_sense_device
        elif self.uses_sense:
            adj_op_func = self._adj_op_sense_host
        elif is_cuda_array(coeffs):
            adj_op_func = self._adj_op_calibless_device
        else:
            adj_op_func = self._adj_op_calibless_host

        ret = adj_op_func(coeffs, img)

        return self._safe_squeeze(ret)

    def _adj_op_sense_host(self, coeffs, img_d=None):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)

        coeffs_f = coeffs.reshape(B * C, NZ * NS)
        # Allocate Memory
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        if img_d is None:
            img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, NS * NZ), dtype=self.cpx_dtype)

        for i in range((B * C) // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils])
            else:
                smaps_batched = self.smaps[idx_coils]
            ksp_batched.set(coeffs_f[i * T : (i + 1) * T])

            tmp_adj = self.operator._adj_op_calibless_device(ksp_batched)
            tmp_adj /= self.norm_factor
            tmp_adj = tmp_adj.reshape((T, NZ, *XYZ[:2]))
            tmp_adj = cp.moveaxis(tmp_adj, 1, -1)
            coil_img_d[:] = 0j
            coil_img_d[..., self.z_index] = tmp_adj
            coil_img_d = self._ifftz(coil_img_d)

            for t, b in enumerate(idx_batch):
                img_d[b, :] += coil_img_d[t] * smaps_batched[t].conj()
        img = img_d.get()
        img = img.reshape((B, 1, *XYZ))
        return img

    def _adj_op_sense_device(self, coeffs, img):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)
        coeffs = cp.asarray(coeffs)
        coeffs_f = coeffs.reshape(B * C, NZ * NS)
        # Allocate Memory
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        if img is None:
            img = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, NS * NZ), dtype=self.cpx_dtype)

        for i in range((B * C) // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils])
            else:
                smaps_batched = self.smaps[idx_coils]
            ksp_batched = coeffs_f[i * T : (i + 1) * T]

            tmp_adj = self.operator._adj_op_calibless_device(ksp_batched)
            tmp_adj /= self.norm_factor
            tmp_adj = tmp_adj.reshape((T, NZ, *XYZ[:2]))
            tmp_adj = cp.moveaxis(tmp_adj, 1, -1)
            coil_img_d[:] = 0j
            coil_img_d[..., self.z_index] = tmp_adj
            coil_img_d = self._ifftz(coil_img_d)

            for t, b in enumerate(idx_batch):
                img[b, :] += coil_img_d[t] * smaps_batched[t].conj()
        img = img.reshape((B, 1, *XYZ))
        return img

    def _adj_op_calibless_host(self, coeffs, img=None):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)
        coeffs_f = coeffs.reshape(B, C, NZ * NS)
        coeffs_f = coeffs_f.reshape(B * C, NZ, NS)
        coeffs_f = coeffs_f.reshape(B * C * NZ, NS)
        # Allocate Memory
        ksp_batched = cp.empty((T, NZ * NS), dtype=self.cpx_dtype)
        if img is None:
            img = np.zeros((B * C, *XYZ), dtype=self.cpx_dtype)
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        TZ = T * NZ
        for i in range((B * C * NZ) // TZ):
            ksp_batched = ksp_batched.reshape(TZ, NS)
            ksp_batched.set(coeffs_f[i * TZ : (i + 1) * TZ])
            ksp_batched = ksp_batched.reshape(TZ, NS)
            tmp_adj = self.operator._adj_op_calibless_device(ksp_batched)
            tmp_adj /= self.norm_factor
            tmp_adj = tmp_adj.reshape((T, NZ, *XYZ[:2]))
            tmp_adj = cp.moveaxis(tmp_adj, 1, -1)
            coil_img_d[:] = 0j
            coil_img_d[..., self.z_index] = tmp_adj
            coil_img_d = self._ifftz(coil_img_d)
            img[i * T : (i + 1) * T, ...] = coil_img_d.get()
        img = img.reshape(B, C, *XYZ)
        return img

    def _adj_op_calibless_device(self, coeffs, img):
        B, C, T, XYZ = self.n_batchs, self.n_coils, self.n_trans, self.shape
        NS, NZ = len(self._samples2d), len(self.z_index)
        coeffs = cp.asarray(coeffs)
        coeffs_f = coeffs.reshape(B, C, NZ * NS)
        coeffs_f = coeffs_f.reshape(B * C, NZ, NS)
        coeffs_f = coeffs_f.reshape(B * C * NZ, NS)
        # Allocate Memory
        ksp_batched = cp.empty((T, NZ * NS), dtype=self.cpx_dtype)
        if img is None:
            img = cp.zeros((B * C, *XYZ), dtype=self.cpx_dtype)
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        TZ = T * NZ
        for i in range((B * C * NZ) // TZ):
            ksp_batched = coeffs_f[i * TZ : (i + 1) * TZ]
            ksp_batched = ksp_batched.reshape(TZ, NS)
            tmp_adj = self.operator._adj_op_calibless_device(ksp_batched)
            tmp_adj /= self.norm_factor
            tmp_adj = tmp_adj.reshape((T, NZ, *XYZ[:2]))
            tmp_adj = cp.moveaxis(tmp_adj, 1, -1)
            coil_img_d[:] = 0j
            coil_img_d[..., self.z_index] = tmp_adj
            coil_img_d = self._ifftz(coil_img_d)
            img[i * T : (i + 1) * T, ...] = coil_img_d
        img = img.reshape(B, C, *XYZ)
        return img

    def get_lipschitz_cst(self, max_iter=10, **kwargs):
        """Return the Lipschitz constant of the operator.

        Parameters
        ----------
        max_iter: int
            Number of iteration to perform to estimate the Lipschitz constant.
        kwargs:
            Extra kwargs for the cufinufft operator.

        Returns
        -------
        float
            Lipschitz constant of the operator.
        """
        # The fourier transform is orthonormal, so it's lipschizt constant is 1.
        # We only compute the lipschitz constant of the 2d underlying nufft.
        #
        tmp_op = self.operator.__class__(
            self.operator.samples,
            self.operator.shape,
            density=self.operator.density,
            smaps=None,
            n_coils=1,
            squeeze_dims=True,
        )

        x = 1j * np.random.random(self.operator.shape).astype(self.cpx_dtype)
        x += np.random.random(self.operator.shape).astype(self.cpx_dtype)

        x = cp.asarray(x)
        return power_method(
            max_iter, tmp_op, norm_func=lambda x: cp.linalg.norm(x.flatten()), x=x
        )


def traj3d2stacked(samples, dim_z, n_samples=0):
    """Convert a 3D trajectory into a trajectory and the z-stack index.

    Parameters
    ----------
    samples: array-like
        3D trajectory
    dim_z: int
        Size of the z dimension
    n_samples: int, default=0
        Number of samples per shot. If 0, the shot length is determined by counting the
        unique z values.

    Returns
    -------
    tuple
        2D trajectory, z_index

    """
    samples = np.asarray(samples).reshape(-1, 3)
    z_kspace, idx = np.unique(samples[:, 2], return_index=True)
    z_kspace = z_kspace[np.argsort(idx)]
    if n_samples == 0:
        n_samples = np.prod(samples.shape[:-1]) // len(z_kspace)
    traj2D = samples[:n_samples, :2]

    z_kspace = proper_trajectory(z_kspace, "unit").flatten()
    z_index = np.int32(z_kspace * dim_z + dim_z // 2)

    return traj2D, z_index


def stacked2traj3d(samples2d, z_indexes, dim_z):
    """Convert a 2D trajectory and list of z_index into a 3D trajectory.

    Note that the trajectory is flatten in the process.

    Parameters
    ----------
    samples2d: array-like
        2D trajectory
    z_indexes: array-like
        List of z_index
    dim_z: int
        Size of the z dimension

    Returns
    -------
    samples3d: array-like
        3D trajectory
    """
    z_kspace = (z_indexes - dim_z // 2) / dim_z
    # create the equivalent 3d trajectory
    kspace_locs_proper = proper_trajectory(samples2d, normalize="unit")
    nsamples = len(kspace_locs_proper)
    nz = len(z_kspace)
    kspace_locs3d = np.zeros((nz, nsamples, 3), dtype=samples2d.dtype)
    # TODO use numpy api for this ?
    for i in range(nz):
        kspace_locs3d[i, :, :2] = kspace_locs_proper
        kspace_locs3d[i, :, 2] = z_kspace[i]

    return kspace_locs3d
