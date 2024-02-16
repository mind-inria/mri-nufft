"""Interface to the GPU NUFFT library."""

import numpy as np
import warnings
from ..base import FourierOperatorBase, with_numpy_cupy
from mrinufft._utils import proper_trajectory, get_array_module
from mrinufft.operators.interfaces.utils import is_cuda_array

GPUNUFFT_AVAILABLE = True
try:
    from gpuNUFFT import NUFFTOp
except ImportError:
    GPUNUFFT_AVAILABLE = False

CUPY_AVAILABLE = True
try:
    import cupyx as cx
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False


def _allocator(size):
    """Allocate pinned memory which is context portable."""
    flags = cp.cuda.runtime.hostAllocPortable
    mem = cp.cuda.PinnedMemory(size, flags=flags)
    return cp.cuda.PinnedMemoryPointer(mem, offset=0)


def make_pinned_smaps(smaps):
    """Make pinned smaps from smaps.

    Parameters
    ----------
    smaps: np.ndarray or None
        the sensitivity maps

    Returns
    -------
    np.ndarray or None
        the pinned sensitivity maps
    """
    if smaps is None:
        return None
    smaps_ = smaps.T.reshape(-1, smaps.shape[0])
    cp.cuda.set_pinned_memory_allocator(_allocator)
    pinned_smaps = cx.empty_pinned(smaps_.shape, dtype=np.complex64, order="F")
    np.copyto(pinned_smaps, smaps_)
    return pinned_smaps


class RawGpuNUFFT:
    """GPU implementation of N-D non-uniform fast Fourier Transform class.

    Attributes
    ----------
    samples: np.ndarray
        the normalized kspace location values in the Fourier domain.
    shape: tuple of int
        shape of the image
    operator: The NUFFTOp object
        to carry out operation
    n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition. If n_coils > 1, please organize data as
            n_coils X data_per_coil
    """

    def __init__(
        self,
        samples,
        shape,
        n_coils=1,
        density_comp=None,
        kernel_width=3,
        sector_width=8,
        osf=2,
        upsampfac=None,
        balance_workload=True,
        smaps=None,
        pinned_smaps=None,
        pinned_image=None,
        pinned_kspace=None,
        use_gpu_direct=False,
    ):
        """Initialize the 'NUFFT' class.

        Parameters
        ----------
        samples: np.ndarray
            the kspace sample locations in the Fourier domain,
            normalized between -0.5 and 0.5
        shape: tuple of int
            shape of the image
        n_coils: int
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        density_comp: np.ndarray default None.
            k-space weighting, density compensation, if not specified
            equal weightage is given.
        kernel_width: int default 3
            interpolation kernel width (usually 3 to 7)
        sector_width: int default 8
            sector width to use
        osf: int default 2
            oversampling factor (usually between 1 and 2)
        upsampfac: int default 2
            Same as osf.
        balance_workload: bool default True
            whether the workloads need to be balanced
        smaps: np.ndarray default None
            Holds the sensitivity maps for SENSE reconstruction
        pinned_smaps: np.ndarray default None
            Pinned memory array for the smaps.
        use_gpu_direct: bool default False
            if True, direct GPU array can be passed.
            In this case pinned memory is not used and this saved memory.
            It will not be an error if this is False and you pass GPU array,
            just that it is inefficient.

        Notes
        -----
        pinned_smaps status (pinned or not) is not checked here, but in the C++ code.
        If its not pinned, then an extra copy will be triggered.
        """
        if GPUNUFFT_AVAILABLE is False:
            raise ValueError(
                "gpuNUFFT library is not installed, please refer to README"
            )
        if (n_coils < 1) or not isinstance(n_coils, int):
            raise ValueError("The number of coils should be an integer >= 1")
        self.n_coils = n_coils
        self.shape = shape
        self.samples = samples
        self.use_gpu_direct = use_gpu_direct
        if density_comp is None:
            density_comp = np.ones(samples.shape[0])

        if upsampfac is not None:
            osf = upsampfac

        # pinned memory stuff
        self.uses_sense = True
        if smaps is not None and pinned_smaps is None:
            pinned_smaps = make_pinned_smaps(smaps)
            warnings.warn("no pinning provided, pinning existing smaps now.")
        elif smaps is not None and pinned_smaps is not None:
            # Pinned memory space exists, we will overwrite it
            np.copyto(pinned_smaps, smaps.T.reshape(-1, n_coils))
            warnings.warn("Overwriting the pinned data.")
        elif smaps is None and pinned_smaps is None:
            # No smaps provided, we will not use SENSE
            self.uses_sense = False
        elif smaps is None and pinned_smaps is not None:
            warnings.warn("Using pinned_smaps as is.")
        else:
            raise ValueError("Unknown case")
        if not use_gpu_direct:
            # We dont need pinned allocations if we are using direct GPU arrays
            if pinned_image is None:
                pinned_image = cx.empty_pinned(
                    (np.prod(shape), (1 if self.uses_sense else n_coils)),
                    dtype=np.complex64,
                    order="F",
                )
            if pinned_kspace is None:
                pinned_kspace = cx.empty_pinned(
                    (n_coils, len(samples)),
                    dtype=np.complex64,
                )
        self.pinned_image = pinned_image
        self.pinned_kspace = pinned_kspace

        self.pinned_smaps = pinned_smaps
        self.operator = NUFFTOp(
            np.reshape(samples, samples.shape[::-1], order="F"),
            self.shape,
            self.n_coils,
            self.pinned_smaps,
            density_comp,
            kernel_width,
            sector_width,
            osf,
            balance_workload,
        )

    def _reshape_image(self, image, direction="op"):
        """Reshape the image to the correct format."""
        xp = get_array_module(image)
        if direction == "op":
            if self.uses_sense or self.n_coils == 1:
                return image.reshape((-1, 1), order="F").astype(xp.complex64)
            return xp.asarray([c.ravel(order="F") for c in image], dtype=xp.complex64).T
        else:
            if self.uses_sense or self.n_coils == 1:
                return image.squeeze().astype(xp.complex64).T
            return xp.asarray([c.T for c in image], dtype=xp.complex64)

    def op_direct(self, image, kspace=None, interpolate_data=False):
        """Compute the masked non-Cartesian Fourier transform.

        The incoming data is on GPU already and we return a GPU array.

        Parameters
        ----------
        image: np.ndarray
            input array with the same shape as self.shape.
        interpolate_data: bool, default False
            if set to True, the image is just apodized and interpolated to
            kspace locations. This is used for density estimation.

        Returns
        -------
        cp.ndarray
            Non-uniform Fourier transform of the input image.
        """
        if kspace is None:
            kspace = cp.empty(
                (self.n_coils, len(self.samples)),
                dtype=cp.complex64,
            )
        reshape_image = self._reshape_image(image)
        self.operator.op_direct(
            reshape_image.data.ptr,
            kspace.data.ptr,
            interpolate_data,
        )
        return kspace

    def op(self, image, kspace=None, interpolate_data=False):
        """Compute the masked non-Cartesian Fourier transform.

        Parameters
        ----------
        image: np.ndarray
            input array with the same shape as self.shape.
        interpolate_data: bool, default False
            if set to True, the image is just apodized and interpolated to
            kspace locations. This is used for density estimation.

        Returns
        -------
        np.ndarray
            Non-uniform Fourier transform of the input image.
        """
        # Base gpuNUFFT Operator is written in CUDA and C++, we need to
        # reorganize data to follow a different memory hierarchy
        # TODO we need to update codes to use np.reshape for all this directly
        make_copy_back = False
        if kspace is None:
            kspace = self.pinned_kspace
            make_copy_back = True
        np.copyto(self.pinned_image, self._reshape_image(image))
        new_ksp = self.operator.op(
            self.pinned_image,
            kspace,
            interpolate_data,
        )
        if make_copy_back:
            new_ksp = cp.copy(new_ksp)
        return new_ksp

    def adj_op(self, coeffs, image=None, grid_data=False):
        """Compute adjoint of non-uniform Fourier transform.

        Parameters
        ----------
        coeff: np.ndarray
            masked non-uniform Fourier transform data.
        grid_data: bool, default False
            if True, the kspace data is gridded and returned,
            this is used for density compensation

        Returns
        -------
        np.ndarray
            adjoint operator of Non Uniform Fourier transform of the
            input coefficients.
        """
        make_copy_back = False
        if image is None:
            image = self.pinned_image
            make_copy_back = True
        np.copyto(self.pinned_kspace, coeffs)
        new_image = self.operator.adj_op(self.pinned_kspace, image, grid_data)
        if make_copy_back:
            new_image = np.copy(new_image)
        return self._reshape_image(new_image, "adjoint")

    def adj_op_direct(self, coeffs, image=None, grid_data=False):
        """Compute adjoint of non-uniform Fourier transform.

        Parameters
        ----------
        coeff: np.ndarray
            masked non-uniform Fourier transform data.
        grid_data: bool, default False
            if True, the kspace data is gridded and returned,
            this is used for density compensation

        Returns
        -------
        np.ndarray
            adjoint operator of Non Uniform Fourier transform of the
            input coefficients.
        """
        coeffs = coeffs.astype(cp.complex64)
        if image is None:
            image = cp.empty(
                (np.prod(self.shape), (1 if self.uses_sense else self.n_coils)),
                dtype=cp.complex64,
                order="F",
            )
        self.operator.adj_op_direct(coeffs.data.ptr, image.data.ptr, grid_data)
        image = image.reshape(self.n_coils, *self.shape[::-1])
        return self._reshape_image(image, "adjoint")


class MRIGpuNUFFT(FourierOperatorBase):
    """Interface for the gpuNUFFT backend.

    Parameters
    ----------
    samples: np.ndarray (Mxd)
        the samples locations in the Fourier domain where M is the number
        of samples and d is the dimensionnality of the output data
        (2D for an image, 3D for a volume).
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    n_coils: int default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition
    density: bool or np.ndarray default None
        if True, the density compensation is estimated from the samples
        locations. If an array is passed, it is used as the density
        compensation.
    squeeze_dims: bool default True
        This has no effect, gpuNUFFT always squeeze the data.
    smaps: np.ndarray default None
        Holds the sensitivity maps for SENSE reconstruction.
    kwargs: extra keyword args
        these arguments are passed to gpuNUFFT operator. This is used
        only in gpuNUFFT
    """

    backend = "gpunufft"
    available = GPUNUFFT_AVAILABLE and CUPY_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        n_coils=1,
        density=None,
        smaps=None,
        squeeze_dims=False,
        eps=1e-3,
        **kwargs,
    ):
        if GPUNUFFT_AVAILABLE is False:
            raise ValueError(
                "gpuNUFFT library is not installed, "
                "please refer to README"
                "or use cpu for implementation"
            )
        self.shape = shape
        self.samples = proper_trajectory(samples, normalize="unit")
        self.dtype = self.samples.dtype
        self.n_coils = n_coils
        self.smaps = smaps

        self.compute_density(density)
        self.impl = RawGpuNUFFT(
            samples=self.samples,
            shape=self.shape,
            n_coils=self.n_coils,
            density_comp=self.density,
            smaps=smaps,
            kernel_width=kwargs.get("kernel_width", -int(np.log10(eps))),
            **kwargs,
        )

    @with_numpy_cupy
    def op(self, data, coeffs=None):
        """Compute forward non-uniform Fourier Transform.

        Parameters
        ----------
        img: np.ndarray
            input N-D array with the same shape as self.shape.
        coeffs: np.ndarray, optional
            output Array. Should be pinned memory for best performances.

        Returns
        -------
        np.ndarray
            Masked Fourier transform of the input image.
        """
        if is_cuda_array(data):
            if not self.impl.use_gpu_direct:
                warnings.warn(
                    "Using direct GPU array without passing "
                    "`use_gpu_direct=True`, this is memory inefficient."
                )
            return self.impl.op_direct(data, coeffs)
        return self.impl.op(
            data,
            coeffs,
        )

    @with_numpy_cupy
    def adj_op(self, coeffs, data=None):
        """Compute adjoint Non Unform Fourier Transform.

        Parameters
        ----------
        coeffs: np.ndarray
            masked non-uniform Fourier transform 1D data.
        data: np.ndarray, optional
            output image array. Should be pinned memory for best performances.

        Returns
        -------
        np.ndarray
            Inverse discrete Fourier transform of the input coefficients.
        """
        if is_cuda_array(coeffs):
            if not self.impl.use_gpu_direct:
                warnings.warn(
                    "Using direct GPU array without passing "
                    "`use_gpu_direct=True`, this is memory inefficient."
                )
            return self.impl.adj_op_direct(coeffs, data)
        return self.impl.adj_op(coeffs, data)

    @property
    def uses_sense(self):
        """Return True if the Fourier Operator uses the SENSE method."""
        return self.impl.uses_sense

    @classmethod
    def pipe(cls, kspace_loc, volume_shape, num_iterations=10, osf=2, **kwargs):
        """Compute the density compensation weights for a given set of kspace locations.

        Parameters
        ----------
        kspace_loc: np.ndarray
            the kspace locations
        volume_shape: np.ndarray
            the volume shape
        num_iterations: int default 10
            the number of iterations for density estimation
        osf: float or int
            The oversampling factor the volume shape
        """
        if GPUNUFFT_AVAILABLE is False:
            raise ValueError(
                "gpuNUFFT is not available, cannot " "estimate the density compensation"
            )
        grid_op = MRIGpuNUFFT(
            samples=kspace_loc,
            shape=volume_shape,
            osf=osf,
            **kwargs,
        )
        density_comp = grid_op.impl.operator.estimate_density_comp(
            max_iter=num_iterations
        )
        return density_comp.squeeze()

    def get_lipschitz_cst(self, max_iter=10, tolerance=1e-5, **kwargs):
        """Return the Lipschitz constant of the operator.

        Parameters
        ----------
        max_iter: int
            Number of iteration to perform to estimate the Lipschitz constant.
        tolerance: float, optional default 1e-5
            Tolerance for the spectral radius estimation.
        kwargs:
            Extra kwargs for the operator.

        Returns
        -------
        float
            Lipschitz constant of the operator.
        """
        tmp_op = self.__class__(
            self.samples,
            self.shape,
            density=self.density,
            n_coils=1,
            smaps=None,
            squeeze_dims=True,
            **kwargs,
        )
        return tmp_op.impl.operator.get_spectral_radius(
            max_iter=max_iter, tolerance=tolerance
        )
