"""Interface to the GPU NUFFT library."""

import numpy as np
from .base import FourierOperatorBase, proper_trajectory

GPUNUFFT_AVAILABLE = True
try:
    from gpuNUFFT import NUFFTOp
except ImportError:
    GPUNUFFT_AVAILABLE = False


class RawGpuNUFFT:
    """GPU implementation of N-D non uniform Fast Fourrier Transform class.

    Original implementation in PySAP-MRI: https://github.com/CEA-COSMIC/pysap-mri
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
        balance_workload=True,
        smaps=None,
    ):
        """Initilize the 'NUFFT' class.

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
        balance_workload: bool default True
            whether the workloads need to be balanced
        smaps: np.ndarray default None
            Holds the sensitivity maps for SENSE reconstruction
        """
        self.n_coils = n_coils
        self.shape = shape

        self.samples = proper_trajectory(samples, normalize="unit")

        if density_comp is None:
            density_comp = np.ones(samples.shape[0])
        if smaps is None:
            self.uses_sense = False
        else:
            smaps = np.asarray(
                [np.reshape(smap_ch.T, smap_ch.size) for smap_ch in smaps]
            ).T
            self.uses_sense = True

        self.operator = NUFFTOp(
            np.reshape(samples, samples.shape[::-1], order="F"),
            shape,
            n_coils,
            smaps,
            density_comp,
            kernel_width,
            sector_width,
            osf,
            balance_workload,
        )

    def op(self, image, interpolate_data=False):
        """This method calculates the masked non-cartesian Fourier transform
        of a 2D / 3D image.

        Parameters
        ----------
        image: np.ndarray
            input array with the same shape as shape.
        interpolate_data: bool, default False
            if set to True, the image is just apodized and interpolated to
            kspace locations. This is used for density estimation.

        Returns
        -------
        np.ndarray
            Non Uniform Fourier transform of the input image.
        """
        # Base gpuNUFFT Operator is written in CUDA and C++, we need to
        # reorganize data to follow a different memory hierarchy
        # TODO we need to update codes to use np.reshape for all this directly
        if self.n_coils > 1 and not self.uses_sense:
            coeff = self.operator.op(
                np.asarray(
                    [np.reshape(image_ch.T, image_ch.size) for image_ch in image]
                ).T,
                interpolate_data,
            )
        else:
            coeff = self.operator.op(np.reshape(image.T, image.size), interpolate_data)
            # Data is always returned as num_channels X coeff_array,
            # so for single channel, we extract single array
        return coeff

    def adj_op(self, coeff, grid_data=False):
        """This method calculates adjoint of non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        coeff: np.ndarray
            masked non-uniform Fourier transform 1D data.
        grid_data: bool, default False
            if True, the kspace data is gridded and returned,
            this is used for density compensation
        Returns
        -------
        np.ndarray
            adjoint operator of Non Uniform Fourier transform of the
            input coefficients.
        """
        image = self.operator.adj_op(coeff, grid_data)

        return np.transpose(image, axes=(0, *range(1, image.ndim)[::-1]))


class MRIGpuNUFFT(FourierOperatorBase):
    """Create a GPU NUFFT operator.

    Parameters
    ----------
    samples : np.ndarray
        Samples locations in k-space.
    shape : tuple
        Shape of the image.
    density : np.ndarray, optional
        Density compensation, by default False
    n_coils : int, optional
        Number of coils, by default 1
    n_batch : int, optional
        Number of batch, by default 1
    smaps : np.ndarray, optional
        Sensitivity maps, by default None
    """

    backend = "gpunufft"

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batch=1,
        smaps=None,
        squeeze_dims=True,
        **kwargs,
    ):
        if not GPUNUFFT_AVAILABLE:
            raise ImportError("gpuNUFFT is not available. Please install it from ")
        super().__init__()
        self.samples = proper_trajectory(samples, normalize="unit")
        self.shape = shape
        self.n_coils = n_coils
        self.n_batch = n_batch
        self.smaps = smaps
        self.squeeze_dims = squeeze_dims
        self.dtype = self.samples.dtype
        # density compensation support
        if density is True:
            self.density = pipe(self.samples, shape)

        elif isinstance(density, np.ndarray):
            self.density = density
        else:
            self.density = None

        self.plan = RawGpuNUFFT(
            samples=self.samples,
            shape=self.shape,
            n_coils=self.n_coils,
            density_comp=self.density,
            smaps=self.smaps,
            **kwargs,
        )

    def op(self, x):
        r"""Non Cartesian MRI forward operator.

        Parameters
        ----------
        data: np.ndarray
            The uniform (2D or 3D) data in image space.

        Returns
        -------
        Results array

        Notes
        -----
        this performs for every coil \ell:
        ..math:: \mathcal{F}\mathcal{S}_\ell x
        """
        return self._safe_squeeze(self.plan.op(x))

    def adj_op(self, x):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        return self._safe_squeeze(self.plan.adj_op(x))

    def _safe_squeeze(self, arr):
        """Squeeze the shape of the operator."""
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

    # TODO Add batch support
    def data_consistency(self, data, obs_data):
        """Data consistency operator.

        Parameters
        ----------
        data: np.ndarray
            The uniform (2D or 3D) data in image space.
        obs_data: np.ndarray
            The non uniform (2D or 3D) data in k-space.

        Returns
        -------
        Results array

        """
        return self.plan.adj_op(self.plan.op(data) - obs_data)


def pipe(kspace_loc, shape, num_iterations):
    """Estimate density compensation weight using the Pipe method.

    Parameters
    ----------
    kspace: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    shape: tuple
        shape of the image grid.
    num_iter: int, optional
        number of iterations.

    Returns
    -------
    density: array_like
        array of shape (M,) containing the density compensation weights.
    """
    kspace_loc = proper_trajectory(kspace_loc, normalize="unit")
    grid_op = RawGpuNUFFT(kspace_loc, shape, n_coils=1, density_comp=None, smaps=None)

    density_comp = np.ones(kspace_loc.shape[0])
    for _ in range(num_iterations):
        density_comp = density_comp / np.abs(
            grid_op.op(grid_op.adj_op(density_comp, True), True)
        )
    return density_comp
