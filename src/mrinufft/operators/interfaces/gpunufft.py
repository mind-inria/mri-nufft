"""Interface to the GPU NUFFT library."""

import numpy as np
from ..base import FourierOperatorBase, proper_trajectory

GPUNUFFT_AVAILABLE = True
try:
    from gpuNUFFT import NUFFTOp
except ImportError:
    GPUNUFFT_AVAILABLE = False


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
        self.samples = proper_trajectory(samples, normalize="unit")
        if density_comp is None:
            density_comp = np.ones(samples.shape[0])

        self.uses_sense = True
        if smaps is not None and pinned_smaps is None:
            # no pinning provided, we will pin it in the C++ code
            pinned_smaps = smaps.T.reshape(-1, n_coils)
        elif smaps is not None and pinned_smaps is not None:
            # Pinned memory space exists, we will overwrite it
            np.copyto(pinned_smaps, smaps.T.reshape(-1, n_coils))
        else:
            # No smaps provided, we will not use SENSE
            self.uses_sense = False

        if upsampfac is not None:
            osf = upsampfac

        self.operator = NUFFTOp(
            np.reshape(samples, samples.shape[::-1], order="F"),
            shape,
            n_coils,
            pinned_smaps,
            density_comp,
            kernel_width,
            sector_width,
            osf,
            balance_workload,
        )

    def op(self, image, interpolate_data=False):
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
            if not self.uses_sense:
                coeff = coeff[0]
        return coeff

    def adj_op(self, coeff, grid_data=False):
        """Compute adjoint of non-uniform Fourier transform.

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
        if self.n_coils > 1 and not self.uses_sense:
            image = np.asarray([image_ch.T for image_ch in image])
        else:
            image = np.squeeze(image).T
        # The recieved data from gpuNUFFT is num_channels x Nx x Ny x Nz,
        # hence we use squeeze
        return np.squeeze(image)


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
    available = GPUNUFFT_AVAILABLE

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
        if density is True:
            self.density = pipe(self.samples, shape)
        elif isinstance(density, np.ndarray):
            self.density = density
        else:
            self.density = None
        self.kwargs = kwargs
        self.impl = RawGpuNUFFT(
            samples=self.samples,
            shape=self.shape,
            n_coils=self.n_coils,
            density_comp=self.density,
            smaps=smaps,
            kernel_width=kwargs.get("kernel_width", -int(np.log10(eps))),
            **self.kwargs,
        )

    def op(self, data, *args):
        """Compute forward non-uniform Fourier Transform.

        Parameters
        ----------
        img: np.ndarray
            input N-D array with the same shape as self.shape.

        Returns
        -------
        np.ndarray
            Masked Fourier transform of the input image.
        """
        return self.impl.op(data, *args)

    def adj_op(self, coeffs, *args):
        """Compute adjoint Non Unform Fourier Transform.

        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
        np.ndarray
            Inverse discrete Fourier transform of the input coefficients.
        """
        return self.impl.adj_op(coeffs, *args)

    @property
    def uses_sense(self):
        """Return True if the Fourier Operator uses the SENSE method."""
        return self.impl.uses_sense


def pipe(kspace_loc, volume_shape, num_iterations=10):
    """Compute the density compensation weights for a given set of kspace locations.

    Parameters
    ----------
    kspace_loc: np.ndarray
        the kspace locations
    volume_shape: np.ndarray
        the volume shape
    num_iterations: int default 10
        the number of iterations for density estimation
    """
    if GPUNUFFT_AVAILABLE is False:
        raise ValueError(
            "gpuNUFFT is not available, cannot " "estimate the density compensation"
        )
    grid_op = MRIGpuNUFFT(
        samples=kspace_loc,
        shape=volume_shape,
        osf=1,
    )
    density_comp = np.ones(kspace_loc.shape[0])
    for _ in range(num_iterations):
        density_comp = density_comp / np.abs(
            grid_op.op(grid_op.adj_op(density_comp, True), True)
        )
    return density_comp
