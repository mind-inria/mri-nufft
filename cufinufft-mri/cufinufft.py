"""This Module provides a wrapper around the python bindings of cufinufft."""
import pycuda
import cufinufft



class CufiNUFFT:
    """GPU implementation of N-D non uniform Fast Fourrier Transform class.

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

    def __init__(self, samples, shape, n_coils=1, smaps=None):
        if CUFINUFFT_AVAILABLE is False:
            raise ValueError(
                "cufinufft library is not installed, " "please refer to README"
            )

        import pycuda.autoinit
        if (n_coils < 1) or not isinstance(n_coils, int):
            raise ValueError("The number of coils should be an integer >= 1")
        self.n_coils = n_coils
        self.shape = shape
        if samples.min() < -0.5 or samples.max() >= 0.5:
            warnings.warn("Samples will be normalized between [-0.5; 0.5[")
            samples = normalize_frequency_locations(samples)
            samples = samples * 2.0 * np.pi
            samples = np.float32(samples)
        if smaps is None:
            self.uses_sense = False
        else:
            self.uses_sense = True
            self.smaps = smaps

        samples_x = to_gpu(samples[:, 0])
        samples_y = to_gpu(samples[:, 1])
        if len(shape) == 2:
            self.plan_op = cufinufft(
                2, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_op.set_pts(samples_x, samples_y)

            self.plan_adj_op = cufinufft(
                1, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_adj_op.set_pts(samples_x, samples_y)
        elif len(shape) == 3:
            self.plan_op = cufinufft(
                2, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_op.set_pts(samples_x, samples_y, to_gpu(samples[:, 2]))

            self.plan_adj_op = cufinufft(
                1, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_adj_op.set_pts(
                samples_x, samples_y, to_gpu(samples[:, 2]))

        else:
            raise ValueError("Unsupported number of dimension. ")

        self.kspace_data_gpu = GPUArray(
            (self.n_coils, len(samples)), dtype=np.complex64
        )
        self.image_gpu_multicoil = GPUArray(
            (self.n_coils, *self.shape), dtype=np.complex64
        )

    def op(self, image):
        """This method calculates the masked non-cartesian Fourier transform.

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
        if self.uses_sense:
            image_mc = np.complex64(image * self.smaps)
            self.image_gpu_multicoil.set(image_mc)
        else:
            self.image_gpu_multicoil.set(image)

        self.plan_op.execute(self.kspace_data_gpu, self.image_gpu_multicoil)

        return self.kspace_data_gpu.get()

    def adj_op(self, coeff):
        """Compute adjoint of non-uniform Fourier.

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
        image: np.ndarray
            adjoint operator of Non Uniform Fourier transform.
        """
        self.plan_adj_op.execute(to_gpu(coeff), self.image_gpu_multicoil)
        image_mc = self.image_gpu_multicoil.get()
        if self.uses_sense:
            # TODO: Do this on device.
            return np.sum(np.conjugate(self.smaps) * image_mc, axis=0)
        return image_mc
