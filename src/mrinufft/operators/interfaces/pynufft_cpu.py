"""PyNUFFT CPU Interface."""

from ..base import FourierOperatorCPU


PYNUFFT_CPU_AVAILABLE = True
try:
    import pynufft
except ImportError:
    PYNUFFT_CPU_AVAILABLE = False


class RawPyNUFFT:
    """Wrapper around PyNUFFT object."""

    def __init__(self, samples, shape, osf=2, interpolator_shape=6):
        self._nufft_obj = pynufft.NUFFT()
        self._nufft_obj.plan(
            samples,
            shape,
            tuple(osf * s for s in shape),
            tuple([interpolator_shape] * len(shape)),
        )

    def op(self, coeffs_data, grid_data):
        """Forward Operator."""
        coeffs_data = self._nufft_obj.forward(grid_data)
        return coeffs_data

    def adj_op(self, coeffs_data, grid_data):
        """Adjoint Operator."""
        grid_data = self._nufft_obj.backward(coeffs_data)
        return grid_data


class MRIPynufft(FourierOperatorCPU):
    """PyNUFFT implementation of MRI NUFFT transform."""

    backend = "pynufft-cpu"
    available = PYNUFFT_CPU_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        smaps=None,
        osf=2,
        **kwargs,
    ):
        super().__init__(samples, shape, density, n_coils, smaps)

        self.raw_op = RawPyNUFFT(samples, shape, osf, **kwargs)
