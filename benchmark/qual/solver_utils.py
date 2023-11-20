import numpy as np
from modopt.opt.algorithms import POGM, ForwardBackward, Condat
from modopt.opt.linear import Identity
from mri.operators.gradient.gradient import GradAnalysis, GradSynthesis

OPTIMIZERS = {
    "pogm": "synthesis",
    "fista": "analysis",
    "condat-vu": "analysis",
    None: None,
}


def get_grad_op(fourier_op, grad_formulation, linear_op=None, verbose=False, **kwargs):
    """Create gradient operator specific to the problem."""
    if grad_formulation == "analysis":
        return GradAnalysis(fourier_op=fourier_op, verbose=verbose, **kwargs)
    if grad_formulation == "synthesis":
        return GradSynthesis(
            linear_op=linear_op,
            fourier_op=fourier_op,
            verbose=verbose,
            **kwargs,
        )


def initialize_opt(
    opt_name,
    grad_op,
    linear_op,
    prox_op,
    x_init=None,
    synthesis_init=False,
    opt_kwargs=None,
    metric_kwargs=None,
):
    """
    Initialize an Optimizer with the suitable parameters.

    Parameters:
    ----------
    grad_op: OperatorBase
        Gradient Operator for the data consistency
    x_init: ndarray, default None
        Initial value for the reconstruction. If None use a zero Array.
    synthesis_init: bool, default False
        Is the initial_value in the image space of the space_linear operator ?
    opt_kwargs: dict, default None
        Extra kwargs for the initialisation of Optimizer
    metric_kwargs: dict, default None
        Extra kwargs for the metric api of ModOpt

    Returns:
    -------
    An Optimizer Instance to perform the reconstruction with.
    See Also:
    --------
    Modopt.opt.algorithms

    """
    if x_init is None:
        x_init = np.squeeze(
            np.zeros(
                (grad_op.fourier_op.n_coils, *grad_op.fourier_op.shape),
                dtype="complex64",
            )
        )

    if not synthesis_init and hasattr(grad_op, "linear_op"):
        alpha_init = grad_op.linear_op.op(x_init)
    elif synthesis_init and not hasattr(grad_op, "linear_op"):
        x_init = linear_op.adj_op(x_init)
    elif not synthesis_init and hasattr(grad_op, "linear_op"):
        alpha_init = x_init
    opt_kwargs = opt_kwargs or dict()
    metric_kwargs = metric_kwargs or dict()

    beta = grad_op.inv_spec_rad
    if opt_name == "pogm":
        opt = POGM(
            u=alpha_init,
            x=alpha_init,
            y=alpha_init,
            z=alpha_init,
            grad=grad_op,
            prox=prox_op,
            linear=linear_op,
            beta_param=beta,
            sigma_bar=opt_kwargs.pop("sigma_bar", 0.96),
            auto_iterate=opt_kwargs.pop("auto_iterate", False),
            **opt_kwargs,
            **metric_kwargs,
        )
    elif opt_name == "fista":
        opt = ForwardBackward(
            x=x_init,
            grad=grad_op,
            prox=prox_op,
            linear=linear_op,
            beta_param=beta,
            lambda_param=opt_kwargs.pop("lambda_param", 1.0),
            auto_iterate=opt_kwargs.pop("auto_iterate", False),
            **opt_kwargs,
            **metric_kwargs,
        )
    elif opt_name == "condat-vu":
        y_init = linear_op.op(x_init)

        opt = Condat(
            x=x_init,
            y=y_init,
            grad=grad_op,
            prox=Identity(),
            prox_dual=prox_op,
            linear=linear_op,
            **opt_kwargs,
            **metric_kwargs,
        )

    else:
        raise ValueError(f"Optimizer {opt_name} not implemented")
    return opt


from modopt.opt.linear import LinearParent
import pywt
from joblib import Parallel, delayed, cpu_count
import numpy as np


class WaveletTransform(LinearParent):
    """
    2D and 3D wavelet transform class.

    This is a light wrapper around PyWavelet, with multicoil support.

    Parameters
    ----------
    wavelet_name: str
        the wavelet name to be used during the decomposition.
    shape: tuple[int,...]
        Shape of the input data. The shape should be a tuple of length 2 or 3.
        It should not contains coils or batch dimension.
    nb_scales: int, default 4
        the number of scales in the decomposition.
    n_coils: int, default 1
        the number of coils for multichannel reconstruction
    n_jobs: int, default 1
        the number of cores to use for multichannel.
    backend: str, default "threading"
        the backend to use for parallel multichannel linear operation.
    verbose: int, default 0
        the verbosity level.

    Attributes
    ----------
    nb_scale: int
        number of scale decomposed in wavelet space.
    n_jobs: int
        number of jobs for parallel computation
    n_coils: int
        number of coils use f
    backend: str
        Backend use for parallel computation
    verbose: int
        Verbosity level
    """

    def __init__(
        self,
        wavelet_name,
        shape,
        level=4,
        n_coils=1,
        n_jobs=1,
        decimated=True,
        backend="threading",
        mode="symmetric",
    ):
        if wavelet_name not in pywt.wavelist(kind="all"):
            raise ValueError(
                "Invalid wavelet name. Check ``pywt.waveletlist(kind='all')``"
            )

        self.wavelet = wavelet_name
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_jobs = n_jobs
        self.mode = mode
        self.level = level
        if not decimated:
            raise NotImplementedError(
                "Undecimated Wavelet Transform is not implemented yet."
            )
        ca, *cds = pywt.wavedecn_shapes(
            self.shape, wavelet=self.wavelet, mode=self.mode, level=self.level
        )
        self.coeffs_shape = [ca] + [s for cd in cds for s in cd.values()]

        if len(shape) > 1:
            self.dwt = pywt.wavedecn
            self.idwt = pywt.waverecn
            self._pywt_fun = "wavedecn"
        else:
            self.dwt = pywt.wavedec
            self.idwt = pywt.waverec
            self._pywt_fun = "wavedec"

        self.n_coils = n_coils
        if self.n_coils == 1 and self.n_jobs != 1:
            print("Making n_jobs = 1 for WaveletN as n_coils = 1")
            self.n_jobs = 1
        self.backend = backend
        n_proc = self.n_jobs
        if n_proc < 0:
            n_proc = cpu_count() + self.n_jobs + 1

    def op(self, data):
        """Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        if self.n_coils > 1:
            coeffs, self.coeffs_slices, self.raw_coeffs_shape = zip(
                *Parallel(
                    n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
                )(delayed(self._op)(data[i]) for i in np.arange(self.n_coils))
            )
            coeffs = np.asarray(coeffs)
        else:
            coeffs, self.coeffs_slices, self.raw_coeffs_shape = self._op(data)
        return coeffs

    def _op(self, data):
        """Single coil wavelet transform."""
        return pywt.ravel_coeffs(
            self.dwt(data, mode=self.mode, level=self.level, wavelet=self.wavelet)
        )

    def adj_op(self, coeffs):
        """Define the wavelet adjoint operator.

        This method returns the reconstructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.n_coils > 1:
            images = Parallel(
                n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
            )(
                delayed(self._adj_op)(coeffs[i], self.coeffs_shape[i])
                for i in np.arange(self.n_coils)
            )
            images = np.asarray(images)
        else:
            images = self._adj_op(coeffs)
        return images

    def _adj_op(self, coeffs):
        """Single coil inverse wavelet transform."""
        return self.idwt(
            pywt.unravel_coeffs(
                coeffs, self.coeffs_slices, self.raw_coeffs_shape, self._pywt_fun
            ),
            wavelet=self.wavelet,
            mode=self.mode,
        )
