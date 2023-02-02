"""Low level interface for finufft."""
import numpy as np

from ctypes import c_int, c_void_p, byref

FINUFFT_AVAILABLE = True
try:
    import finufft._finufft as _finufft
except ImportError:
    FINUFFT_AVAILABLE = False


def check_error(ier, message):  # noqa: D103
    if ier != 0:
        raise RuntimeError(message)


def get_default_opts(nufft_type, dim):

    nufft_opts = _finufft.FinufftOpts()
    ier = _finufft._default_opts(nufft_type, dim, nufft_opts)
    check_error(ier, "Configuration not yet implemented")

    return nufft_opts


def get_ptr(array):
    """Return a ctype pointer to the array data."""
    return array.ctypes.data_as(c_void_p)


class RawFinufft:
    """Middleware interface for finufft.

    This object is responsible for the handling of finufft plans and low level execution.

    Parameters
    ----------
    samples: np.ndarray
        Samples in the non uniform space (K-space)
    shape: tuple
        Shape of the uniform space (Image space)
    n_trans: int, default 1
        Number of transform  executed by the plan.
    dtype: str or np.dtype
        Base dtype for the input data, default float32 (and thus complex64)
    opts: dict or tuple of two dict, optional default None.
        Extra parameters for the type 1  and type 2 plan.
        It will be used to set non default argument for NufftOpts object.
    init_plans: bool default False
        If True, initialize cufinuffts plans at the end of constructor.

    Methods
    -------
    type1(coef, data)
        Type 1 tranform. data is updated with the result.
    type2(coef, data)
        Type 2 tranform. coef is updated with the results

    See Also
    --------
    ..gpu.cufinufft.RawCufinufft

    Notes
    -----
    This implementation bypass the `Plan` interface
    """

    def __init__(
        self,
        samples,
        shape,
        n_trans=1,
        eps=1e-4,
        dtype=np.float32,
        init_plans=False,
        opts=None,
    ):
        self.dtype = np.dtype(dtype)

        if self.dtype == np.float32:
            self.__make_plan = _finufft._make_planf
            self.__set_pts = _finufft._set_ptsf
            self.__exec_plan = _finufft._exec_planf
            self.__destroy_plan = _finufft._destroy_planf
            self.complex_dtype = np.complex64
        elif self.dtype == np.float64:
            self.__make_plan = _finufft._make_pland
            self.__set_pts = _finufft._set_ptsd
            self.__exec_plan = _finufft._exec_pland
            self.__destroy_plan = _finufft._destroy_pland
            self.complex_dtype = np.complex128
        else:
            raise TypeError("Expected np.float32.")

        if not samples.flags.f_contiguous:
            raise ValueError("samples should be a f-contiguous (column major) array.")

        self.samples = samples

        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans

        # We extend the mode tuple to 3D as needed,
        #   and reorder from C/python ndarray.shape style input (nZ, nY, nX)
        #   to the (F) order expected by the low level library (nX, nY, nZ).
        shape = shape[::-1] + (1,) * (3 - self.ndim)
        self.modes = (c_int * 3)(*shape)

        # setup optional parameters of the plan.
        use_opts1 = get_default_opts(1, self.ndim)
        use_opts2 = get_default_opts(2, self.ndim)

        if opts is not None:
            if isinstance(opts, dict):
                _opts = (opts, opts)
            elif isinstance(opts, tuple) and len(opts) == 2:
                _opts = opts
            else:
                raise ValueError("opts should be a dict or 2-tuple of dict.")

            for cls_opts, opt in zip([use_opts1, use_opts2], _opts):
                field_names = [name for name, _ in cls_opts._fields_]
                # Assign field names from kwargs if they match up.
                for key, val in opt.items():
                    try:
                        setattr(cls_opts, key, val)
                    except AttributeError as exc:
                        raise ValueError(
                            f"Invalid option '{key}', "
                            f"it should be one of {field_names}"
                        ) from exc

        # Easy access to the plans and opts.
        # the first element is dummy so that we can use index 1 and 2 to access
        # the relevant type.
        self.plans = [None, None, None]
        self.opts = [None, use_opts1, use_opts2]

        if init_plans:
            for typ in [1, 2]:
                self._make_plan(typ)
                self._set_pts(typ)

        def _make_plan(self, typ):
            if self.plans[typ] is None:
                plan = c_void_p(None)
                ier = self.__make_plan(
                    typ,
                    self.ndim,
                    self.modes,
                    1 if typ == 1 else -1,
                    self.n_trans,
                    self.eps,
                    1,
                    byref(plan),
                    self.opts[typ],
                )
                check_error(ier, f"Type {typ} plan initialisation failed.")
                self.plans[typ] = plan
            else:
                raise RuntimeError(f"Type {typ} plan already exist.")

        def _set_pts(self, typ):
            if self.samples.dtype != self.dtype:
                raise TypeError(
                    "cufinufft plan.dtype and " "samples dtypes do not match."
                )

            n_samples = len(self.samples)
            fpts_axes = [None, None, None]
            # samples are column-major ordered.
            for i in range(self.samples.shape[1]):
                fpts_axes[i] = np.asarrayself.samples[:, i]

            ier = self.__set_pts(
                self.plans[typ],
                n_samples,
                *fpts_axes,
                0,
                None,
                None,
                None,
            )
            check_error(ier, f"Error setting non-uniforms points of type{typ}")

        def _exec_plan(self, typ, c_ptr, f_ptr):
            ier = self.__exec_plan(c_ptr, f_ptr, self.plans[typ])
            check_error(ier, f"Error executing Type {typ} plan.")

        def _destroy_plan(self, typ):
            if self.plans[typ] is None:
                return None  # nothing to do.
            ier = self.__destroy_plan(self.plans[typ])
            check_error(ier, f"Error deleting Type {typ} plan.")
            self.plans[typ] = None
            return None

        def _type_exec(self, typ, c_ptr, grid_ptr):
            if self.plans[typ] is not None:
                self._exec_plan(typ, c_ptr, grid_ptr)
            else:
                self._make_plan(typ)
                self._set_pts(typ)
                self._exec_plan(typ, c_ptr, grid_ptr)
                self._destroy_plan(typ)

        # Exposed middle level interface #

        def type1(self, coeff_data, grid_data):
            """Type 1 transform.

            Parameters
            ----------
            c_ptr: int
                pointer to on-device non uniform coefficient array.

            grid_ptr: int
                pointer to on-device uniform grid array.
            """
            return self._type_exec(1, get_ptr(coeff_data), get_ptr(grid_data))

        def type2(self, coeff_data, grid_data):
            """
            Type 2 transform.

            Parameters
            ----------
            d_c_ptr: int
                pointer to on-device non uniform coefficient array.

            d_grid_ptr: int
                pointer to on-device uniform grid array.
            """
            return self._type_exec(2, get_ptr(coeff_data), get_ptr(grid_data))

        def __del__(self):
            """Destroy this instance's associated plan and data."""
            # If the process is exiting or we've already cleaned up plan, return.
            if self.plans[1] is None and self.plans[2] is None:
                return
            self._destroy_plan(1)
            self._destroy_plan(2)
            # Reset plan to avoid double destroy.
            self.plans = [None, None, None]
