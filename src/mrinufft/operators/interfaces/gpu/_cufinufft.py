#!/usr/bin/env python
"""
Low level python bindings for the cufinufft CUDA libraries.

This apply some glue between the C++/CUDA library of Cufinufft and the python interface

Seperate bindings are provided for single and double precision libraries,
differentiated by 'f' suffix.
"""

import atexit
import sys

import ctypes
from ctypes import c_void_p, byref, c_int64, c_float
import numpy as np

from .utils import check_error, is_cuda_array
c_float_p = ctypes.POINTER(c_float)

try:
    from cufinufft._cufinufft import (
        NufftOpts, _default_opts,
        _make_plan, _make_planf,
        _set_pts, _set_ptsf,
        _exec_plan, _exec_planf,
        _spread_interp, _spread_interpf,
        _destroy_plan, _destroy_planf,
    )
    CUFINUFFT_LIB_AVAILABLE = True
    EXITING = False
    atexit.register(setattr, sys.modules[__name__], "EXITING", True)
except ImportError:
    CUFINUFFT_LIB_AVAILABLE = False



OPTS_FIELD_DECODE = {
    "gpu_method": {1: "nonuniform pts driven", 2: "shared memory"},
    "gpu_sort": {0: "no sort (GM)", 1: "sort (GM-sort)"},
    "kerevalmeth": {0: "direct eval exp(sqrt())", 1: "Horner ppval"},
    "gpu_spreadinterponly": {
        0: "NUFFT",
        1: "spread or interpolate only",
    },
}

def repr_opts(self):
    """Get the value of the struct, like a dict."""
    ret = "Struct(\n"
    for fieldname, _ in self._fields_:
        ret += f"{fieldname}: {getattr(self, fieldname)},\n"
    ret += ")"
    return ret
def str_opts(self):
    """Get the value of the struct, with their meaning."""
    ret = "Struct(\n"
    for fieldname, _ in self._fields_:
        ret += f"{fieldname}: {getattr(self, fieldname)}"
        decode = OPTS_FIELD_DECODE.get(fieldname)
        if decode:
            ret += f" [{decode[getattr(self, fieldname)]}]"
        ret += "\n"
    ret += ")"
    return ret
    
NufftOpts.__repr__ = lambda self: repr_opts(self)
NufftOpts.__str__ = lambda self: str_opts(self)


def get_default_opts(nufft_type, dim):
    """
    Generate a cufinufft opt struct of the dtype coresponding to plan.

    Parameters
    ----------
    finufft_type: int
        Finufft Type (`1` or `2`)
    dim: int
        Number of dimension (1, 2, 3)

    Returns
    -------
    nufft_opts structure.
    """
    nufft_opts = NufftOpts()

    ier = _default_opts(nufft_type, dim, nufft_opts)
    check_error(ier, "Configuration not yet implemented.")

    return nufft_opts


def get_kx_ky_kz_pointers(samples):
    n_samples = len(samples)
    itemsize = samples.dtype.itemsize
    ptr = samples.data.ptr
    fpts_axes = [None, None, None]
    # samples are column-major ordered.
    # We get the internal pointer associated with each axis.
    for i in range(samples.shape[-1]):
        fpts_axes[i] = ptr + i * n_samples * itemsize
    return n_samples, fpts_axes
    
def convert_shape_to_3D(shape, dim):
    return shape[::-1] + (1,) * (3 - dim) 
    
def get_spread_interp_func(dtype):
    if dtype == np.float64:
        spread_interp = _spread_interp
    elif dtype == np.float32:
        spread_interp = _spread_interpf
    return spread_interp


def _spread(samples, c, f):
    spread_interp = get_spread_interp_func(samples.dtype)
    n_samples, fpts_axes = get_kx_ky_kz_pointers(samples)
    shape = convert_shape_to_3D(f.shape, samples.shape[-1])
    opts = get_default_opts(1, samples.shape[-1])
    spread_interp(1, samples.shape[-1], *shape, n_samples, *fpts_axes, c.data.ptr, f.data.ptr, opts, 1e-4)


def _interp(samples, c, f):
    spread_interp = get_spread_interp_func(samples.dtype)
    n_samples, fpts_axes = get_kx_ky_kz_pointers(samples)
    shape = convert_shape_to_3D(f.shape, samples.shape[-1])
    opts = get_default_opts(2, samples.shape[-1])
    spread_interp(2, samples.shape[-1], *shape, n_samples, *fpts_axes, c.data.ptr, f.data.ptr, opts, 1e-4)
    

class RawCufinufft:
    """GPU implementation of N-D non uniform Fast Fourrier Transform class.

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
        if not CUFINUFFT_LIB_AVAILABLE:
            raise RuntimeError("cufinufft library not found. Please install it first.")

        self.dtype = np.dtype(dtype)

        if self.dtype == np.float32:
            self.__make_plan = _make_planf
            self.__set_pts = _set_ptsf
            self.__exec_plan = _exec_planf
            self.__destroy_plan = _destroy_planf
            self.complex_dtype = np.complex64
        elif self.dtype == np.float64:
            self.__make_plan = _make_plan
            self.__set_pts = _set_pts
            self.__exec_plan = _exec_plan
            self.__destroy_plan = _destroy_plan
            self.complex_dtype = np.complex128
        else:
            raise TypeError("Expected np.float32.")

        if not samples.flags.f_contiguous and not is_cuda_array(samples):
            raise ValueError(
                "samples should be a f-contiguous (column major) GPUarray."
            )

        self.samples = samples

        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans

        # We extend the mode tuple to 3D as needed,
        #   and reorder from C/python ndarray.shape style input (nZ, nY, nX)
        #   to the (F) order expected by the low level library (nX, nY, nZ).
        shape = shape[::-1] + (1,) * (3 - self.ndim)
        self.modes = (c_int64 * 3)(*shape)

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
                byref(plan),
                self.opts[typ],
            )
            check_error(ier, f"Type {typ} plan initialisation failed.")
            self.plans[typ] = plan
        else:
            raise RuntimeError(f"Type {typ} plan already exist.")

    def _set_pts(self, typ):
        if self.samples.dtype != self.dtype:
            raise TypeError("cufinufft plan.dtype and " "samples dtypes do not match.")

        n_samples, fpts_axes = get_kx_ky_kz_pointers(self.samples)
        
        ier = self.__set_pts(
            self.plans[typ], n_samples, *fpts_axes, 0, None, None, None,
        )
        check_error(ier, f"Error setting non-uniforms points of type{typ}")

    def _exec_plan(self, typ, c_ptr, f_ptr):
        ier = self.__exec_plan(self.plans[typ], c_ptr, f_ptr)
        check_error(ier, f"Error executing Type {typ} plan.")

    def _destroy_plan(self, typ):
        if self.plans[typ] is None:
            return None  # nothing to do.
        ier = self.__destroy_plan(self.plans[typ])
        check_error(ier, f"Error deleting Type {typ} plan.")
        self.plans[typ] = None
        return None

    def _type_exec(self, typ, d_c_ptr, d_grid_ptr):
        if self.plans[typ] is not None:
            self._exec_plan(typ, d_c_ptr, d_grid_ptr)
        else:
            self._make_plan(typ)
            self._set_pts(typ)
            self._exec_plan(typ, d_c_ptr, d_grid_ptr)
            self._destroy_plan(typ)

    # Exposed middle level interface #

    def type1(self, d_c_ptr, d_grid_ptr):
        """Type 1 transform, using on-gpu data.

        Parameters
        ----------
        d_c_ptr: int
            pointer to on-device non uniform coefficient array.

        d_grid_ptr: int
            pointer to on-device uniform grid array.
        """
        return self._type_exec(1, d_c_ptr, d_grid_ptr)

    def type2(self, d_c_ptr, d_grid_ptr):
        """
        Type 2 transform, using on-gpu data.

        Parameters
        ----------
        d_c_ptr: int
            pointer to on-device non uniform coefficient array.

        d_grid_ptr: int
            pointer to on-device uniform grid array.
        """
        return self._type_exec(2, d_c_ptr, d_grid_ptr)

    def __del__(self):
        """Destroy this instance's associated plan and data."""
        # If the process is exiting or we've already cleaned up plan, return.
        if EXITING or (self.plans[1] is None and self.plans[2] is None):
            return
        self._destroy_plan(1)
        self._destroy_plan(2)
        # Reset plan to avoid double destroy.
        self.plans = [None, None, None]
