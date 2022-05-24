#!/usr/bin/env python
"""
Low level python bindings for the cufinufft CUDA libraries.

Seperate bindings are provided for single and double precision libraries,
differentiated by 'f' suffix.

It was copied from the core cufinufft library.
"""

import ctypes
import os
import importlib

from ctypes import c_double, c_float, c_int, c_void_p

import numpy as np

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)

LIB = None
# Try to load a local library directly.
try:
    LIB = ctypes.cdll.LoadLibrary('libcufinufft.so')
except OSError:
    # Should that not work, try to find the full path of a packaged lib.
    #   The packaged lib should have a py/platform decorated name,
    #   and be rpath'ed the true CUDA C cufinufft library through the
    #   Extension and wheel systems.
    try:
        if LIB is None:
            # Find the library.
            fh = importlib.util.find_spec('cufinufftc')[0]
            # Get the full path for the ctypes loader.
            full_lib_path = os.path.realpath(fh.name)
            fh.close()    # Be nice and close the open file handle.

            # Load the library,
            #    which rpaths the libraries we care about.
            LIB = ctypes.cdll.LoadLibrary(full_lib_path)

    except Exception as exc:
        raise RuntimeError('Failed to find cufinufft library') from exc


def _get_ctypes(dtype):
    """Check if dtype is float32 or float64.

    Returns floating point and floating point pointer.
    """
    if dtype == np.float64:
        real_t = c_double
    elif dtype == np.float32:
        real_t = c_float
    else:
        raise TypeError("Expected np.float32 or np.float64.")

    real_ptr = ctypes.POINTER(real_t)

    return real_t, real_ptr


OPTS_FIELD_DECODE = {
    "gpu_method": {1: "nonuniform pts driven",
                   2: "shared memory"},
    "gpu_sort": {0: "no sort (GM)",
                 1: "sort (GM-sort)"},
    "kerevalmeth": {0: "direct eval exp(sqrt())",
                    1: "Horner ppval"},
    "gpu_spreadinterponly": {0: "NUFFT",
                             1: "spread or interpolate only", }
}


class NufftOpts(ctypes.Structure):
    """Optional Parameters for the plan setup."""

    _fields_ = [
        ('upsampfac', c_double),
        ('gpu_method', c_int),
        ('gpu_sort', c_int),
        ('gpu_binsizex', c_int),
        ('gpu_binsizey', c_int),
        ('gpu_binsizez', c_int),
        ('gpu_obinsizex', c_int),
        ('gpu_obinsizey', c_int),
        ('gpu_obinsizez', c_int),
        ('gpu_maxsubprobsize', c_int),
        ('gpu_nstreams', c_int),
        ('gpu_kerevalmeth', c_int),
        ('gpu_spreadinterponly', c_int),
        ('gpu_device_id', c_int)]

    def __repr__(self):
        """Get the value of the struct, like a dict."""
        ret = "Struct(\n"
        for fieldname, _ in self._fields_:
            ret += f"{fieldname}: {getattr(self, fieldname)},\n"
        ret += ")"
        return ret

    def __str__(self):
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


CufinufftPlan = c_void_p
CufinufftPlanf = c_void_p

CufinufftPlan_p = ctypes.POINTER(CufinufftPlan)
CufinufftPlanf_p = ctypes.POINTER(CufinufftPlanf)

NufftOpts_p = ctypes.POINTER(NufftOpts)

_default_opts = LIB.cufinufft_default_opts
_default_opts.argtypes = [c_int, c_int, NufftOpts_p]
_default_opts.restype = c_int

_make_pland = LIB.cufinufft_makeplan
_make_pland.argtypes = [
    c_int, c_int, c_int_p, c_int,
    c_int, c_double, c_int, CufinufftPlan_p, NufftOpts_p]
_make_pland.restypes = c_int

_make_planf = LIB.cufinufftf_makeplan
_make_planf.argtypes = [
    c_int, c_int, c_int_p, c_int,
    c_int, c_float, c_int, CufinufftPlanf_p, NufftOpts_p]
_make_planf.restypes = c_int

_set_ptsd = LIB.cufinufft_setpts
_set_ptsd.argtypes = [
    c_int, c_void_p, c_void_p, c_void_p, ctypes.c_int, c_double_p,
    c_double_p, c_double_p, c_void_p]
_set_ptsd.restype = c_int

_set_ptsf = LIB.cufinufftf_setpts
_set_ptsf.argtypes = [
    c_int, c_void_p, c_void_p, c_void_p, ctypes.c_int, c_float_p,
    c_float_p, c_float_p, c_void_p]
_set_ptsf.restype = c_int

_exec_pland = LIB.cufinufft_execute
_exec_pland.argtypes = [c_void_p, c_void_p, c_void_p]
_exec_pland.restype = c_int

_exec_planf = LIB.cufinufftf_execute
_exec_planf.argtypes = [c_void_p, c_void_p, c_void_p]
_exec_planf.restype = c_int

_destroy_pland = LIB.cufinufft_destroy
_destroy_pland.argtypes = [c_void_p]
_destroy_pland.restype = c_int

_destroy_planf = LIB.cufinufftf_destroy
_destroy_planf.argtypes = [c_void_p]
_destroy_planf.restype = c_int
