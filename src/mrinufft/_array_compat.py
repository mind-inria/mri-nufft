"""Array libraries compatibility utils."""

from __future__ import annotations

import importlib.util
import logging
from functools import partial, wraps
from inspect import cleandoc
from numbers import Number  # abstract base type for python numeric type
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

logger = logging.getLogger(__name__)


def _module_available(name: str) -> bool:
    """Check availability of an optional dependency without importing it."""
    return importlib.util.find_spec(name) is not None


CUPY_AVAILABLE = _module_available("cupy")
TORCH_AVAILABLE = _module_available("torch")
AUTOGRAD_AVAILABLE = TORCH_AVAILABLE
TENSORFLOW_AVAILABLE = _module_available("tensorflow")
DEEPINV_AVAILABLE = TORCH_AVAILABLE and _module_available("deepinv")

_ARRAY_NAMESPACES = ("numpy", "cupy", "torch", "tensorflow")


def is_array(x: Any) -> bool:
    """Return True if x is a numpy/cupy/torch/tensorflow array (not a np scalar)."""
    return (
        not isinstance(x, np.generic) and _get_array_module_name(x) in _ARRAY_NAMESPACES
    )


_NP2TORCH = None
_TF_CUDA_AVAILABLE = None


def _np2torch():
    """Lazily build and cache the numpy->torch dtype mapping."""
    global _NP2TORCH
    if _NP2TORCH is None:
        import torch

        _NP2TORCH = {
            np.dtype("float64"): torch.float64,
            np.dtype("float32"): torch.float32,
            np.dtype("complex64"): torch.complex64,
            np.dtype("complex128"): torch.complex128,
        }
    return _NP2TORCH


def _tf_cuda_is_available():
    """Check (and cache) whether Tensorflow has CUDA support."""
    global _TF_CUDA_AVAILABLE
    if _TF_CUDA_AVAILABLE is None:
        if TENSORFLOW_AVAILABLE:
            import tensorflow as tf

            devices = tf.config.list_physical_devices()
            _TF_CUDA_AVAILABLE = "GPU" in [d.device_type for d in devices]
        else:
            _TF_CUDA_AVAILABLE = False
    return _TF_CUDA_AVAILABLE


def __getattr__(name):
    """Expose lazily-computed values as module attributes (PEP 562)."""
    if name == "NP2TORCH":
        return _np2torch()
    if name == "TF_CUDA_AVAILABLE":
        return _tf_cuda_is_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_array_module_name(array) -> str:
    return type(array).__module__.partition(".")[0]


def get_array_module(array):
    """Get the array module (numpy/cupy/torch/tnp) of an array."""
    if isinstance(array, Number | np.generic):
        return np
    root = _get_array_module_name(array)
    if root == "numpy":
        return np
    if root in ("cupy", "torch"):
        return importlib.import_module(root)
    if root == "tensorflow":
        from tensorflow.experimental import numpy as tnp

        return tnp
    raise ValueError(f"Unknown array library (={type(array)}.")


def auto_cast(array, dtype: DTypeLike):
    """Cast an array or a tensor to the desired dtype.

    This automatically convert numpy/torch dtype to suitable format.
    """
    if _get_array_module_name(array) == "torch":
        return array.to(_np2torch()[np.dtype(dtype)], copy=False)
    return array.astype(dtype, copy=False)


def is_cuda_array(var) -> bool:
    """Check if var implement the CUDA Array interface."""
    try:
        return hasattr(var, "__cuda_array_interface__")
    except Exception:
        return False


def is_cuda_tensor(var: Any) -> bool:
    """Check if var is a CUDA tensor."""
    return _get_array_module_name(var) == "torch" and var.is_cuda


def is_host_array(var: Any) -> bool:
    """Check if var is a host contiguous np array."""
    try:
        if isinstance(var, np.ndarray):
            if not var.flags.c_contiguous:
                logger.warning("The input is CPU array but not C-contiguous. ")
                return False
            return True
    except Exception:
        pass
    return False


def pin_memory(array: NDArray) -> NDArray:
    """Create a copy of the array in pinned memory."""
    import cupy as cp

    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


NUMPY_NOTE = """
.. note::

    This function uses ``numpy`` internally, and will convert all its array
    argument to numpy arrays. The outputs will be converted back to the original
    array module and device.
"""
NUMPY_CUPY_NOTE = """
.. note::

    This function uses ``numpy`` for all CPU arrays, and ``cupy`` for all on-gpu
    array. It will convert all its array argument to the respective array
    library. The outputs will be converted back to the original array module and
    device.
"""

TORCH_NOTE = """
.. note::

    This function uses ``torch`` internally, and will convert all its array
    argument to torch tensors, but will respect the device they are allocated
    on. The outputs will be converted back to the original array module and
    device.
"""

TF_NOTE = """
.. note::

    This function uses ``tensorflow`` internally, and will convert all its array
    argument to tensorflow tensors. but will respect the device they are
    allocated on. The outputs will be converted back to the original array
    module and device.
"""


def _with(fun, _to_inner_xp, note):
    """Create a decorator to convert to a given array interface."""

    @wraps(fun)
    def wrapper(*args, **kwargs):

        leading_arg = _get_leading_argument(args, kwargs)

        # get array module from leading argument
        xp = get_array_module(leading_arg)
        device = _get_device(leading_arg)

        # convert all to interface
        args, kwargs = _to_inner_xp(args, kwargs, device=device)

        # run function
        ret_ = fun(*args, **kwargs)

        # convert output to original array module and device
        return _to_interface(ret_, xp, device)

    if wrapper.__doc__:
        wrapper.__doc__ = cleandoc(wrapper.__doc__) + "\n\n" + note

    return wrapper


def with_numpy(fun):
    """Ensure the function works internally with numpy array."""
    return _with(fun, _to_numpy, NUMPY_NOTE)


def with_tensorflow(fun):
    """Ensure the function works internally with tensorflow array."""
    return _with(fun, _to_tensorflow, TF_NOTE)


def with_numpy_cupy(fun):
    """Ensure the function works internally with numpy or cupy array."""
    return _with(fun, _to_numpy_cupy, NUMPY_CUPY_NOTE)


def with_torch(fun):
    """Ensure the function works internally with Torch."""
    return _with(fun, _to_torch, TORCH_NOTE)


def _get_device(input):
    """Determine computational device from input array."""
    try:
        return input.device
    except AttributeError:
        return "cpu"


def _get_leading_argument(args, kwargs):
    """Find first array argument."""
    for arg in args:
        try:
            get_array_module(arg)
            return arg
        except Exception:
            pass
    for arg in kwargs.values():
        try:
            get_array_module(arg)
            return arg
        except Exception:
            pass
    return np.asarray(0.0)  # by default, use dummy numpy leading arg


def _array_to_numpy(_arg):
    """Convert array to Numpy."""
    if is_cuda_array(_arg):
        logger.warning("data is on gpu, it will be moved to CPU.")
    xp = get_array_module(_arg)
    if xp.__name__ == "torch":
        _arg = _arg.numpy(force=True)
    elif xp.__name__ == "cupy":
        import cupy as cp

        _arg = cp.asnumpy(_arg)
    elif "tensorflow" in xp.__name__:
        _arg = _arg.numpy()
    return _arg


def _array_to_cupy(_arg, device=None):
    """Convert array to Cupy."""
    import cupy as cp

    xp = get_array_module(_arg)
    if xp.__name__ == "numpy":
        with cp.cuda.Device(device):
            _arg = cp.asarray(_arg)
    elif xp.__name__ == "torch":
        if _arg.requires_grad:
            _arg = _arg.detach()
        if _arg.is_cpu:
            with cp.cuda.Device(device):
                _arg = cp.asarray(_arg.numpy())
        else:
            _arg = cp.from_dlpack(_arg)
    elif "tensorflow" in xp.__name__:
        import tensorflow as tf

        if "CPU" in _arg.device:
            with cp.cuda.Device(device):
                _arg = cp.asarray(_arg.numpy())
        else:
            _arg = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(_arg))

    return _arg


def _array_to_torch(_arg, device=None):
    """Convert array to torch."""
    import torch

    if device is None:
        device = _get_device(_arg)
    xp = get_array_module(_arg)
    if xp.__name__ == "numpy":
        _arg = torch.as_tensor(_arg, device=device)
    elif xp.__name__ == "cupy":
        if torch.cuda.is_available():
            _arg = torch.from_dlpack(_arg)
        else:
            import cupy as cp

            logger.warning("data is on gpu, it will be moved to CPU.")
            _arg = torch.as_tensor(cp.asnumpy(_arg))
    elif "tensorflow" in xp.__name__:
        import tensorflow as tf

        if "CPU" in _arg.device:
            _arg = torch.as_tensor(_arg.numpy(), device=device)
        else:
            _arg = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(_arg))
    return _arg


def _array_to_tensorflow(_arg):
    import tensorflow as tf

    xp = get_array_module(_arg)
    if xp.__name__ == "numpy":
        with tf.device("CPU"):
            _arg = tf.convert_to_tensor(_arg)
    elif xp.__name__ == "cupy":
        if _tf_cuda_is_available():
            _arg = tf.experimental.dlpack.from_dlpack(_arg.toDlpack())
        else:
            import cupy as cp

            logger.warning("data is on gpu, it will be moved to CPU.")
            _arg = tf.convert_to_tensor(cp.asnumpy(_arg))
    elif xp.__name__ == "torch":
        import torch

        if _arg.requires_grad:
            _arg = _arg.detach()
        if _arg.is_cpu:
            _arg = tf.convert_to_tensor(_arg)
        elif _tf_cuda_is_available():
            _arg = tf.experimental.dlpack.from_dlpack(
                torch.utils.dlpack.to_dlpack(_arg)
            )
        else:
            _arg = tf.convert_to_tensor(_arg.numpy(force=True))


def _convert(_array_to_xp, args, kwargs=None):
    # convert positional arguments
    args = list(args)
    for n in range(len(args)):
        _arg = args[n]
        # All array are converted to the detected module
        # but numpy scalars are left as is.
        if is_array(_arg):
            args[n] = _array_to_xp(_arg)
        elif isinstance(_arg, tuple | list) and is_array(_arg[0]):
            args[n], _ = _convert(_array_to_xp, _arg)
        elif isinstance(_arg, dict):
            _, args[n] = _convert(_array_to_xp, [], _arg)
    # convert keyworded
    if kwargs:
        process_kwargs_vals, _ = _convert(_array_to_xp, kwargs.values())
        kwargs = {k: v for k, v in zip(kwargs.keys(), process_kwargs_vals)}

    return args, kwargs


def _to_numpy(args, kwargs=None, device=None):
    """Convert a sequence of arguments to numpy.

    Non-arrays are ignored.
    """
    return _convert(_array_to_numpy, args, kwargs)


def _to_cupy(args, kwargs=None, device=None):
    """Convert a sequence of arguments to cupy.

    Non-arrays are ignored.

    This avoid transfers between different devices (e.g., different GPUs).
    """
    # enforce mutable
    if str(device) == "cpu":
        device = 0
    else:
        try:
            device = device.index
        except AttributeError:
            pass

    return _convert(partial(_array_to_cupy, device=device), args, kwargs)


def _to_numpy_cupy(args, kwargs=None, device=None):
    """Convert a sequence of arguments to numpy or cupy.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
    if device is None:
        leading_arg = _get_leading_argument(args, kwargs)
        device = _get_device(leading_arg)
    if CUPY_AVAILABLE and str(device) != "cpu":
        return _to_cupy(args, kwargs, device)
    else:
        return _to_numpy(args, kwargs)


def _to_torch(args, kwargs=None, device=None):
    """Convert a sequence of arguments to Pytorch Tensors.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
    return _convert(partial(_array_to_torch, device=device), args, kwargs)


def _to_tensorflow(args, kwargs=None):
    """Convert a sequence of arguments to Tensorflow tensors.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
    return _convert(_array_to_tensorflow, args, kwargs)


def _to_interface(args, array_interface, device=None):
    """
    Convert a list of arguments to a given array interface.

    User may provide the desired computational device.
    Non-arrays are ignored.

    Parameters
    ----------
    args : list[object]
        List of objects to be converted.
    array_interface : ModuleType
        Desired array backend (e.g., numpy).
    device : Device, optional
        Desired computational device, (e.g., "cpu" or Device('cuda')).
        The default is None (i.e., maintain same device as input argument).

    Returns
    -------
    list[object]
        List of converted objects.

    """
    # enforce iterable
    if isinstance(args, list | tuple) is False:
        args = [args]

    # convert to target interface
    if array_interface.__name__ == "numpy":
        args, _ = _to_numpy(args)
    elif array_interface.__name__ == "cupy":
        args, _ = _to_cupy(args, device=device)
    elif array_interface.__name__ == "torch":
        args, _ = _to_torch(args, device=device)

    if len(args) == 1:
        return args[0]

    return tuple(args)
