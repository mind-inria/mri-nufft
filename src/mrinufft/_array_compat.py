"""Array libraries compatibility utils."""

import warnings
from numbers import Number  # abstract base type for python numeric type
from functools import wraps, partial
from inspect import cleandoc
from numpy.typing import NDArray, DTypeLike
import numpy as np

ARRAY_LIBS = {
    "numpy": (np, np.ndarray),
    "cupy": (None, None),
    "torch": (None, None),
    "tensorflow": (None, None),
}

ArrayTypes = np.ndarray
# TEST import of array libraries.

TENSORFLOW_AVAILABLE = True
CUPY_AVAILABLE = True
AUTOGRAD_AVAILABLE = True
TORCH_AVAILABLE = True

try:
    import cupy as cp

    ARRAY_LIBS["cupy"] = (cp, cp.ndarray)
    ArrayTypes = ArrayTypes | cp.ndarray
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch

    ARRAY_LIBS["torch"] = (torch, torch.Tensor)
    ArrayTypes = ArrayTypes | torch.Tensor
except ImportError:
    AUTOGRAD_AVAILABLE = False
    TORCH_AVAILABLE = False
    pass
    NP2TORCH = {}
else:
    NP2TORCH = {
        np.dtype("float64"): torch.float64,
        np.dtype("float32"): torch.float32,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
    }
try:
    import tensorflow as tf
    from tensorflow.experimental import numpy as tnp

    ARRAY_LIBS["tensorflow"] = (tnp, tnp.ndarray)
    ArrayTypes = ArrayTypes | tnp.ndarray
except ImportError:
    TENSORFLOW_AVAILABLE = False
    pass


def get_array_module(array: NDArray | Number) -> np:  # type: ignore
    """Get the module of the array."""
    if isinstance(array, Number | np.generic):
        return np
    for lib, array_type in ARRAY_LIBS.values():
        if lib is not None and isinstance(array, array_type):
            return lib
    raise ValueError(f"Unknown array library (={type(array)}.")


def auto_cast(array, dtype: DTypeLike):
    """Cast an array or a tensor to the desired dtype.

    This automatically convert numpy/torch dtype to suitable format.
    """
    module = get_array_module(array)
    if module.__name__ == "torch":
        return array.to(NP2TORCH[np.dtype(dtype)], copy=False)
    else:
        return array.astype(dtype, copy=False)


def _tf_cuda_is_available():
    """Check whether Tensorflow has CUDA support or not."""
    if TENSORFLOW_AVAILABLE:
        devices = tf.config.list_physical_devices()
        device_type = [device.device_type for device in devices]
        return "GPU" in device_type
    else:
        return False


TF_CUDA_AVAILABLE = _tf_cuda_is_available()


def is_cuda_array(var) -> bool:
    """Check if var implement the CUDA Array interface."""
    try:
        return hasattr(var, "__cuda_array_interface__")
    except Exception:
        return False


def is_cuda_tensor(var) -> bool:
    """Check if var is a CUDA tensor."""
    return TORCH_AVAILABLE and isinstance(var, torch.Tensor) and var.is_cuda


def is_host_array(var) -> bool:
    """Check if var is a host contiguous np array."""
    try:
        if isinstance(var, np.ndarray):
            if not var.flags.c_contiguous:
                warnings.warn("The input is CPU array but not C-contiguous. ")
                return False
            return True
    except Exception:
        pass
    return False


def pin_memory(array: NDArray) -> NDArray:
    """Create a copy of the array in pinned memory."""
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
        warnings.warn("data is on gpu, it will be moved to CPU.")
    xp = get_array_module(_arg)
    if xp.__name__ == "torch":
        _arg = _arg.numpy(force=True)
    elif xp.__name__ == "cupy":
        _arg = cp.asnumpy(_arg)
    elif "tensorflow" in xp.__name__:
        _arg = _arg.numpy()
    return _arg


def _array_to_cupy(_arg, device=None):
    """Convert array to Cupy."""
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
        if "CPU" in _arg.device:
            with cp.cuda.Device(device):
                _arg = cp.asarray(_arg.numpy())
        else:
            _arg = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(_arg))

    return _arg


def _array_to_torch(_arg, device=None):
    """Convert array to torch."""
    xp = get_array_module(_arg)
    if xp.__name__ == "numpy":
        _arg = torch.as_tensor(_arg, device=device)
    elif xp.__name__ == "cupy":
        if torch.cuda.is_available():
            _arg = torch.from_dlpack(_arg)
        else:
            warnings.warn("data is on gpu, it will be moved to CPU.")
            _arg = torch.as_tensor(cp.asnumpy(_arg))
    elif "tensorflow" in xp.__name__:
        if "CPU" in _arg.device:
            _arg = torch.as_tensor(_arg.numpy(), device=device)
        else:
            _arg = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(_arg))
    return _arg


def _array_to_tensorflow(_arg):
    xp = get_array_module(_arg)
    if xp.__name__ == "numpy":
        with tf.device("CPU"):
            _arg = tf.convert_to_tensor(_arg)
    elif xp.__name__ == "cupy":
        if TF_CUDA_AVAILABLE:
            _arg = tf.experimental.dlpack.from_dlpack(_arg.toDlpack())
        else:
            warnings.warn("data is on gpu, it will be moved to CPU.")
            _arg = tf.convert_to_tensor(cp.asnumpy(_arg))
    elif xp.__name__ == "torch":
        if _arg.requires_grad:
            _arg = _arg.detach()
        if _arg.is_cpu:
            _arg = tf.convert_to_tensor(_arg)
        elif TF_CUDA_AVAILABLE:
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
        if isinstance(_arg, ArrayTypes):
            args[n] = _array_to_xp(_arg)
        elif isinstance(_arg, tuple | list) and isinstance(_arg[0], ArrayTypes):
            args[n], _ = _convert(_array_to_xp, _arg)
        elif isinstance(_arg, dict):
            args[n], _ = _convert(_array_to_xp, [], _arg)
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


def _to_numpy_cupy(args, kwargs=None, device="cpu"):
    """Convert a sequence of arguments to numpy or cupy.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
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
