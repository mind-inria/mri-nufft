"""Array libraries compatibility utils."""

import warnings

from functools import wraps

import numpy as np

from mrinufft._utils import get_array_module
from mrinufft.operators.interfaces.utils import is_cuda_array, is_cuda_tensor


CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False

AUTOGRAD_AVAILABLE = True
try:
    import torch
except ImportError:
    AUTOGRAD_AVAILABLE = False


TENSORFLOW_AVAILABLE = True
try:
    import tensorflow as tf
except ImportError:
    TENSORFLOW_AVAILABLE = False


def _tf_cuda_is_available():
    """Check whether Tensorflow has CUDA support or not."""
    if TENSORFLOW_AVAILABLE:
        devices = tf.config.list_physical_devices()
        device_type = [device.device_type for device in devices]
        return "GPU" in device_type
    else:
        return False


TF_CUDA_AVAILABLE = _tf_cuda_is_available()


def with_numpy(fun):
    """Ensure the function works internally with numpy array."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        leading_arg = _get_leading_argument(args, kwargs)

        # get array module from leading argument
        xp = get_array_module(leading_arg)
        device = _get_device(leading_arg)

        # convert all to numpy
        args, kwargs = _to_numpy(*args, **kwargs)

        # run function
        ret_ = fun(*args, **kwargs)

        # convert output to original array module and device
        return _to_interface(ret_, xp, device)

    return wrapper


def with_tensorflow(fun):
    """Ensure the function works internally with tensorflow array."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        leading_arg = _get_leading_argument(args, kwargs)

        # get array module from leading argument
        xp = get_array_module(leading_arg)
        device = _get_device(leading_arg)

        # convert all to tensorflow
        args, kwargs = _to_tensorflow(*args, **kwargs)

        # run function
        ret_ = fun(*args, **kwargs)

        # convert output to original array module and device
        return _to_interface(ret_, xp, device)

    return wrapper


def with_numpy_cupy(fun):
    """Ensure the function works internally with numpy or cupy array."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        leading_arg = _get_leading_argument(args, kwargs)

        # get array module from leading argument
        xp = get_array_module(leading_arg)

        # convert all to cupy / numpy according to data arg device
        args, kwargs = _to_numpy_cupy(args, kwargs, leading_arg)

        # run function
        ret_ = fun(*args, **kwargs)

        # convert output to original array module and device
        return _to_interface(ret_, xp)

    return wrapper


def with_torch(fun):
    """Ensure the function works internally with Torch."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        leading_arg = _get_leading_argument(args, kwargs)

        # get array module from leading argument
        xp = get_array_module(leading_arg)
        device = _get_device(leading_arg)

        # convert all arrays to torch
        args, kwargs = _to_torch(*args, device=device, **kwargs)

        # run function
        ret_ = fun(*args, **kwargs)

        # convert output to original array module and device
        return _to_interface(ret_, xp, device)

    return wrapper


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


def _to_numpy(*args, **kwargs):
    """Convert a sequence of arguments to numpy.

    Non-arrays are ignored.

    """
    # enforce mutable
    args = list(args)

    # convert positional arguments
    for n in range(len(args)):
        _arg = args[n]
        if hasattr(_arg, "__array__"):
            if is_cuda_array(_arg):
                warnings.warn("data is on gpu, it will be moved to CPU.")
            xp = get_array_module(_arg)
            if xp.__name__ == "torch":
                _arg = _arg.numpy(force=True)
            elif xp.__name__ == "cupy":
                _arg = cp.asnumpy(_arg)
            elif "tensorflow" in xp.__name__:
                _arg = _arg.numpy()
        if isinstance(_arg, (tuple, list)):
            _arg, _ = _to_numpy(*_arg)
        args[n] = _arg

    # convert keyworded
    if kwargs:
        process_kwargs_vals, _ = _to_numpy(*kwargs.values())
        kwargs = {k: v for k, v in zip(kwargs.keys(), process_kwargs_vals)}

    return args, kwargs


def _to_cupy(*args, device=None, **kwargs):
    """Convert a sequence of arguments to cupy.

    Non-arrays are ignored.

    This avoid transfers between different devices (e.g., different GPUs).
    """
    # enforce mutable
    args = list(args)

    # convert positional arguments
    for n in range(len(args)):
        _arg = args[n]
        if hasattr(_arg, "__array__"):
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
        if isinstance(_arg, (tuple, list)):
            _arg, _ = _to_cupy(*_arg, device=device)

        args[n] = _arg

    # convert keyworded
    if kwargs:
        process_kwargs_vals, _ = _to_cupy(*kwargs.values(), device)
        kwargs = {k: v for k, v in zip(kwargs.keys(), process_kwargs_vals)}

    return args, kwargs


def _to_numpy_cupy(args, kwargs, leading_argument):
    """Convert a sequence of arguments to numpy or cupy.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
    if (
        is_cuda_array(leading_argument) or is_cuda_tensor(leading_argument)
    ) and CUPY_AVAILABLE:
        return _to_cupy(*args, **kwargs)
    else:
        return _to_numpy(*args, **kwargs)


def _to_torch(*args, device=None, **kwargs):
    """Convert a sequence of arguments to Pytorch Tensors.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
    # enforce mutable
    args = list(args)

    # convert positional arguments
    for n in range(len(args)):
        _arg = args[n]
        if hasattr(_arg, "__array__"):
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
        if isinstance(_arg, (tuple, list)):
            _arg, _ = _to_torch(*_arg, device=device)

        args[n] = _arg

    # convert keyworded
    if kwargs:
        process_kwargs_vals, _ = _to_torch(*kwargs.values(), device)
        kwargs = {k: v for k, v in zip(kwargs.keys(), process_kwargs_vals)}

    return args, kwargs


def _to_tensorflow(*args, **kwargs):
    """Convert a sequence of arguments to Tensorflow tensors.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
    # enforce mutable
    args = list(args)

    # convert positional arguments
    for n in range(len(args)):
        _arg = args[n]
        if hasattr(_arg, "__array__"):
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
        if isinstance(_arg, (tuple, list)):
            _arg, _ = _to_tensorflow(*_arg)

        args[n] = _arg

    # convert keyworded
    if kwargs:
        process_kwargs_vals, _ = _to_tensorflow(*kwargs.values())
        kwargs = {k: v for k, v in zip(kwargs.keys(), process_kwargs_vals)}

    return args, kwargs


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
    if isinstance(args, (list, tuple)) is False:
        args = [args]

    # convert to target interface
    if array_interface.__name__ == "numpy":
        args, _ = _to_numpy(*args)
    elif array_interface.__name__ == "cupy":
        args, _ = _to_cupy(*args, device=device)
    elif array_interface.__name__ == "torch":
        args, _ = _to_torch(*args, device=device)

    if len(args) == 1:
        return args[0]

    return tuple(args)
