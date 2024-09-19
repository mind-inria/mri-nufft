"""Array libraries compatibility utils."""

import warnings
import inspect

from functools import wraps

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


tf_cuda_is_available = _tf_cuda_is_available()


def with_numpy(fun):
    """Ensure the function works internally with numpy array."""
    data_arg_idx = _ismethod(fun)

    @wraps(fun)
    def wrapper(*args, **kwargs):
        args = _get_args(fun, args, kwargs)

        # get array module and device from non-self argument
        xp = get_array_module(args[data_arg_idx])
        device = _get_device(args[data_arg_idx])

        # convert all to numpy
        args = _to_numpy(*args)

        # run function
        ret_ = fun(*args)

        # convert output to original array module and device
        return _to_interface(ret_, xp, device)

    return wrapper


def with_tensorflow(fun):
    """Ensure the function works internally with tensorflow array."""
    data_arg_idx = _ismethod(fun)

    @wraps(fun)
    def wrapper(*args, **kwargs):
        args = _get_args(fun, args, kwargs)

        # get array module from non-self argument
        xp = get_array_module(args[data_arg_idx])
        device = _get_device(args[data_arg_idx])

        # convert all to tensorflow
        args = _to_tensorflow(*args)

        # run function
        ret_ = fun(*args)

        # convert output to original array module and device
        print(xp.__name__)
        print(device)
        return _to_interface(ret_, xp, device)

    return wrapper


def with_numpy_cupy(fun):
    """Ensure the function works internally with numpy or cupy array."""
    data_arg_idx = _ismethod(fun)

    @wraps(fun)
    def wrapper(*args, **kwargs):
        args = _get_args(fun, args, kwargs)

        # get array module from non-self argument
        xp = get_array_module(args[data_arg_idx])

        # convert all to cupy / numpy according to data arg device
        args = _to_numpy_cupy(args, data_arg_idx)

        # run function
        ret_ = fun(*args)

        # convert output to original array module and device
        return _to_interface(ret_, xp)

    return wrapper


def with_torch(fun):
    """Ensure the function works internally with Torch."""
    data_arg_idx = _ismethod(fun)

    @wraps(fun)
    def wrapper(*args, **kwargs):
        args = _get_args(fun, args, kwargs)

        # get array module from non-self argument
        xp = get_array_module(args[data_arg_idx])
        device = _get_device(args[data_arg_idx])

        # convert all to tensorflow
        args = _to_torch(*args)

        # run function
        ret_ = fun(*args)

        # convert output to original array module and device
        return _to_interface(ret_, xp, device)

    return wrapper


def _ismethod(fun):
    """Determine whether input fun is instance-/classmethod or not."""
    first_arg = list(inspect.signature(fun).parameters)[0]
    return first_arg in ["self", "cls"]


def _get_device(input):
    """Determine computational device from input array."""
    try:
        return input.device
    except Exception:
        return "cpu"


def _get_args(func, args, kwargs):
    """Convert input args/kwargs mix to a list of positional arguments.

    This automatically fills missing kwargs with default values.
    """
    signature = inspect.signature(func)

    # Get number of arguments
    n_args = len(args)

    # Create a dictionary of keyword arguments and their default values
    _kwargs = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            _kwargs[k] = v.default
        else:
            _kwargs[k] = None

    # Merge the default keyword arguments with the provided kwargs
    for k in kwargs.keys():
        _kwargs[k] = kwargs[k]

    # Replace args
    _args = list(_kwargs.values())

    return list(args) + _args[n_args:]


def _to_numpy(*args):
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
        args[n] = _arg

    return args


def _to_cupy(*args, device=None):
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

        args[n] = _arg

    return args


def _to_numpy_cupy(args, data_arg_idx):
    """Convert a sequence of arguments to numpy or cupy.

    Non-arrays are ignored.

    This avoid transfers between different devices
    (e.g., CPU->GPU, GPU->CPU or different GPUs).
    """
    if is_cuda_array(args[data_arg_idx]) and CUPY_AVAILABLE:
        return _to_cupy(*args)
    elif is_cuda_tensor(args[data_arg_idx]) and CUPY_AVAILABLE:
        return _to_cupy(*args)
    else:
        return _to_numpy(*args)


def _to_torch(*args, device=None):
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

        args[n] = _arg

    return args


def _to_tensorflow(*args):
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
                if tf_cuda_is_available:
                    _arg = tf.experimental.dlpack.from_dlpack(_arg.toDlpack())
                else:
                    warnings.warn("data is on gpu, it will be moved to CPU.")
                    _arg = tf.convert_to_tensor(cp.asnumpy(_arg))
            elif xp.__name__ == "torch":
                if _arg.requires_grad:
                    _arg = _arg.detach()
                if _arg.is_cpu:
                    _arg = tf.convert_to_tensor(_arg)
                elif tf_cuda_is_available:
                    _arg = tf.experimental.dlpack.from_dlpack(
                        torch.utils.dlpack.to_dlpack(_arg)
                    )
                else:
                    _arg = tf.convert_to_tensor(_arg.numpy(force=True))
        args[n] = _arg

    return args


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
        args = _to_numpy(*args)
    elif array_interface.__name__ == "cupy":
        args = _to_cupy(*args, device=device)
    elif array_interface.__name__ == "torch":
        args = _to_torch(*args, device=device)

    if len(args) == 1:
        return args[0]

    return tuple(args)
