"""General utility functions for MRI-NUFFT."""

from __future__ import annotations

import inspect
import logging
from inspect import cleandoc
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload
from tqdm.auto import tqdm

import numpy as np
from numpy.typing import DTypeLike, NDArray

from mrinufft._array_compat import get_array_module, Array

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


baselogger = logging.getLogger("mrinufft")
# avoid null handler duplications.
if not any(isinstance(h, logging.NullHandler) for h in baselogger.handlers):
    baselogger.addHandler(logging.NullHandler())


def set_log_level(level):
    """Set the log level of the mrinufft logger.

    Parameters
    ----------
    level: int | str
        Logging level, e.g. ``logging.DEBUG`` or ``"DEBUG"``.
    """
    baselogger.setLevel(level)


def sizeof_fmt(num, suffix="B"):
    """
    Return a number as a XiB format.

    Parameters
    ----------
    num: int
        The number to format
    suffix: str, default "B"
        The unit suffix

    References
    ----------
    https://stackoverflow.com/a/1094933
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


@overload
def proper_trajectory[S: np.floating](
    trajectory: NDArray[S], normalize: Literal["pi", "unit"] = "pi"
) -> NDArray[S]: ...
@overload
def proper_trajectory(
    trajectory: torch.Tensor, normalize: Literal["pi", "unit"] = "pi"
) -> torch.Tensor: ...
def proper_trajectory(
    trajectory: Array, normalize: Literal["pi", "unit"] = "pi"
) -> Array:
    """Normalize the trajectory to be used by NUFFT operators.

    Parameters
    ----------
    trajectory: np.ndarray
        The trajectory to normalize, it might be of shape (Nc, Ns, dim) of (Ns, dim)

    normalize: str
        if "pi" trajectory will be rescaled in [-pi, pi], if it was in [-0.5, 0.5]
        if "unit" trajectory will be rescaled in [-0.5, 0.5] if it was not [-0.5, 0.5]

    Returns
    -------
    new_traj: np.ndarray
        The normalized trajectory of shape (Nc * Ns, dim) or (Ns, dim) in -pi, pi
    """
    # flatten to a list of point
    xp = get_array_module(trajectory)  # check if the trajectory is a tensor
    new_traj = trajectory.reshape(-1, trajectory.shape[-1])
    max_abs_val = xp.max(xp.abs(new_traj))

    caller_logger = logger
    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if module and module.__name__ != __name__:
            caller_logger = logging.getLogger(module.__name__)
            break
    if normalize == "pi" and max_abs_val - 1e-4 < 0.5:
        caller_logger.warning(
            "Samples will be rescaled to [-pi, pi), assuming they were in [-0.5, 0.5)"
        )
        new_traj = new_traj * 2 * xp.pi
    elif normalize == "unit" and max_abs_val - 1e-4 > 0.5:
        caller_logger.warning(
            "Samples will be rescaled to [-0.5, 0.5), assuming they were in [-pi, pi)"
        )
        new_traj = new_traj * 1 / (2 * xp.pi)
    if normalize == "unit" and max_abs_val >= 0.5:
        new_traj = (new_traj + 0.5) % 1 - 0.5
    return new_traj


_SEE_ALSO_REGISTRY = """

.. seealso::

    This function is part of the `{registry}` registry, :py:func:`.get_{registry}`.
    You can find other registered functions in this registry below:

    .. autoregistry:: {registry}
"""

_GETTER_DOCSTRING = """
Get the {registry} function from its name.

Available methods are:

.. autoregistry:: {registry}

Parameters
----------
method_name: str
    The name of the method to retrieve.
*args, **kwargs:
    Arguments to pass to the method if it is callable.

Returns
-------
The method corresponding to the given name, or the result of calling it with the
provided arguments.

Raises
------
ValueError
    If the method name is not found in the registry.
"""


def _apply_docstring_subs[F: Callable[..., Any]](
    func: F, docstring_subs: dict[str, str], registry: str = ""
) -> F:
    if func.__doc__:
        docstring = cleandoc(func.__doc__)
        for key, sub in docstring_subs.items():
            docstring = docstring.replace(f"${{{key}}}", sub)
        func.__doc__ = docstring
        if registry:
            func.__doc__ += _SEE_ALSO_REGISTRY.format(registry=registry)
    return func


def _fill_doc[F: Callable[..., Any]](
    docstring_subs: dict[str, str],
) -> Callable[[F], F]:
    """Fill in docstrings with substitutions."""

    @wraps(_fill_doc)
    def wrapper(func: F) -> F:
        return _apply_docstring_subs(func, docstring_subs)

    return wrapper


class RegistryGetter[F](Protocol):
    """``get_<registry>`` function returned by :py:meth:`MethodRegister.make_getter`."""

    @overload
    def __call__(self, method_name: str, /) -> F: ...
    @overload
    def __call__(self, method_name: str, /, *args: Any, **kwargs: Any) -> Any: ...


class MethodRegister[F: Callable[..., Any]]:
    """
    A Decorator to register methods of the same type in dictionnaries.

    Parameters
    ----------
    name: str
        The  register name
    docstring_sub: dict[str,str]
        List of potential subsititutions to apply to the docstring.
    """

    registry_global: defaultdict[str, dict[str, Any]] = defaultdict(dict)

    def __init__(
        self, register_name: str, docstring_subs: dict[str, str] | None = None
    ):
        self.register_name = register_name
        if docstring_subs is None:
            docstring_subs = {}
        self.docstring_subs = docstring_subs

    @overload
    def __call__[G: Callable[..., Any]](self, method_name: G) -> G: ...
    @overload
    def __call__[G: Callable[..., Any]](self, method_name: str) -> Callable[[G], G]: ...
    def __call__(self, method_name: Any = None):
        """Register the function in the registry.

        It also substitute placeholder in docstrings.
        """

        def decorator(func):
            self.registry[method_name] = func
            func = _apply_docstring_subs(
                func, self.docstring_subs, registry=self.register_name
            )
            return func

        if callable(method_name):
            func = method_name
            method_name = func.__name__
            return decorator(func)
        else:
            return decorator

    def make_getter(self) -> RegistryGetter[F]:
        """Create a `get_{register_name}` function to get methods from the registry."""

        def getter(method_name, *args, **kwargs):
            try:
                method = self.registry[method_name]
            except KeyError as e:
                raise ValueError(
                    f"Unknown {self.register_name} method {method_name}."
                    " Available methods are \n"
                    f"{list(self.registry.keys())}"
                ) from e

            if args or kwargs:
                return method(*args, **kwargs)
            return method

        getter.__doc__ = _GETTER_DOCSTRING.format(registry=self.register_name)
        getter.__name__ = f"get_{self.register_name}"
        return getter

    @property
    def registry(self) -> dict[str, F]:
        """Get the registry dictionary."""
        return self.registry_global[self.register_name]


def _progressbar(progressbar: bool | tqdm, max_iter: int) -> tqdm:
    """Set progressbar for iterations."""
    if isinstance(progressbar, tqdm):
        progressbar.reset(max_iter)
    elif isinstance(progressbar, bool):
        progressbar = tqdm(range(max_iter), disable=not progressbar)
    else:
        raise ValueError("progressbar must be bool or tqdm instance")
    return progressbar
