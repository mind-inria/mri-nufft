"""Deprecation helpers for the public API.

A deprecated object keeps working, with a :class:`MRINufftDeprecationWarning`,
for at least two minor releases before it is removed.

``since`` and ``removed_in`` must be given as string literals:
``tests/test_deprecation.py`` reads them statically (optional dependencies and
lazy imports would hide them from a runtime registry), and fails once
``removed_in`` is reached.
"""

from __future__ import annotations

import functools
import importlib
import warnings
from collections.abc import Callable
from typing import Any, LiteralString, TypeVar, cast

from typing_extensions import deprecated as _deprecated

T = TypeVar("T")


class MRINufftDeprecationWarning(FutureWarning):
    """Warning for the deprecated parts of the MRI-NUFFT public API.

    It subclasses :class:`FutureWarning` rather than :class:`DeprecationWarning`,
    which Python hides outside of ``__main__`` and test runners.
    """


def _message(
    what: str, since: str, removed_in: str, alternative: str | None = None
) -> str:
    msg = (
        f"{what} is deprecated since mri-nufft {since} "
        f"and will be removed in {removed_in}."
    )
    if alternative is not None:
        msg += f" Use {alternative} instead."
    return msg


def _qualified_name(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def deprecated(
    *, since: str, removed_in: str, alternative: str | None = None
) -> Callable[[T], T]:
    """Mark a function or a class as deprecated.

    Calling the function, or instantiating the class, emits a
    :class:`MRINufftDeprecationWarning`.

    Parameters
    ----------
    since : str
        Version in which the deprecation was introduced.
    removed_in : str
        Version in which the object will be removed.
    alternative : str, optional
        What to use instead, e.g. ``"`mrinufft.trajectories.initialize_2D_radial`"``.
    """

    def decorator(obj: T) -> T:
        msg = _message(f"`{_qualified_name(obj)}`", since, removed_in, alternative)
        # The stubs require a LiteralString message, which a formatted one is not.
        return _deprecated(
            cast(LiteralString, msg), category=MRINufftDeprecationWarning
        )(obj)

    return decorator


def renamed_parameter(
    old: str, new: str, *, since: str, removed_in: str
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Accept the keyword argument ``old`` as a deprecated alias of ``new``.

    Parameters
    ----------
    old : str
        Former name of the parameter.
    new : str
        Current name of the parameter.
    since : str
        Version in which the parameter was renamed.
    removed_in : str
        Version in which ``old`` will stop being accepted.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = _qualified_name(func)
        msg = _message(f"Parameter `{old}` of `{name}`", since, removed_in, f"`{new}`")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(
                        f"{name}() got both `{old}` and its new name `{new}`."
                    )
                warnings.warn(msg, MRINufftDeprecationWarning, stacklevel=2)
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def moved_attribute(
    old_module: str, name: str, new_module: str, *, since: str, removed_in: str
) -> Any:
    """Resolve ``name`` from ``new_module`` and warn that its old path is deprecated.

    Meant to be returned from the module-level ``__getattr__`` (PEP 562) of
    ``old_module``.

    Parameters
    ----------
    old_module : str
        Module ``name`` used to be importable from, usually ``__name__``.
    name : str
        Name of the moved object.
    new_module : str
        Module ``name`` should now be imported from.
    since : str
        Version in which the object was moved.
    removed_in : str
        Version in which the old import path will stop working.
    """
    warnings.warn(
        _message(
            f"Importing `{name}` from `{old_module}`",
            since,
            removed_in,
            f"`from {new_module} import {name}`",
        ),
        MRINufftDeprecationWarning,
        stacklevel=3,
    )
    return getattr(importlib.import_module(new_module), name)
