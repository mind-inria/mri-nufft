"""Tests for the deprecation helpers and the deprecation schedule.

``test_no_expired_deprecation`` checks the version from ``MRINUFFT_RELEASE`` if
set, else the installed one, so a release can be checked before tagging::

    MRINUFFT_RELEASE=1.7.0 pytest tests/test_deprecation.py
"""

import ast
import os
from pathlib import Path

import pytest
from packaging.version import Version

import mrinufft
from mrinufft._array_compat import DEEPINV_AVAILABLE
from mrinufft._deprecation import (
    MRINufftDeprecationWarning,
    deprecated,
    moved_attribute,
    renamed_parameter,
)

SRC = Path(mrinufft.__file__).parent


def _scheduled_removals():
    """Yield ``(location, since, removed_in)`` for each deprecation helper call."""
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "_deprecation.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if not isinstance(node, ast.Call):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            if "removed_in" not in kwargs:
                continue
            location = f"{path.relative_to(SRC.parent)}:{node.lineno}"
            values = [kwargs.get(k) for k in ("since", "removed_in")]
            if not all(isinstance(v, ast.Constant) for v in values):
                raise AssertionError(
                    f"{location}: `since` and `removed_in` must be string literals."
                )
            yield location, *(Version(v.value) for v in values)


def test_no_expired_deprecation():
    current = Version(os.environ.get("MRINUFFT_RELEASE", mrinufft.__version__))
    current = Version(current.base_version)
    scheduled = list(_scheduled_removals())
    malformed = [loc for loc, since, removed_in in scheduled if since >= removed_in]
    assert not malformed, f"`removed_in` must come after `since`: {malformed}"
    expired = [loc for loc, _, removed_in in scheduled if current >= removed_in]
    assert (
        not expired
    ), f"Deprecated objects due for removal in mri-nufft <= {current}: {expired}"


def test_deprecated_function():
    @deprecated(since="1.7.0", removed_in="1.9.0", alternative="`new`")
    def old(x):
        return 2 * x

    with pytest.warns(MRINufftDeprecationWarning, match=r"old` is deprecated.*`new`"):
        assert old(1) == 2


def test_deprecated_class():
    @deprecated(since="1.7.0", removed_in="1.9.0")
    class Old:
        pass

    with pytest.warns(MRINufftDeprecationWarning, match="removed in 1.9.0"):
        Old()


def test_renamed_parameter():
    @renamed_parameter("n", "n_bins", since="1.7.0", removed_in="1.9.0")
    def func(n_bins=1):
        return n_bins

    with pytest.warns(MRINufftDeprecationWarning, match="`n`.*Use `n_bins`"):
        assert func(n=3) == 3
    assert func(n_bins=4) == 4
    with pytest.raises(TypeError, match="both"):
        func(n=3, n_bins=4)


def test_moved_attribute():
    with pytest.warns(MRINufftDeprecationWarning, match="from mrinufft import"):
        obj = moved_attribute(
            "old.module", "set_log_level", "mrinufft", since="1.7.0", removed_in="1.9.0"
        )
    assert obj is mrinufft.set_log_level


@pytest.mark.skipif(not DEEPINV_AVAILABLE, reason="deepinv not available")
def test_legacy_deepinv_import_path():
    from mrinufft.operators import DeepInvPhyNufft

    with pytest.warns(MRINufftDeprecationWarning, match="mrinufft.operators.autodiff"):
        from mrinufft.operators.autodiff import DeepInvPhyNufft as legacy
    assert legacy is DeepInvPhyNufft
