"""Check that importing mri-nufft leaves the heavy optional dependencies alone.

The check only bites where these dependencies are installed. The NUFFT backends
that are imported to check their availability (finufft, cupy, tensorflow...) are
not listed.
"""

import subprocess
import sys

import pytest

HEAVY = ["torch", "pylops", "pyproximal", "matplotlib", "sigpy"]


@pytest.mark.parametrize("module", ["mrinufft", "mrinufft.operators"])
def test_lazy_imports(module):
    code = (
        f"import sys, {module}; "
        f"print(*(name for name in {HEAVY!r} if name in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert not loaded, f"`import {module}` imports {loaded}"
