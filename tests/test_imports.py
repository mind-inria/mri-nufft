"""Check that importing mri-nufft leaves the heavy optional dependencies alone.

The check only bites where these dependencies are installed. The NUFFT backends
that are imported to check their availability (finufft, cupy, tensorflow...) are
not listed.
"""

import subprocess
import sys

import pytest

HEAVY = ["torch", "pylops", "pyproximal", "matplotlib", "sigpy"]
MODULES = [
    "mrinufft",
    "mrinufft.operators",
    "mrinufft.trajectories",
    "mrinufft.io",
    "mrinufft.display",
    "mrinufft.extras",
    "mrinufft.density",
]
ALLOWED = {"mrinufft.display": ["matplotlib"]}


@pytest.mark.parametrize("module", MODULES)
def test_lazy_imports(module):
    heavy = [name for name in HEAVY if name not in ALLOWED.get(module, [])]
    code = (
        f"import sys, {module}; "
        f"print(*(name for name in {heavy!r} if name in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert not loaded, f"`import {module}` imports {loaded}"


@pytest.mark.parametrize("module", MODULES)
def test_all(module):
    exec(f"from {module} import *", {})
