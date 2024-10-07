"""TEST CONFIGURATION.

This module contains methods for configuring the testing of the example
scripts.

:Author: Pierre-Antoine Comby

Notes
-----
Based on:
https://stackoverflow.com/questions/56807698/how-to-run-script-as-pytest-test

"""

import runpy
import sys
from pathlib import Path

import matplotlib as mpl
import pytest

mpl.use("agg")


def pytest_collect_file(path, parent):
    """Pytest hook.

    Create a collector for the given path, or None if not relevant.
    The new node needs to have the specified parent as parent.
    """
    p = Path(path)
    if p.suffix == ".py" and "example" in p.name:
        return Script.from_parent(parent, path=p, name=p.name)


class Script(pytest.File):
    """Script files collected by pytest."""

    def collect(self):
        """Collect the script as its own item."""
        yield ScriptItem.from_parent(self, name=self.name)


class ScriptItem(pytest.Item):
    """Item script collected by pytest."""

    def runtest(self):
        """Run the script as a test."""
        sys.path.insert(0, str(self.path.parent))
        runpy.run_path(str(self.path))

    def repr_failure(self, excinfo):
        """Return only the error traceback of the script."""
        excinfo.traceback = excinfo.traceback.cut(path=self.path)
        return super().repr_failure(excinfo)
