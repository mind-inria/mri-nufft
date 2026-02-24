"""Display configuration class."""

from __future__ import annotations
import itertools
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class displayConfig:
    """
    A container class used to share arguments related to display.

    The values can be updated either directy (and permanently) or temporarily by using
    a context manager.

    Examples
    --------
    >>> from mrinufft.trajectories.display import displayConfig
    >>> displayConfig.alpha
    0.2
    >>> with displayConfig(alpha=0.5):
            print(displayConfig.alpha)
    0.5
    >>> displayConfig.alpha
    0.2
    """

    alpha: float = 0.2
    """Transparency used for area plots, by default ``0.2``."""
    linewidth: float = 2
    """Width for lines or curves, by default ``2``."""
    pointsize: int = 10
    """Size for points used to show constraints, by default ``10``."""
    fontsize: int = 18
    """Font size for most labels and texts, by default ``18``."""
    small_fontsize: int = 14
    """Font size for smaller texts, by default ``14``."""
    nb_colors: int = 10
    """Number of colors to use in the color cycle, by default ``10``."""
    palette: str = "tab10"
    """Name of the color palette to use, by default ``"tab10"``.
    This can be any of the matplotlib colormaps, or a list of colors."""
    one_shot_color: str = "k"
    """Matplotlib color for the highlighted shot, by default ``"k"`` (black)."""
    one_shot_linewidth_factor: float = 2
    """Factor to multiply the linewidth of the highlighted shot, by default ``2``."""
    gradient_point_color: str = "r"
    """Matplotlib color for gradient constraint points, by default ``"r"`` (red)."""
    slewrate_point_color: str = "b"
    """Matplotlib color for slew rate constraint points, by default ``"b"`` (blue)."""

    def __init__(self, **kwargs: Any) -> None:  # noqa ANN401
        """Update the display configuration."""
        self.update(**kwargs)

    def update(self, **kwargs: Any) -> None:  # noqa ANN401
        """Update the display configuration."""
        self._old_values = {}
        for key, value in kwargs.items():
            self._old_values[key] = getattr(displayConfig, key)
            setattr(displayConfig, key, value)

    def reset(self) -> None:
        """Restore the display configuration."""
        for key, value in self._old_values.items():
            setattr(displayConfig, key, value)
        delattr(self, "_old_values")

    def __enter__(self) -> displayConfig:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: Any) -> None:  # noqa ANN401
        """Exit the context manager."""
        self.reset()

    @classmethod
    def get_colorlist(cls) -> list[str | NDArray]:
        """Extract a list of colors from a matplotlib palette.

        If the palette is continuous, the colors will be sampled from it.
        If its a categorical palette, the colors will be used in cycle.

        Parameters
        ----------
        palette : str, or list of colors, or matplotlib colormap
            Name of the palette to use, or list of colors, or matplotlib colormap.
        nb_colors : int, optional
            Number of colors to extract from the palette.
            The default is -1, and the value will be read from displayConfig.nb_colors.

        Returns
        -------
        colorlist : list of matplotlib colors.
        """
        if isinstance(cls.palette, str):
            cm = mpl.colormaps[cls.palette]
        elif isinstance(cls.palette, mpl.colors.Colormap):
            cm = cls.palette
        elif isinstance(cls.palette, list):
            cm = mpl.cm.ListedColormap(cls.palette)
        colorlist = []
        colors = getattr(cm, "colors", [])
        if 0 < len(colors) < cls.nb_colors:
            colorlist = [
                c for _, c in zip(range(cls.nb_colors), itertools.cycle(cm.colors))
            ]
        else:
            colorlist = cm(np.linspace(0, 1, cls.nb_colors))
        return colorlist
