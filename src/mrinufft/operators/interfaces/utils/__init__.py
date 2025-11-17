"""Utilities for interfaces."""

from .gpu_utils import (
    get_maxThreadBlock,
    nvtx_mark,
)

__all__ = [
    "get_maxThreadBlock",
    "nvtx_mark",
]
