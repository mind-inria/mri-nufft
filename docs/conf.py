"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = "mri-nufft"
copyright = "2022, Pierre-Antoine Comby"
author = "Pierre-Antoine Comby"

release = version("mri-nufft")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# generate autosummary even if no references
autosummary_generate = True
# autosummary_imported_members = True
autodoc_inherit_docstrings = True

napoleon_include_private_with_doc = True

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../examples/"],
    "filename_pattern": "/example_",
    "ignore_pattern": r"/(__init__|conftest)\.py",
    "reference_url": {
        "numpy": "http://docs.scipy.org/doc/numpy-1.9.1",
        "scipy": "http://docs.scipy.org/doc/scipy-0.17.0/reference",
    },
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_context = {"default_mode": "light"}
