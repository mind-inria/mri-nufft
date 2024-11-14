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
import coverage

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = "mri-nufft"
copyright = "2022, MRI-NUFFT Contributors"
author = "MRI-NUFFT Contributors"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx_add_colab_link",
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
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_include_private_with_doc = True
napolon_numpy_docstring = True
napoleon_use_admonition_for_references = True


pygments_style = "sphinx"
highlight_language = "python"

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "doc_module": "mrinufft",
    "backreferences_dir": "generated/gallery_backreferences",
    "reference_url": {"mrinufft": None},
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["generated/autoexamples"],
    "within_subsection_order": "ExampleTitleSortKey",
    "filename_pattern": "/example_",
    "ignore_pattern": r"(__init__|conftest|utils).py",
    "nested_sections": True,
    "binder": {
        "org": "mind-inria",
        "repo": "mri-nufft",
        "branch": "gh-pages",
        "binderhub_url": "https://mybinder.org",
        "dependencies": [
            "./binder/apt.txt",
            "./binder/environment.yml",
        ],
        "notebooks_dir": "examples",
        "use_jupyter_lab": True,
    },
    "parallel": 5,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/mind-inria/mri-nufft/",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
}

html_logo = "_static/logos/mri-nufft.png"
html_favicon = "_static/logos/mri-nufft-icon.png"
html_title = "MRI-nufft Documentation"
