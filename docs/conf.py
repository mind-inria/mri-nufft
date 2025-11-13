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


sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file

# import after updating the path.
from link_info import linkcode_resolve_file_suffix  # noqa: E402

# -- Project information -----------------------------------------------------

project = "mri-nufft"
copyright = "2022, MRI-NUFFT Contributors"
author = "MRI-NUFFT Contributors"


GITHUB_REPO = "https://github.com/mind-inria/mri-nufft"
GITHUB_VERSION = "master"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.video",
    "sphinx_gallery.gen_gallery",
    "sphinx_add_colab_link",
    "sphinx_autoregistry",
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
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented_params"

napoleon_include_private_with_doc = True
napolon_numpy_docstring = True
napoleon_use_admonition_for_references = True


pygments_style = "sphinx"
highlight_language = "python"

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "doc_module": ("mrinufft",),
    "backreferences_dir": "generated/gallery_backreferences",
    "reference_url": {"mrinufft": None},
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["generated/autoexamples"],
    "within_subsection_order": "ExampleTitleSortKey",
    "filename_pattern": "/example_",
    "ignore_pattern": r"(__init__|conftest|utils).py",
    "prefer_full_module": {r".*"},
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
    "parallel": True,
    "matplotlib_animations": (True, "mp4"),
    "first_notebook_cell": (
        "!pip install mri-nufft[cufinufft,finufft,gpunufft,extra,autodiff]"
    ),  # for binder and colab
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
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

html_css_files = [
    "custom.css",
]

html_theme_options = {
    "use_edit_page_button": True,
    "secondary_sidebar_items": {
        "generated/autoexamples/**": [
            "page-toc",
            "sg_download_links",
            "sg_launcher_links",
            "colab_link",
        ],
    },
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/mind-inria/mri-nufft",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
    ],
}

html_logo = "_static/logos/mri-nufft.png"
html_favicon = "_static/logos/mri-nufft-icon.png"
html_title = "MRI-nufft Documentation"
html_copy_source = False
html_show_sourcelink = False
html_context = {
    "github_user": "mind-inria",
    "github_repo": "mri-nufft",
    "github_version": GITHUB_VERSION,
    "doc_path": "docs/",
}


def linkcode_resolve(domain, info):
    file_suffix = linkcode_resolve_file_suffix(domain, info)
    if file_suffix is None:
        return None
    return f"{GITHUB_REPO}/blob/{GITHUB_VERSION}/" + file_suffix
