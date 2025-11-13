from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx_gallery.notebook import add_code_cell, add_markdown_cell

import os
import json


def setup_colab_link_getter(app, pagename, templatename, context, doctree):
    """Add a function to the HTML context to get the Colab link for a notebook."""

    def get_colab_link() -> str:
        """Assume that the notebook path is the same as the pagename"""
        return f"https://colab.research.google.com/github/mind-inria/mri-nufft/blob/colab-examples/examples/{pagename}.ipynb"

    context["get_colab_link"] = get_colab_link


def setup(app):
    app.connect("html-page-context", setup_colab_link_getter)

    return {
        "version": "0.4",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
