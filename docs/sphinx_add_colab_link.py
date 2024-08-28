from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx_gallery.notebook import add_code_cell, add_markdown_cell

import os
import json


class ColabLinkNode(nodes.General, nodes.Element):
    """A custom docutils node to represent the Colab link."""


def visit_colab_link_node_html(self, node):
    self.body.append(node["html"])


def depart_colab_link_node_html(self, node):
    pass


class ColabLinkDirective(SphinxDirective):
    """Directive to insert a link to open a notebook in Google Colab."""

    has_content = True
    option_spec = {
        "needs_gpu": int,
    }

    def run(self):
        """Run the directive."""
        # Determine the path of the current .rst file
        rst_file_path = self.env.doc2path(self.env.docname)
        rst_file_dir = os.path.dirname(rst_file_path)

        # Determine the notebook file path assuming it is in the same directory as the .rst file
        notebook_filename = os.path.basename(rst_file_path).replace(".rst", ".ipynb")

        # Full path to the notebook
        notebook_full_path = os.path.join(rst_file_dir, notebook_filename)

        # Convert the full path back to a relative path from the repo root
        # repo_root = self.config.project_root_dir
        notebook_repo_relative_path = os.path.relpath(
            notebook_full_path, os.path.join(os.getcwd(), "docs")
        )

        # Generate the Colab URL based on GitHub repo information
        self.colab_url = f"https://colab.research.google.com/github/mind-inria/mri-nufft/blob/gh-pages/examples/{notebook_repo_relative_path}"

        # Create the HTML button or link
        self.html = f"""<div class="colab-button">
            <a href="{self.colab_url}" target="_blank">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" 
                alt="Open In Colab"/>
            </a>
        </div>
        """
        self.notebook_modifier(notebook_full_path, "\n".join(self.content))

        # Create the node to insert the HTML
        node = ColabLinkNode(html=self.html)
        return [node]

    def notebook_modifier(self, notebook_path, commands):
        """Modify the notebook to add a warning about GPU requirement."""
        with open(notebook_path) as f:
            notebook = json.load(f)
        if "cells" not in notebook:
            notebook["cells"] = []

        # Add a cell to install the required libraries at the position where we have
        # colab link
        idx = self.find_index_of_colab_link(notebook)
        code_lines = ["# Install libraries"]
        code_lines.append(commands)
        dummy_notebook_content = {"cells": []}
        add_code_cell(
            dummy_notebook_content,
            "\n".join(code_lines),
        )
        notebook["cells"][idx] = dummy_notebook_content["cells"][0]

        needs_GPU = self.options.get("needs_gpu", False)
        if needs_GPU:
            # Add a warning cell at the top of the notebook
            warning_template = "\n".join(
                [
                    "<div class='alert alert-{message_class}'>",
                    "",
                    "# Need GPU warning",
                    "",
                    "{message}",
                    "</div>",
                    self.html,
                ]
            )
            message_class = "warning"
            message = (
                "Running this mri-nufft example requires a GPU, and hence is NOT "
                "possible on binder currently We request you to kindly run this notebook "
                "on Google Colab by clicking the link below. Additionally, please make "
                "sure to set the runtime on Colab to use a GPU and install the below "
                "libraries before running."
            )
            idx = 0
        else:
            # Add a warning cell at the top of the notebook
            warning_template = "\n".join(
                [
                    "<div class='alert alert-{message_class}'>",
                    "",
                    "# Install libraries needed for Colab",
                    "",
                    "{message}",
                    "</div>",
                    self.html,
                ]
            )
            message_class = "info"
            message = (
                "The below installation commands are needed to be run only on "
                "Google Colab."
            )

        dummy_notebook_content = {"cells": []}
        add_markdown_cell(
            dummy_notebook_content,
            warning_template.format(message_class=message_class, message=message),
        )
        notebook["cells"] = (
            notebook["cells"][:idx]
            + dummy_notebook_content["cells"]
            + notebook["cells"][idx:]
        )

        # Write back updated notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False, indent=2)

    def find_index_of_colab_link(self, notebook):
        """Find the index of the cell containing the Colab link."""
        for idx, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "markdown" and ".. colab-link::" in "".join(
                cell.get("source", "")
            ):
                return idx
        return 0


def setup(app):
    """Set up the Sphinx extension."""
    app.add_node(
        ColabLinkNode, html=(visit_colab_link_node_html, depart_colab_link_node_html)
    )
    app.add_directive("colab-link", ColabLinkDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
