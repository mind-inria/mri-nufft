#!/usr/bin/env python

import inspect
from docutils import nodes
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective  # <--- New Base Class

logger = logging.getLogger(__name__)

from mrinufft._utils import MethodRegister

import inspect
import re


def get_signature(func):
    """Safely extracts a clean function signature, without type annotations."""
    sig = str(inspect.signature(func))
    sig = sig.split("->")[0].strip()
    # iterative removal of bracketed expressions:
    while "[" in sig:
        sig = re.sub(r"\[[^\[\]]*\]", "", sig)
    sig = re.sub(r"\[.*?\]", "", sig)  # remove complex type annotation with brackets
    sig = re.sub(r":\s*[^,=\)]+", " ", sig)  # remove type annotations
    sig = re.sub(r"\s{2,}", " ", sig)  # collapse multiple spaces
    sig = re.sub(r"\s,", ",", sig)  # collapse multiple spaces
    sig = re.sub(r"\s=\s", "=", sig)  # collapse multiple spaces
    return sig


class AutoregistryDirective(SphinxDirective):  # <--- Inherit from SphinxDirective
    """Directive to list all entries in a specified sub-registry key."""

    required_arguments = 1
    has_content = False

    # We now have access to self.env, self.app, and self.parse_text_to_nodes

    def run(self):
        registry_key = self.arguments[0]

        try:
            # 1. Build rendering context
            registry_dict = MethodRegister.registry[registry_key]
            items_context = [
                {
                    "name": name,
                    "truename": (
                        func.__name__ if hasattr(func, "__name__") else str(func)
                    ),
                    "path": (
                        (func.__module__ + "." + func.__name__)
                        if hasattr(func, "__module__")
                        else str(func)
                    ),
                    "sig": get_signature(func),
                    # ... (rest of your context building logic) ...
                }
                for name, func in registry_dict.items()
            ]

            context = {
                "registry_key": registry_key,
                "items": items_context,
            }

            # 2. Render template to RST string
            # self.app is available because we inherit from SphinxDirective
            rst_content = self.env.app.builder.templates.render(
                "autoregistry.rst", context
            )

            # 3. THE FIX: Use the built-in utility
            # self.parse_text_to_nodes is a simple wrapper for nested_parse_to_nodes
            # that correctly passes the required RSTState (self.state).
            result_nodes = self.parse_text_to_nodes(rst_content)

            # The nodes are now fully parsed and contain pending_xref nodes
            # which Sphinx will automatically resolve later in the build process.
            # No manual env.resolve_references call is needed!
            return result_nodes

        except Exception as e:
            error_message = (
                f"Error resolving autoregistry for key '{registry_key}': {e}"
            )
            logger.error(error_message)
            return [nodes.literal_block(error_message, error_message)]


# Delete resolve_autoregistry function
def setup(app):
    """Main Sphinx extension entry point."""
    app.add_directive("autoregistry", AutoregistryDirective)

    return {
        "version": "0.4",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
