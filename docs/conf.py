# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "ƒVDB Reality Capture"
copyright = "Contributors to the OpenVDB Project"
author = "Contributors to the OpenVDB Project"

# Stable fvdb-core version shown in installation examples.
# Updated automatically by fvdb-core's devtools/update-doc-versions.sh during release.
fvdb_core_stable_version = "0.5.0"

version = fvdb_core_stable_version
release = fvdb_core_stable_version

rst_prolog = f"""\
.. |fvdb_core_version_pt210_cu128| replace:: {fvdb_core_stable_version}+pt210.cu128
.. |fvdb_core_version_pt210_cu130| replace:: {fvdb_core_stable_version}+pt210.cu130
"""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx.ext.napoleon", "myst_parser"]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Fix return-type in google-style docstrings
napoleon_custom_sections = [("Returns", "params_style")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = [".rst", ".md"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "wip"]

autodoc_default_options = {"undoc-members": "forward, extra_repr"}

# Mock compiled / GPU / heavy dependencies so Sphinx can introspect the
# Python API on build hosts that lack CUDA (e.g. Read the Docs).
#
# Everything that is pure-Python or installs cleanly from PyPI is pip-installed
# via docs/requirements.txt instead of mocked. Mocked modules become
# sphinx _MockObject instances, which do NOT support PEP 604 unions
# ("Foo | None" -> TypeError: unsupported operand type(s) for |: 'Foo' and
# 'NoneType'); installing the real package avoids that class of failure.
autodoc_mock_imports = [
    "_fvdb_cpp",
    "dlnr_lite",
    "fvdb",
    "open_clip",
    "point_cloud_utils",
    "pxr",
    "pye57",
    "sam2",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {"analytics_id": "G-60P7VJJ09C"}  # Google Analytics ID

html_context = {
    "display_github": True,
    "github_user": "openvdb",
    "github_repo": "fvdb-reality-capture",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    "tutorials/radiance_field_and_mesh_reconstruction_files",
    "tutorials/sensor_data_loading_and_manipulation_files",
    "_static",
]
html_css_files = [
    "css/custom.css",
]

myst_heading_anchors = 3

# -- Custom hooks ------------------------------------------------------------


def process_signature(app, what, name, obj, options, signature, return_annotation):
    if signature is not None:
        signature = signature.replace("fvdb::", "fvdb.")

    if return_annotation is not None:
        return_annotation = return_annotation.replace("fvdb::", "fvdb.")

    return signature, return_annotation


def setup(app):
    app.connect("autodoc-process-signature", process_signature)
