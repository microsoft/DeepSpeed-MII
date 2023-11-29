# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'DeepSpeed-MII'
copyright = '2023, Microsoft'
author = 'Microsoft'

with open("../../version.txt", "r") as f:
    release = f.readline().rstrip()

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/',
               None),
}
intersphinx_disabled_domains = ['std']

#autodoc_mock_imports = ["deepspeed", "torch"]
autosummary_generate = True

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
