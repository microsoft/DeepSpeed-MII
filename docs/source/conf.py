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
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx-prompt',
    'sphinxcontrib.autodoc_pydantic',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/',
               None),
}
intersphinx_disabled_domains = ['std']

# sphinx_autodoc_typehints config
typehints_defaults = "braces"

# autodoc_pyandtic config
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_field_signature_prefix = ' '
autodoc_pydantic_model_signature_prefix = 'class'
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_summary_list_order = 'bysource'
autodoc_pydantic_model_member_order = 'bysource'
autodoc_pydantic_field_list_validators = False

# sphinx_copybutton config
copybutton_prompt_text = r">>> |\$ |\(.venv\) \$ "
copybutton_prompt_is_regexp = True

#autodoc_mock_imports = ["deepspeed", "torch"]
autodoc_member_order = 'bysource'
autosummary_generate = True

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "logo_only": True,
}
html_logo = "../images/mii-dark.svg"
logo_only = True

# -- Options for EPUB output
epub_show_urls = 'footnote'
