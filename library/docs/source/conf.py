# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import commonmark

from recommonmark.parser import CommonMarkParser
import recommonmark

import sphinx_rtd_theme

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(os.path.join(os.path.dirname(__file__), '..', 'HEA'))
sys.path.insert(0, basedir)

sys.setrecursionlimit(5000)

# -- Project information -----------------------------------------------------

project = 'HEA'
copyright = '2021, Anthony Correia'
author = 'Anthony Correia'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    "sphinx_rtd_theme",
    "recommonmark",
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


source_parsers = {
    '.md': CommonMarkParser,
}

source_suffix = ['.rst', '.md']

latex_engine = 'pdflatex'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'a4paper',
    'releasename':" ",
    # Sonny, Lenny, Glenn, Conny, Rejne, Bjarne and Bjornstrup
    # 'fncychap': '\\usepackage[Lenny]{fncychap}',
    'fncychap': '\\usepackage{fncychap}',
    'fontpkg': '\\usepackage{amsmath,amsfonts,amssymb,amsthm}',

    'figure_align':'htbp',
    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': r'''
        \usepackage{amsmath, amsfonts, amssymb, amsthm}
        \usepackage{graphicx}

        \usepackage{color}
        \usepackage{transparent}
        \usepackage{eso-pic}
    '''
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = "logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

