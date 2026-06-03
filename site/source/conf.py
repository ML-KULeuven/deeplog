"""Sphinx configuration for the DeepLog documentation site."""

# -- Project information -----------------------------------------------------

import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sphinx_helpers.notebooks import copy_example_notebooks
from sphinx_helpers.references import resolve_autoapi_xref
from sphinx_helpers.switcher import configure_version_switcher


project = "DeepLog"
copyright = "2026, KU Leuven"
author = "KU Leuven"
release = "3.0.2"

# -- General configuration ---------------------------------------------------

# Silence noisy tqdm progress bars during notebook execution (e.g., MNIST downloads)
os.environ.setdefault("TQDM_DISABLE", "1")

extensions = [
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
    "jupyter_sphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_multiversion",
]


autoapi_dirs = ["../../src/deeplog"]
autoapi_options = [
    "members",
    "undoc-members",
    "imported-members",  # <-- REQUIRED for cross-module types
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "show-type-annotations",
]
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_template_dir = "_autoapi_templates"

templates_path = ["_templates"]
exclude_patterns = ["_autoapi_templates", "**/*-checkpoint.ipynb"]

add_module_names = False
python_use_unqualified_type_names = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# autoapi can't resolve a handful of reference targets — most are Python 3.12
# generic-syntax artefacts (`T` from `def f[T](...)`, `F` TypeVars in engine.py)
# or typing-syntax artefacts (`Ellipsis` from `tuple[X, ...]`). None are real
# doc gaps; silence them so `-W -n` stays useful for catching real issues.
nitpick_ignore = [
    ("py:class", "Ellipsis"),
    # Generic/TypeVar parameters surfaced by Python 3.12 generics syntax.
    ("py:class", "F"),
    ("py:class", "foldr.T"),
    ("py:class", "as_tuple.T"),
    ("py:class", "parse_formula.T"),
    ("py:class", "parse_dimacs_cnf.T"),
    ("py:class", "EngineResult.F"),
    ("py:class", "_Step"),
    # Private types referenced from public-looking places (intentional).
    ("py:class", "_DeepLogCircuitNode"),
    (
        "py:class",
        "deeplog.formula.deeplogmodulefactory.deeplogmodulefactory._DeepLogCircuitNode",
    ),
    # numpy typing aliases that aren't in numpy's intersphinx inventory.
    ("py:class", "NDArray"),
    ("py:class", "numpy.typing.ArrayLike"),
    ("py:class", "numpy.typing.NDArray"),
    ("py:class", "np.object_"),
    # External/private implementation types referenced from docstrings.
    ("py:class", "klay.Circuit"),
    ("py:class", "_NetworkPredicate"),
    ("py:obj", "_LabelProbabilityPredicate"),
    ("py:class", "_StructureCast"),
    ("py:data", "_CAST_FUNCTIONS"),
]

myst_enable_extensions = [
    "dollarmath",
]

# -- Options for HTML output -------------------------------------------------

pygments_style = "sphinx"
html_title = "DeepLog"
html_theme = "pydata_sphinx_theme"
html_baseurl = os.environ.get("DOCS_BASE_URL", "")
html_static_path = ["_static"]
html_show_sphinx = False
html_show_sourcelink = False
html_permalinks = False  # Disable permalinks (the # symbol next to headers)
html_favicon = "_static/images/favicon.ico"
# nb_execution_cache_path = str(Path(__file__).parent.parent / "build")

graphviz_output_format = "svg"
viewcode_follow_imported_members = True

# Customize the theme (optional)
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "show_nav_level": 3,
    "show_toc_level": 3,
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": release,
    },
    "logo": {
        "text": "<b>DeepLog</b>",  # Name shown next to the logo
        "image_light": "_static/images/deeplog_fat.png",  # Light mode logo
        "image_dark": "_static/images/deeplog_white.png",  # Dark mode logo
    },
    # Show the version + light/dark mode toggles next to the icon links.
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ML-KULeuven/deeplog",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": True,
}

html_context = {"default_mode": "light"}


# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = os.environ.get("SMV_TAG_WHITELIST", r"^.*$")

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = os.environ.get("SMV_BRANCH_WHITELIST", r"^.*$")

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = os.environ.get("SMV_REMOTE_WHITELIST", r"^origin$")

# Pattern for released versions
smv_released_pattern = os.environ.get("SMV_RELEASED_PATTERN", r"^tags/.*$")

# Format for versioned output directories inside the build directory
smv_outputdir_format = os.environ.get("SMV_OUTPUTDIR_FORMAT", "{ref.name}")

# Branch/tag that should live at the root of the built docs.
smv_root_ref = os.environ.get("SMV_ROOT_REF", "main")

# Determines whether remote or local git branches/tags are preferred if their output dirs conflict
smv_prefer_remote_refs = os.environ.get("SMV_PREFER_REMOTE_REFS", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Add a custom CSS file if you'd like further styling
html_css_files = [
    "custom.css",
    "resources.css",
]

html_js_files = [
    "autoapi.js",
]

html_sidebars = {
    "*": [],
    "autoapi/**": ["search-field.html", "sidebar-nav-bs.html"],
    "tutorial": [],
    "tutorial/**": [],
}

nb_execution_mode = "auto"
nb_execution_timeout = int(os.environ.get("NB_EXECUTION_TIMEOUT", "300"))
nb_execution_raise_on_error = False


def setup(app):
    """Set up the build environment."""
    app.add_config_value("smv_root_ref", smv_root_ref, "env")
    app.connect("config-inited", configure_version_switcher)
    app.connect("builder-inited", copy_example_notebooks)
    app.connect("missing-reference", resolve_autoapi_xref)
