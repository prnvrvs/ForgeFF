# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ForgeFF"
copyright = "2026, Pranav Kumar"
author = "Pranav Kumar"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

from pathlib import Path

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

default_role = "code"

sphinx_gallery_conf = {
    "filename_pattern": r"/*\.py",
    "ignore_pattern": r".*/_common\.py$",
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["examples"],
    "within_subsection_order": "FileNameSortKey",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/forgeff-icon.png"
html_favicon = "_static/forgeff-icon.png"
html_theme_options = {
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html
    "github_url": "https://github.com/prnvrvs/ForgeFF",
}

autodoc_default_options = {
    "show-inheritance": True,
}


def _remove_extra_gallery_page(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    outdir = Path(app.builder.outdir)
    installation_html = outdir / "installation.html"
    if installation_html.exists():
        text = installation_html.read_text(encoding="utf-8")
        text = text.replace('href="examples/index.html"', 'href="example.html"')
        text = text.replace(
            '<p class="prev-next-title">Examples</p>',
            '<p class="prev-next-title">Example</p>',
        )
        installation_html.write_text(text, encoding="utf-8")
    for relpath in [
        Path("examples/index.html"),
        Path("_sources/examples/index.rst"),
    ]:
        target = outdir / relpath
        if target.exists():
            target.unlink()


def setup(app):
    app.connect("build-finished", _remove_extra_gallery_page)
