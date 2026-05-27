"""Version switcher helpers for the PyData Sphinx theme."""

from __future__ import annotations


#  Copyright (c) 2024-2026. KU Leuven


def configure_version_switcher(app, config) -> None:
    """Configure the theme switcher for the currently-built version."""
    switcher = dict(config.html_theme_options.get("switcher", {}))
    switcher.setdefault("json_url", "_static/switcher.json")
    version_match = config.smv_current_version or config.release or config.version or ""
    if version_match:
        switcher["version_match"] = str(version_match)
    config.html_theme_options["switcher"] = switcher
