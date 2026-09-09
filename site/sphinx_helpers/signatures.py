"""Signature-rendering helpers for Sphinx."""

from __future__ import annotations

import re


#  Copyright (c) 2024-2026. KU Leuven

# A default whose repr is not an expression: ``<factory>``, ``<function _times>``,
# ``<object object at 0x1049>``. Sphinx parses an argument list as Python, so one
# of these drops the whole signature onto the fallback path, which renders it as a
# single unbroken line with its annotations left as raw ``~module.Name`` text.
_UNREPRESENTABLE_DEFAULT = re.compile(r"<[^<>]*(?:<[^<>]*>[^<>]*)*>")


def elide_unrepresentable_defaults(
    app, what, name, obj, options, signature, return_annotation
):
    """Render a default that has no source form as ``...``."""
    if signature is None or "<" not in signature:
        return None
    return _UNREPRESENTABLE_DEFAULT.sub("...", signature), return_annotation
