"""Rewrite Sphinx multiversion switcher metadata across the published site."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


#  Copyright (c) 2024-2026. KU Leuven

# A published version directory is named for the release tag that built it, the
# same shape the pipeline's tag rule accepts.
_VERSION_DIR = re.compile(r"^v\d+\.\d+\.\d+([-.].+)?$")


def _release_order(name: str) -> tuple[tuple[int, ...], str]:
    """Sort key placing the newest release first."""
    return tuple(int(part) for part in re.findall(r"\d+", name)[:3]), name


def rewrite_switchers(site_root: Path, base_url: str, root_ref: str) -> None:
    """Rewrite switcher metadata for every version published under ``site_root``.

    The site holds one directory per release plus a copy of ``root_ref`` at the
    top level. Every one of those copies is given the same list, so a reader on
    an older version can still reach one released after it.

    Raises:
        SystemExit: If ``root_ref`` names no directory under ``site_root``.
    """
    normalized_base_url = base_url.rstrip("/") + "/"
    version_dirs = sorted(
        (
            path.name
            for path in site_root.iterdir()
            if path.is_dir() and _VERSION_DIR.match(path.name)
        ),
        key=_release_order,
        reverse=True,
    )

    if root_ref not in version_dirs:
        raise SystemExit(
            f"Expected root ref {root_ref!r} among the published versions, "
            f"got {version_dirs!r}"
        )

    versions = []
    for name in version_dirs:
        url = (
            normalized_base_url if name == root_ref else f"{normalized_base_url}{name}/"
        )
        entry = {"name": name, "version": name, "url": url}
        if name == root_ref:
            entry["preferred"] = True
        versions.append(entry)

    payload = json.dumps(versions, indent=2)
    for copy in (site_root, *(site_root / name for name in version_dirs)):
        target = copy / "_static" / "switcher.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")


def main() -> None:
    """Rewrite switcher.json across the published site."""
    site_root = Path(os.environ["STATIC_SITE_DIR"])
    base_url = os.environ["DOCS_BASE_URL"]
    root_ref = os.environ["SMV_ROOT_REF"]
    rewrite_switchers(site_root, base_url, root_ref)


if __name__ == "__main__":
    main()
