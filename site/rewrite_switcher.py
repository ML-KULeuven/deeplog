"""Rewrite Sphinx multiversion switcher metadata from built output dirs."""

from __future__ import annotations

import json
import os
from pathlib import Path


#  Copyright (c) 2024-2026. KU Leuven


def rewrite_switchers(build_root: Path, base_url: str, root_ref: str) -> None:
    """Rewrite switcher metadata for all built version directories."""
    normalized_base_url = base_url.rstrip("/") + "/"
    version_dirs = sorted(
        path.name
        for path in build_root.iterdir()
        if path.is_dir() and (path / "index.html").exists()
    )

    if root_ref not in version_dirs:
        raise SystemExit(
            f"Expected root ref {root_ref!r} in built versions, got {version_dirs!r}"
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
    for name in version_dirs:
        target = build_root / name / "_static" / "switcher.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")


def main() -> None:
    """Rewrite switcher.json for each built version from the current build tree."""
    build_root = Path("site/build/html")
    base_url = os.environ["DOCS_BASE_URL"]
    root_ref = os.environ["SMV_ROOT_REF"]
    rewrite_switchers(build_root, base_url, root_ref)


if __name__ == "__main__":
    main()
