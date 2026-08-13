"""Preserve the repository asset hierarchy in built distributions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent
SHARE = Path("share/contraption")
ASSET_ROOTS = (
    Path("model_catalog"),
    Path("docs"),
    Path("prompts"),
    Path("web"),
    Path("assembled_contraptions"),
)


def asset_data_files() -> list[tuple[str, list[str]]]:
    grouped: dict[Path, list[str]] = defaultdict(list)
    grouped[SHARE].append("ARCHITECTURE.md")
    for asset_root in ASSET_ROOTS:
        for path in sorted((ROOT / asset_root).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                relative = path.relative_to(ROOT)
                grouped[SHARE / relative.parent].append(relative.as_posix())
    return [
        (target.as_posix(), sorted(paths))
        for target, paths in sorted(grouped.items(), key=lambda item: item[0].as_posix())
    ]


setup(data_files=asset_data_files())
