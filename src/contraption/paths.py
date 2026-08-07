"""Locate immutable project assets in source and installed-wheel layouts."""

from __future__ import annotations

import os
from pathlib import Path
import sysconfig


def source_root() -> Path:
    """Return the checkout root implied by this module's source location."""

    return Path(__file__).resolve().parents[2]


def installed_data_root() -> Path:
    """Return the platform data-files location used by the built wheel."""

    return Path(sysconfig.get_path("data")) / "share" / "contraption"


def adjacent_data_root() -> Path:
    """Return the data-files root used by ``pip --target`` installations."""

    return Path(__file__).resolve().parents[1] / "share" / "contraption"


def _is_asset_root(path: Path) -> bool:
    return (
        (path / "data" / "taxonomy.json").is_file()
        and (path / "models" / "electrical" / "resistor.pmdl").is_file()
        and (path / "examples" / "scanner_robot" / "contraption.json").is_file()
    )


def asset_root() -> Path:
    """Find the complete read-only asset tree or fail with searched paths.

    ``CONTRAPTION_DATA_ROOT`` is an explicit deployment override. A checkout is
    preferred for editable development; normal wheels use ``share/contraption``
    under the interpreter's data prefix. Partial trees are never accepted.
    """

    candidates: list[Path] = []
    configured = os.environ.get("CONTRAPTION_DATA_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser().resolve())
    candidates.extend((source_root(), adjacent_data_root(), installed_data_root()))
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
        if _is_asset_root(candidate):
            return candidate
    searched = ", ".join(str(path) for path in unique)
    raise FileNotFoundError(
        "Contraption runtime assets are incomplete; expected taxonomy, models, "
        f"and scanner example under one root. Searched: {searched}"
    )


__all__ = ["adjacent_data_root", "asset_root", "installed_data_root", "source_root"]
