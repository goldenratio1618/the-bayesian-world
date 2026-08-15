"""Authoritative structured-format guides bundled into every Luna workspace."""

from __future__ import annotations

from pathlib import Path

from ..paths import asset_root


STRUCTURED_FORMAT_GUIDES = (
    "docs/structured_formats/README.md",
    "docs/structured_formats/PMDL.md",
    "docs/structured_formats/PMDL_INTERFACES.md",
    "docs/structured_formats/STATIC_PART.md",
    "docs/structured_formats/MODEL_INSTANCE.md",
    "docs/structured_formats/CONTRAPTION.md",
    "docs/structured_formats/CONTROL.md",
    "docs/structured_formats/VERIFICATION.md",
    "docs/structured_formats/DETERMINISTIC_INGESTION.md",
    "docs/structured_formats/TRIANGLE_MESH.md",
    "docs/structured_formats/SHAPE_ARTIFACT.md",
    "docs/structured_formats/OPTICAL_MATERIAL.md",
    "docs/structured_formats/OPTICAL_SENSOR.md",
    "docs/structured_formats/OPTICAL_SCENE.md",
    "docs/structured_formats/OPTICAL_OBSERVATION.md",
    "docs/structured_formats/RECONSTRUCTION_STATE.md",
    "docs/structured_formats/OPTICAL_WORKFLOWS.md",
    "docs/structured_formats/RENDER_BUNDLE.md",
)


def structured_format_guides(root: str | Path | None = None) -> tuple[Path, ...]:
    """Resolve every required guide and fail closed if the asset tree is partial."""

    base = Path(root).resolve() if root is not None else asset_root().resolve()
    paths = tuple(base / relative for relative in STRUCTURED_FORMAT_GUIDES)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        relative = ", ".join(
            path.relative_to(base).as_posix() if path.is_relative_to(base) else str(path)
            for path in missing
        )
        raise FileNotFoundError(
            "structured-format documentation is incomplete; missing " + relative
        )
    return paths


__all__ = ["STRUCTURED_FORMAT_GUIDES", "structured_format_guides"]
