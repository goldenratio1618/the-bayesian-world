"""Filesystem closure loader for strict ``contraption-4`` bundles."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path, PureWindowsPath
from typing import Any

from .catalog.instantiations import PartInstantiationRegistry
from .catalog.procurement import ProcurementRegistry
from .catalog.interfaces import load_interface_catalog
from .control import ControlSpec, control_digest, load_control
from .physics.dsl import ModelRegistry
from .physics.resolved import ResolvedAssembly, resolve_assembly
from .physics.specs import ContraptionSpec
from .verification import VerificationProgram, load_verification


class ContraptionLoadError(ValueError):
    """A contraption bundle path, artifact, or closure is invalid."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContraptionLoadError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ContraptionLoadError(f"non-finite JSON number {value!r} is forbidden")


def _load_manifest(path: Path) -> ContraptionSpec:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContraptionLoadError(f"cannot read contraption manifest {path}: {exc}") from exc
    try:
        data = json.loads(
            source,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ContraptionLoadError:
        raise
    except json.JSONDecodeError as exc:
        raise ContraptionLoadError(
            f"invalid contraption JSON in {path}: {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}"
        ) from exc
    if not isinstance(data, Mapping):
        raise ContraptionLoadError("contraption manifest must contain a JSON object")
    try:
        return ContraptionSpec.from_dict(data)
    except Exception as exc:
        raise ContraptionLoadError(f"invalid contraption manifest {path}: {exc}") from exc


def _relative_path(value: str, context: str) -> Path:
    candidate = Path(value)
    if (
        not value
        or candidate.is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        raise ContraptionLoadError(f"{context} must be a non-empty relative path")
    return candidate


def _catalog_roots(manifest: Path, specification: ContraptionSpec) -> tuple[Path, ...]:
    roots: list[Path] = []
    for index, link in enumerate(specification.catalogs):
        relative = _relative_path(link.path, f"catalogs[{index}].path")
        candidate = (manifest.parent / relative).resolve()
        if not candidate.is_dir():
            raise ContraptionLoadError(
                f"catalogs[{index}].path does not resolve to an existing directory: {candidate}"
            )
        roots.append(candidate)
    duplicates = sorted(
        str(root) for root in set(roots) if roots.count(root) > 1
    )
    if duplicates:
        raise ContraptionLoadError(
            "contraption catalog roots must be unique after canonical resolution: "
            + ", ".join(duplicates)
        )
    return tuple(roots)


def _contained_artifact(manifest: Path, value: str, context: str) -> Path:
    relative = _relative_path(value, context)
    bundle_root = manifest.parent.resolve()
    candidate = (bundle_root / relative).resolve()
    try:
        candidate.relative_to(bundle_root)
    except ValueError as exc:
        raise ContraptionLoadError(
            f"{context} escapes the contraption bundle after canonical resolution"
        ) from exc
    if not candidate.is_file():
        raise ContraptionLoadError(
            f"{context} does not resolve to an existing regular file: {candidate}"
        )
    return candidate


def _load_catalogs(
    roots: tuple[Path, ...],
) -> tuple[ModelRegistry, PartInstantiationRegistry]:
    models = ModelRegistry()
    try:
        for root in roots:
            interfaces = load_interface_catalog(root)
            models.load_directory(root, interfaces=interfaces)
        values = []
        procurement_records = []
        for root in roots:
            registry = PartInstantiationRegistry.load_catalog(
                root,
                models=models,
                validate_procurement=False,
            )
            values.extend(registry.values())
            procurement_records.extend(registry.procurement.values())
        instantiations = PartInstantiationRegistry(
            values,
            procurement=ProcurementRegistry(procurement_records),
        )
        instantiations.validate_models(models)
    except Exception as exc:
        raise ContraptionLoadError(f"catalog closure validation failed: {exc}") from exc
    return models, instantiations


def _load_controllers(
    manifest: Path, specification: ContraptionSpec
) -> dict[str, ControlSpec]:
    result: dict[str, ControlSpec] = {}
    for link in specification.controllers:
        path = _contained_artifact(
            manifest,
            link.program.path,
            f"controller {link.id!r} program.path",
        )
        try:
            program = load_control(path)
        except Exception as exc:
            raise ContraptionLoadError(
                f"cannot load controller {link.id!r} from {path}: {exc}"
            ) from exc
        actual = control_digest(program)
        if actual != link.program.sha256:
            raise ContraptionLoadError(
                f"controller {link.id!r} canonical content hash mismatch: expected "
                f"{link.program.sha256}, got {actual}"
            )
        result[link.id] = program
    return result


def _load_verifications(
    manifest: Path, specification: ContraptionSpec
) -> dict[str, VerificationProgram]:
    result: dict[str, VerificationProgram] = {}
    for link in specification.verifications:
        path = _contained_artifact(
            manifest,
            link.program.path,
            f"verification {link.id!r} program.path",
        )
        try:
            program = load_verification(path)
        except Exception as exc:
            raise ContraptionLoadError(
                f"cannot load verification {link.id!r} from {path}: {exc}"
            ) from exc
        if program.sha256 != link.program.sha256:
            raise ContraptionLoadError(
                f"verification {link.id!r} canonical content hash mismatch: expected "
                f"{link.program.sha256}, got {program.sha256}"
            )
        result[link.id] = program
    return result


def load_contraption(path: str | Path) -> ResolvedAssembly:
    """Load, hash-check, resolve, and compile one contraption-4 manifest."""

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise ContraptionLoadError(
            f"contraption path must be an existing manifest file: {manifest}"
        )
    specification = _load_manifest(manifest)
    models, instantiations = _load_catalogs(_catalog_roots(manifest, specification))
    controllers = _load_controllers(manifest, specification)
    verifications = _load_verifications(manifest, specification)
    try:
        return resolve_assembly(
            specification,
            instantiations,
            models,
            controller_specs=controllers,
            verification_specs=verifications,
        )
    except Exception as exc:
        if isinstance(exc, ContraptionLoadError):
            raise
        raise ContraptionLoadError(
            f"contraption closure resolution failed for {manifest}: {exc}"
        ) from exc


__all__ = ["ContraptionLoadError", "load_contraption"]
