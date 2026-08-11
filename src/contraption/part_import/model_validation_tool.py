"""Isolated deterministic validation for modeling-agent catalog imports.

The tool validates either one PMDL draft or a complete catalog-relative bundle
under the current modeling workspace's ``candidate`` directory.  It never
imports or executes generated code.  A harness-owned context file outside the
writable workspace binds the workspace path and hashes every input/control
file; those hashes are checked both before and after model parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Sequence


CONTEXT_FILENAME = ".model-validator-context.json"
CALL_LOG_FILENAME = "validation-calls.jsonl"
CONTEXT_SCHEMA = "contraption.model-validator-context/v1"
RESULT_SCHEMA = "contraption.model-validation/v1"
EXCESSIVE_CALL_THRESHOLD = 5
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024


class ValidationToolError(ValueError):
    """Raised for a broken validator contract or workspace-integrity failure."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json_strict(path: Path) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationToolError(
                    f"{path.name}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationToolError(f"cannot read strict JSON {path.name}: {exc}") from exc


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValidationToolError(f"{label} must be a regular, non-symlink file: {path}")


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def write_validation_context(
    run_root: str | Path,
    protected_files: Sequence[str | Path],
) -> Path:
    """Write the harness-owned hash manifest used by validator subprocesses.

    ``protected_files`` must already exist below ``run_root``.  The context is
    placed beside the writable ``workspace`` directory, not inside it.
    """

    root = Path(run_root).resolve()
    workspace = root / "workspace"
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValidationToolError(f"workspace must be a regular directory: {workspace}")
    entries: list[dict[str, Any]] = []
    normalized: set[str] = set()
    for raw in protected_files:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        _regular_file(path, "protected input")
        resolved = path.resolve()
        if not _within(resolved, root):
            raise ValidationToolError(f"protected input escapes run root: {path}")
        relative = resolved.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in normalized:
            raise ValidationToolError(f"duplicate protected input path: {relative}")
        normalized.add(folded)
        payload = resolved.read_bytes()
        entries.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    context = {
        "schema": CONTEXT_SCHEMA,
        "workspace": "workspace",
        "candidate_directory": "candidate",
        "protected_files": sorted(entries, key=lambda item: item["path"]),
    }
    target = root / CONTEXT_FILENAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(root)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(context, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _context_for(workspace: Path) -> tuple[Path, dict[str, Any]]:
    workspace = workspace.resolve()
    run_root = workspace.parent
    context_path = run_root / CONTEXT_FILENAME
    _regular_file(context_path, "validator context")
    value = _load_json_strict(context_path)
    if not isinstance(value, dict):
        raise ValidationToolError("validator context must be a JSON object")
    expected_keys = {
        "schema",
        "workspace",
        "candidate_directory",
        "protected_files",
    }
    if set(value) != expected_keys:
        raise ValidationToolError(
            f"validator context keys differ: expected {sorted(expected_keys)}, got {sorted(value)}"
        )
    if value["schema"] != CONTEXT_SCHEMA:
        raise ValidationToolError(f"unsupported validator context schema: {value['schema']!r}")
    if value["workspace"] != "workspace" or (run_root / "workspace").resolve() != workspace:
        raise ValidationToolError("validator context is not bound to the current workspace")
    if value["candidate_directory"] != "candidate":
        raise ValidationToolError("validator candidate directory must be exactly 'candidate'")
    if not isinstance(value["protected_files"], list) or not value["protected_files"]:
        raise ValidationToolError("validator context has no protected input files")
    return run_root, value


def assert_workspace_integrity(workspace: str | Path) -> dict[str, Any]:
    """Re-hash all harness-protected inputs and fail on any drift."""

    workspace_path = Path(workspace).resolve()
    run_root, context = _context_for(workspace_path)
    checked: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(context["protected_files"]):
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise ValidationToolError(
                f"protected_files[{index}] must contain exactly path, bytes, and sha256"
            )
        relative_raw = raw["path"]
        if not isinstance(relative_raw, str) or "\\" in relative_raw:
            raise ValidationToolError(f"protected_files[{index}].path is unsafe")
        relative = PurePosixPath(relative_raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValidationToolError(f"protected_files[{index}].path is unsafe")
        name = relative.as_posix()
        if name.casefold() in seen:
            raise ValidationToolError(f"duplicate protected input entry: {name}")
        seen.add(name.casefold())
        path = run_root.joinpath(*relative.parts)
        _regular_file(path, f"protected input {name!r}")
        resolved = path.resolve()
        if not _within(resolved, run_root):
            raise ValidationToolError(f"protected input escaped run root: {name}")
        payload = resolved.read_bytes()
        expected_size = raw["bytes"]
        expected_hash = raw["sha256"]
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise ValidationToolError(f"invalid hash metadata for protected input {name}")
        actual_hash = sha256_bytes(payload)
        if len(payload) != expected_size or actual_hash != expected_hash:
            raise ValidationToolError(
                f"protected input integrity mismatch for {name}: "
                f"expected {expected_size} bytes sha256={expected_hash}, "
                f"got {len(payload)} bytes sha256={actual_hash}"
            )
        checked.append(name)
    return {"context": str(run_root / CONTEXT_FILENAME), "checked": checked}


def _candidate_path(workspace: Path, raw: str) -> tuple[Path, str]:
    if not raw or "\\" in raw or "\x00" in raw:
        raise ValidationToolError(f"unsafe candidate path: {raw!r}")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValidationToolError(f"unsafe candidate path: {raw!r}")
    if relative.parts[0] != "candidate" or len(relative.parts) < 2:
        raise ValidationToolError("candidate must be below candidate/")
    if relative.suffix != ".pmdl":
        raise ValidationToolError("candidate must have the .pmdl suffix")
    candidate_root = (workspace / "candidate").resolve()
    path = workspace.joinpath(*relative.parts)
    _regular_file(path, "candidate")
    resolved = path.resolve()
    if not _within(resolved, candidate_root):
        raise ValidationToolError(f"candidate escaped candidate/: {raw!r}")
    for parent in (path, *path.parents):
        if parent == workspace.parent:
            break
        if parent.is_symlink():
            raise ValidationToolError(f"candidate path contains a symlink: {raw!r}")
    return resolved, relative.as_posix()


def _read_call_log(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / CALL_LOG_FILENAME
    if not path.exists():
        return []
    _regular_file(path, "validation call log")
    values: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationToolError(
                f"{CALL_LOG_FILENAME}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict) or value.get("schema") != RESULT_SCHEMA:
            raise ValidationToolError(
                f"{CALL_LOG_FILENAME}:{line_number}: invalid validation result record"
            )
        values.append(value)
    return values


def validation_activity(workspace: str | Path) -> dict[str, Any]:
    """Summarize the observable validator log; it is telemetry, not authority."""

    values = _read_call_log(Path(workspace).resolve())
    successful = sum(value.get("valid") is True for value in values)
    calls = len(values)
    return {
        "logged_calls": calls,
        "successful_calls": successful,
        "failed_calls": calls - successful,
        "excessive_calls": calls > EXCESSIVE_CALL_THRESHOLD,
        "excessive_call_threshold": EXCESSIVE_CALL_THRESHOLD,
        "note": "workspace log is observability only; final host validation remains authoritative",
    }


def _append_call(workspace: Path, result: dict[str, Any]) -> None:
    path = workspace / CALL_LOG_FILENAME
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValidationToolError(f"validation call log is not a regular file: {path}")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_candidate(candidate: str, *, workspace: str | Path | None = None) -> dict[str, Any]:
    """Validate one candidate and return a deterministic, machine-readable report."""

    workspace_path = Path.cwd().resolve() if workspace is None else Path(workspace).resolve()
    prior_calls = _read_call_log(workspace_path)
    call_number = len(prior_calls) + 1
    issues: list[dict[str, str]] = []
    candidate_hash: str | None = None
    integrity_checked: list[str] = []
    try:
        before = assert_workspace_integrity(workspace_path)
        integrity_checked = before["checked"]
        candidate_path, candidate_name = _candidate_path(workspace_path, candidate)
        payload = candidate_path.read_bytes()
        if len(payload) > MAX_CANDIDATE_BYTES:
            raise ValidationToolError(
                f"candidate exceeds {MAX_CANDIDATE_BYTES} byte limit: {len(payload)} bytes"
            )
        candidate_hash = sha256_bytes(payload)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            issues.append(
                {
                    "severity": "error",
                    "code": "pmdl.encoding",
                    "path": candidate_name,
                    "message": f"candidate is not UTF-8: {exc}",
                }
            )
        else:
            try:
                if candidate_path.name == "interface.pmdl":
                    from ..catalog.interfaces import parse_interface

                    parse_interface(text, source_name=candidate_name)
                else:
                    from ..catalog.interfaces import load_default_interface_catalog
                    from ..physics.dsl import parse_model
                    from ..physics.validation import validate_model

                    model = parse_model(text, source_name=candidate_name)
                    report = validate_model(model, load_default_interface_catalog())
                    issues.extend(report.to_dict()["issues"])
            except Exception as exc:  # Parser/spec exceptions are rendered as diagnostics.
                issues.append(
                    {
                        "severity": "error",
                        "code": "pmdl.parse",
                        "path": candidate_name,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
        # Detect concurrent or attempted input mutation during parsing as well.
        assert_workspace_integrity(workspace_path)
    except ValidationToolError as exc:
        issues.append(
            {
                "severity": "error",
                "code": "validator.contract",
                "path": "$",
                "message": str(exc),
            }
        )

    issues = sorted(
        issues,
        key=lambda item: (
            item.get("path", ""),
            item.get("severity", ""),
            item.get("code", ""),
            item.get("message", ""),
        ),
    )
    valid = not any(item.get("severity") == "error" for item in issues)
    result = {
        "schema": RESULT_SCHEMA,
        "call_number": call_number,
        "candidate": candidate,
        "candidate_sha256": candidate_hash,
        "valid": valid,
        "issues": issues,
        "protected_files_checked": integrity_checked,
        "guidance": (
            "Candidate is valid; copy these exact bytes into the final structured response."
            if valid
            else "Correct every error and validate again. If repeated attempts fail, re-read the supplied constraints and examples instead of guessing."
        ),
    }
    _append_call(workspace_path, result)
    return result


def validate_bundle(bundle: str, *, workspace: str | Path | None = None) -> dict[str, Any]:
    """Validate the complete candidate import in a catalog overlay."""

    workspace_path = Path.cwd().resolve() if workspace is None else Path(workspace).resolve()
    call_number = len(_read_call_log(workspace_path)) + 1
    issues: list[dict[str, str]] = []
    checked: list[str] = []
    try:
        checked = assert_workspace_integrity(workspace_path)["checked"]
        if bundle != "candidate":
            raise ValidationToolError("--bundle must name the workspace candidate directory")
        candidate_root = (workspace_path / "candidate").resolve()
        if candidate_root.is_symlink() or not candidate_root.is_dir():
            raise ValidationToolError("candidate bundle must be a regular directory")
        files = [path for path in candidate_root.rglob("*") if path.is_file()]
        if not files:
            raise ValidationToolError("candidate bundle contains no files")
        if any(path.is_symlink() for path in candidate_root.rglob("*")):
            raise ValidationToolError("candidate bundle may not contain symlinks")
        if sum(path.stat().st_size for path in files) > MAX_CANDIDATE_BYTES:
            raise ValidationToolError("candidate bundle exceeds the byte limit")
        from .agents import ModelingAgent

        ModelingAgent.validate_artifacts(candidate_root)
        assert_workspace_integrity(workspace_path)
    except Exception as exc:
        issues.append(
            {
                "severity": "error",
                "code": "catalog.bundle",
                "path": bundle,
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
    result = {
        "schema": RESULT_SCHEMA,
        "call_number": call_number,
        "candidate": bundle,
        "candidate_sha256": None,
        "valid": not issues,
        "issues": issues,
        "protected_files_checked": checked,
        "guidance": (
            "Catalog import is valid; copy these exact bytes into the structured response."
            if not issues
            else "Correct every catalog error and validate the complete bundle again."
        ),
    }
    _append_call(workspace_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contraption-model-validate",
        description="Validate one PMDL draft or a complete catalog import bundle.",
    )
    parser.add_argument("candidate", nargs="?", help="relative candidate/**/*.pmdl path")
    parser.add_argument("--bundle", help="validate the complete candidate directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if bool(args.candidate) == bool(args.bundle):
            raise ValidationToolError("provide exactly one candidate path or --bundle candidate")
        result = validate_bundle(args.bundle) if args.bundle else validate_candidate(args.candidate)
    except Exception as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "valid": False,
            "issues": [
                {
                    "severity": "error",
                    "code": "validator.internal",
                    "path": "$",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
            "guidance": "The validator itself failed; do not claim the candidate is valid.",
        }
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if result.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
