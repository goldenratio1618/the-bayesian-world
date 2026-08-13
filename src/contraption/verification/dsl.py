"""Parsing and canonical serialization helpers for verification artifacts."""

from __future__ import annotations

from pathlib import Path

from .specs import (
    VerificationProgram,
    VerificationSpecError,
    load_verification,
    parse_verification,
)


def dump_verification(
    program: VerificationProgram,
    path: str | Path | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize a validated program, optionally writing normalized JSON."""

    if not isinstance(program, VerificationProgram):
        raise TypeError("program must be a VerificationProgram")
    if indent is None:
        text = program.canonical_json()
    else:
        import json

        text = json.dumps(
            program.to_dict(),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )
    if path is not None:
        destination = Path(path)
        destination.write_text(text + "\n", encoding="utf-8")
    return text


__all__ = [
    "VerificationProgram",
    "VerificationSpecError",
    "dump_verification",
    "load_verification",
    "parse_verification",
]
