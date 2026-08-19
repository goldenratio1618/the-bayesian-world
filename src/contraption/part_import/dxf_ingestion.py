"""Strict, lossless group-pair extraction for bounded ASCII DXF drawings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


class DeterministicDxfError(ValueError):
    """Raised when a DXF drawing is unsupported, unsafe, or malformed."""


@dataclass(frozen=True, slots=True)
class DXFLimits:
    max_bytes: int = 4 * 1024 * 1024
    max_pairs: int = 250_000
    max_value_characters: int = 16_384
    max_sections: int = 64

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_pairs", "max_value_characters", "max_sections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DeterministicDxfError(f"{name} must be a positive integer")


_CODE = re.compile(r"^[+-]?\d+$")
_UNITS = {
    0: "unitless",
    1: "inch",
    2: "foot",
    3: "mile",
    4: "millimetre",
    5: "centimetre",
    6: "metre",
    7: "kilometre",
    8: "microinch",
    9: "mil",
    10: "yard",
    11: "angstrom",
    12: "nanometre",
    13: "micrometre",
    14: "decimetre",
    15: "decametre",
    16: "hectometre",
    17: "gigametre",
    18: "astronomical_unit",
    19: "light_year",
    20: "parsec",
    21: "us_survey_foot",
    22: "us_survey_inch",
    23: "us_survey_yard",
    24: "us_survey_mile",
}


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _safe_value(value: str, *, index: int, limits: DXFLimits) -> str:
    if len(value) > limits.max_value_characters:
        raise DeterministicDxfError(f"DXF value at pair {index} exceeds the length limit")
    normalized = unicodedata.normalize("NFKC", value)
    if normalized != value:
        raise DeterministicDxfError(
            f"DXF value at pair {index} is not already NFKC-normalized"
        )
    if any(unicodedata.category(char) in {"Cc", "Cf"} for char in normalized):
        raise DeterministicDxfError(f"DXF value at pair {index} contains control characters")
    return value


def extract_dxf(
    source: str | Path,
    *,
    limits: DXFLimits = DXFLimits(),
) -> dict[str, Any]:
    """Return every semantic DXF group pair plus bounded structural evidence.

    The extractor intentionally does not invent an extrusion or substitute an
    outline bounding box. It preserves all ASCII group codes and values so a
    later manufacturing-specific DXF consumer can revalidate exact entities.
    """

    path = Path(source).resolve()
    if not path.is_file() or path.is_symlink():
        raise DeterministicDxfError(f"DXF source is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > limits.max_bytes:
        raise DeterministicDxfError(
            f"DXF size must be between 1 and {limits.max_bytes} bytes"
        )
    payload = path.read_bytes()
    if len(payload) != size:
        raise DeterministicDxfError("DXF source changed while being read")
    if payload.startswith(b"AutoCAD Binary DXF"):
        raise DeterministicDxfError("binary DXF is not supported; provide ASCII DXF")
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise DeterministicDxfError("DXF must be strict UTF-8/ASCII text") from exc
    lines = text.splitlines()
    if not lines or len(lines) % 2:
        raise DeterministicDxfError("DXF must contain complete code/value line pairs")
    pair_count = len(lines) // 2
    if pair_count > limits.max_pairs:
        raise DeterministicDxfError("DXF exceeds the group-pair safety limit")
    pairs: list[dict[str, Any]] = []
    raw_pairs: list[tuple[int, str]] = []
    for index in range(pair_count):
        code_text = lines[index * 2].strip()
        if _CODE.fullmatch(code_text) is None:
            raise DeterministicDxfError(f"DXF pair {index} has an invalid group code")
        code = int(code_text)
        if code < -5 or code > 1071:
            raise DeterministicDxfError(f"DXF pair {index} group code is out of range")
        value = _safe_value(lines[index * 2 + 1], index=index, limits=limits)
        pairs.append({"code": code, "value": value})
        raw_pairs.append((code, value))
    if raw_pairs[-1] != (0, "EOF"):
        raise DeterministicDxfError("DXF must terminate with the 0/EOF group pair")

    sections: list[str] = []
    current: str | None = None
    entity_types: Counter[str] = Counter()
    insunits: int | None = None
    index = 0
    while index < len(raw_pairs):
        code, value = raw_pairs[index]
        if code == 0 and value == "SECTION":
            if current is not None or index + 1 >= len(raw_pairs) or raw_pairs[index + 1][0] != 2:
                raise DeterministicDxfError("DXF has a malformed or nested SECTION")
            current = raw_pairs[index + 1][1]
            if not current or current in sections:
                raise DeterministicDxfError("DXF section names must be nonempty and unique")
            sections.append(current)
            if len(sections) > limits.max_sections:
                raise DeterministicDxfError("DXF exceeds the section-count safety limit")
            index += 2
            continue
        if code == 0 and value == "ENDSEC":
            if current is None:
                raise DeterministicDxfError("DXF ENDSEC appears outside a section")
            current = None
        elif current == "ENTITIES" and code == 0:
            entity_types[value] += 1
        elif current == "HEADER" and code == 9 and value == "$INSUNITS":
            if index + 1 >= len(raw_pairs) or raw_pairs[index + 1][0] != 70:
                raise DeterministicDxfError("DXF $INSUNITS lacks an integer group 70 value")
            try:
                candidate = int(raw_pairs[index + 1][1].strip())
            except ValueError as exc:
                raise DeterministicDxfError("DXF $INSUNITS is not an integer") from exc
            if insunits is not None and candidate != insunits:
                raise DeterministicDxfError("DXF contains conflicting $INSUNITS values")
            insunits = candidate
        index += 1
    if current is not None:
        raise DeterministicDxfError("DXF final section is not closed")
    if "ENTITIES" not in sections:
        raise DeterministicDxfError("DXF has no ENTITIES section")

    canonical_pairs = (
        json.dumps(pairs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    units: dict[str, Any] | None = None
    if insunits is not None:
        units = {"code": insunits, "name": _UNITS.get(insunits, "unknown")}
    return {
        "format": "deterministic-dxf-extraction-1",
        "source": {
            "name": path.name,
            "media_type": "image/vnd.dxf",
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        "parser": {
            "name": "contraption-ascii-dxf",
            "version": "1",
            "lossless_group_pairs": True,
        },
        "pair_count": len(pairs),
        "pairs_sha256": _sha256(canonical_pairs),
        "sections": sections,
        "drawing_units": units,
        "entity_counts": dict(sorted(entity_types.items())),
        "pairs": pairs,
    }


__all__ = ["DXFLimits", "DeterministicDxfError", "extract_dxf"]
