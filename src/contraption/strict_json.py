"""Shared strict JSON decoding for authoritative Contraption formats."""

from __future__ import annotations

import json
from typing import Any


def loads_strict_json(source: str | bytes | bytearray) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate keys and non-finite numbers."""

    if isinstance(source, (bytes, bytearray)):
        document = bytes(source).decode("utf-8")
    elif isinstance(source, str):
        document = source
    else:
        raise TypeError("strict JSON source must be text or bytes")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise json.JSONDecodeError(
                    f"duplicate JSON field {key!r}", document, 0
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise json.JSONDecodeError(
            f"non-finite JSON number {value!r} is forbidden", document, 0
        )

    return json.loads(
        document,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )
