"""Deterministic, evidence-preserving extraction from supported ECAD S-expressions.

The parser intentionally extracts only fields whose source format gives them an
unambiguous identity meaning.  It never derives a manufacturer or part number
from a symbol name, description, URL, or filename.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


class DeterministicEcadError(ValueError):
    """Raised when an ECAD source cannot be parsed safely and unambiguously."""


_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_NODES = 750_000
_MAX_DEPTH = 128
_MAX_VALUE_LENGTH = 16_384


@dataclass(frozen=True, slots=True)
class _Atom:
    value: str
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class _List:
    items: tuple["_Node", ...]
    line: int
    column: int


_Node = _Atom | _List


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_source(source: str | Path) -> tuple[Path, bytes, str]:
    path = Path(source).resolve()
    if not path.is_file() or path.is_symlink():
        raise DeterministicEcadError(f"ECAD source is missing or not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_SOURCE_BYTES:
        raise DeterministicEcadError(
            f"ECAD source size must be between 1 and {_MAX_SOURCE_BYTES} bytes: {path}"
        )
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DeterministicEcadError(f"ECAD source is not strict UTF-8: {path}") from exc
    return path, payload, text


def _lex(text: str) -> Iterable[str | _Atom]:
    index = 0
    line = 1
    column = 1
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            if char == "\n":
                line += 1
                column = 1
            else:
                column += 1
            index += 1
            continue
        if char in "()":
            yield char
            index += 1
            column += 1
            continue
        start_line, start_column = line, column
        if char == '"':
            index += 1
            column += 1
            value: list[str] = []
            while index < length:
                char = text[index]
                if char == '"':
                    index += 1
                    column += 1
                    break
                if char == "\\":
                    if index + 1 >= length:
                        raise DeterministicEcadError(
                            f"unterminated escape at line {line}, column {column}"
                        )
                    escaped = text[index + 1]
                    replacements = {
                        '"': '"',
                        "\\": "\\",
                        "n": "\n",
                        "r": "\r",
                        "t": "\t",
                    }
                    if escaped not in replacements:
                        raise DeterministicEcadError(
                            f"unsupported string escape at line {line}, column {column}"
                        )
                    value.append(replacements[escaped])
                    index += 2
                    column += 2
                    continue
                if char in "\r\n":
                    raise DeterministicEcadError(
                        f"literal newline in string at line {line}, column {column}"
                    )
                value.append(char)
                if len(value) > _MAX_VALUE_LENGTH:
                    raise DeterministicEcadError("ECAD string exceeds the safety limit")
                index += 1
                column += 1
            else:
                raise DeterministicEcadError(
                    f"unterminated string at line {start_line}, column {start_column}"
                )
            yield _Atom("".join(value), start_line, start_column)
            continue
        value_start = index
        while index < length and not text[index].isspace() and text[index] not in "()":
            if ord(text[index]) < 0x20:
                raise DeterministicEcadError(
                    f"control character at line {line}, column {column}"
                )
            index += 1
            column += 1
            if index - value_start > _MAX_VALUE_LENGTH:
                raise DeterministicEcadError("ECAD atom exceeds the safety limit")
        yield _Atom(text[value_start:index], start_line, start_column)


def _parse(text: str) -> _List:
    stack: list[tuple[list[_Node], int, int]] = []
    roots: list[_Node] = []
    nodes = 0
    for token in _lex(text):
        if token == "(":
            if len(stack) >= _MAX_DEPTH:
                raise DeterministicEcadError("ECAD nesting exceeds the safety limit")
            # The lexer does not attach a position to punctuation. The position
            # of the first child is the useful evidence location for a form.
            stack.append(([], 1, 1))
            continue
        if token == ")":
            if not stack:
                raise DeterministicEcadError("unmatched closing parenthesis in ECAD source")
            items, fallback_line, fallback_column = stack.pop()
            line = items[0].line if items else fallback_line
            column = items[0].column if items else fallback_column
            node = _List(tuple(items), line, column)
            nodes += 1
            if nodes > _MAX_NODES:
                raise DeterministicEcadError("ECAD source exceeds the node safety limit")
            (stack[-1][0] if stack else roots).append(node)
            continue
        assert isinstance(token, _Atom)
        nodes += 1
        if nodes > _MAX_NODES:
            raise DeterministicEcadError("ECAD source exceeds the node safety limit")
        (stack[-1][0] if stack else roots).append(token)
    if stack:
        raise DeterministicEcadError("unterminated list in ECAD source")
    if len(roots) != 1 or not isinstance(roots[0], _List):
        raise DeterministicEcadError("ECAD source must contain exactly one root list")
    return roots[0]


def _head(node: _List) -> str | None:
    return node.items[0].value if node.items and isinstance(node.items[0], _Atom) else None


def _forms(node: _List, name: str) -> tuple[_List, ...]:
    return tuple(
        child
        for child in node.items[1:]
        if isinstance(child, _List) and _head(child) == name
    )


def _scalar(node: _List, index: int, context: str) -> _Atom:
    if len(node.items) <= index or not isinstance(node.items[index], _Atom):
        raise DeterministicEcadError(f"{context} is missing its scalar value")
    return node.items[index]


def _value(node: _List, index: int, context: str) -> _Atom:
    value = _scalar(node, index, context)
    if not value.value or value.value != value.value.strip():
        raise DeterministicEcadError(f"{context} must contain a nonempty trimmed value")
    if any(ord(char) < 0x20 for char in value.value):
        raise DeterministicEcadError(f"{context} contains a control character")
    return value


def _evidence(atom: _Atom, locator: str, source_sha256: str) -> dict[str, Any]:
    return {
        "source_sha256": source_sha256,
        "locator": locator,
        "line": atom.line,
        "column": atom.column,
        "value_sha256": _sha256(atom.value.encode("utf-8")),
    }


def _fact(atom: _Atom, locator: str, source_sha256: str) -> dict[str, Any]:
    return {
        "value": atom.value,
        "evidence": _evidence(atom, locator, source_sha256),
    }


def _http_url(atom: _Atom) -> str | None:
    if len(atom.value) > 4096 or atom.value != atom.value.strip():
        return None
    parsed = urlsplit(atom.value)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(ord(char) < 0x20 for char in atom.value)
    ):
        return None
    return atom.value


def _unique_property(
    values: list[tuple[_Atom, str]],
    name: str,
    omissions: list[dict[str, str]],
) -> tuple[_Atom, str] | None:
    distinct = {atom.value for atom, _locator in values}
    if len(distinct) > 1:
        omissions.append(
            {
                "reason": f"conflicting explicit {name} properties",
                "locator": values[0][1],
            }
        )
        return None
    return values[0] if values else None


def _extract_kicad(root: _List, source_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if _head(root) != "kicad_symbol_lib":
        raise DeterministicEcadError("declared KiCad source is not a kicad_symbol_lib")
    records: list[dict[str, Any]] = []
    omissions: list[dict[str, str]] = []
    manufacturer_names = {"manufacturer", "mfr"}
    part_number_names = {"mpn", "manufacturer part number", "manufacturer_part_number"}
    for symbol_index, symbol in enumerate(_forms(root, "symbol")):
        symbol_atom = _value(symbol, 1, f"symbol[{symbol_index}]")
        base = f"symbol[{symbol_index}]"
        manufacturers: list[tuple[_Atom, str]] = []
        part_numbers: list[tuple[_Atom, str]] = []
        documents: list[dict[str, Any]] = []
        for property_index, prop in enumerate(_forms(symbol, "property")):
            name_atom = _value(prop, 1, f"{base}.property[{property_index}].name")
            value_atom = _scalar(prop, 2, f"{base}.property[{property_index}].value")
            locator = f"{base}.property[{property_index}]"
            normalized_name = name_atom.value.casefold()
            if normalized_name in manufacturer_names:
                if value_atom.value and value_atom.value == value_atom.value.strip():
                    manufacturers.append((value_atom, locator + ".value"))
            elif normalized_name in part_number_names:
                if value_atom.value and value_atom.value == value_atom.value.strip():
                    part_numbers.append((value_atom, locator + ".value"))
            elif normalized_name == "datasheet":
                url = _http_url(value_atom)
                if url is None:
                    if value_atom.value not in {"~", ""}:
                        omissions.append(
                            {"reason": "datasheet value is not an absolute HTTP(S) URL", "locator": locator}
                        )
                else:
                    documents.append(
                        {
                            "kind": "datasheet",
                            "url": url,
                            "evidence": _evidence(value_atom, locator + ".value", source_sha256),
                        }
                    )
        record: dict[str, Any] = {
            "catalog_identifier": {
                "namespace": "kicad_symbol",
                **_fact(symbol_atom, base + ".name", source_sha256),
            },
            "documents": documents,
        }
        manufacturer = _unique_property(manufacturers, "manufacturer", omissions)
        part_number = _unique_property(part_numbers, "manufacturer part number", omissions)
        if manufacturer is not None:
            record["manufacturer"] = _fact(manufacturer[0], manufacturer[1], source_sha256)
        if part_number is not None:
            record["manufacturer_part_number"] = _fact(part_number[0], part_number[1], source_sha256)
        records.append(record)
    return records, omissions


def _extract_librepcb(
    root: _List, source_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    root_kind = _head(root)
    if root_kind is None or not root_kind.startswith("librepcb_"):
        raise DeterministicEcadError("declared LibrePCB source does not have a librepcb_* root")
    entity_atom = _value(root, 1, root_kind)
    entity = {
        "namespace": root_kind,
        **_fact(entity_atom, "root.identifier", source_sha256),
    }
    omissions: list[dict[str, str]] = []
    documents: list[dict[str, Any]] = []
    for resource_index, resource in enumerate(_forms(root, "resource")):
        base = f"resource[{resource_index}]"
        title = _value(resource, 1, base + ".title")
        media_forms = _forms(resource, "mediatype")
        url_forms = _forms(resource, "url")
        media = _value(media_forms[0], 1, base + ".mediatype") if len(media_forms) == 1 else None
        if len(media_forms) > 1:
            omissions.append({"reason": "resource has multiple media types", "locator": base})
        for url_index, url_form in enumerate(url_forms):
            url_atom = _value(url_form, 1, f"{base}.url[{url_index}]")
            url = _http_url(url_atom)
            if url is None:
                omissions.append(
                    {"reason": "resource URL is not absolute HTTP(S)", "locator": f"{base}.url[{url_index}]"}
                )
                continue
            document: dict[str, Any] = {
                "kind": "datasheet" if media and media.value == "application/pdf" else "product_document",
                "title": _fact(title, base + ".title", source_sha256),
                "url": url,
                "evidence": _evidence(url_atom, f"{base}.url[{url_index}]", source_sha256),
            }
            if media is not None:
                document["media_type"] = _fact(media, base + ".mediatype", source_sha256)
            documents.append(document)
    records: list[dict[str, Any]] = []
    for part_index, part in enumerate(_forms(root, "part")):
        base = f"part[{part_index}]"
        mpn = _value(part, 1, base + ".manufacturer_part_number")
        manufacturer_forms = _forms(part, "manufacturer")
        record: dict[str, Any] = {
            "manufacturer_part_number": _fact(mpn, base + ".manufacturer_part_number", source_sha256)
        }
        if len(manufacturer_forms) == 1:
            manufacturer = _value(manufacturer_forms[0], 1, base + ".manufacturer")
            record["manufacturer"] = _fact(manufacturer, base + ".manufacturer", source_sha256)
        elif len(manufacturer_forms) > 1:
            omissions.append({"reason": "part has multiple manufacturer fields", "locator": base})
        records.append(record)
    return entity, records, documents, omissions


def extract_ecad(source: str | Path, *, source_format: str = "auto") -> dict[str, Any]:
    """Extract explicit identifiers and document URLs from KiCad or LibrePCB.

    ``source_format`` is ``auto``, ``kicad``, or ``librepcb``.  The result is
    JSON-compatible and every extracted identity fact contains a source hash
    and exact S-expression locator.  Missing facts stay absent.
    """

    if source_format not in {"auto", "kicad", "librepcb"}:
        raise DeterministicEcadError("source_format must be auto, kicad, or librepcb")
    path, payload, text = _read_source(source)
    root = _parse(text)
    root_kind = _head(root)
    detected = "kicad" if root_kind == "kicad_symbol_lib" else "librepcb" if root_kind and root_kind.startswith("librepcb_") else None
    if detected is None:
        raise DeterministicEcadError(f"unsupported ECAD root form {root_kind!r}")
    if source_format != "auto" and source_format != detected:
        raise DeterministicEcadError(
            f"declared {source_format} source has detected format {detected}"
        )
    digest = _sha256(payload)
    result: dict[str, Any] = {
        "format": "deterministic-ecad-extraction-1",
        "source": {
            "name": path.name,
            "kind": detected,
            "bytes": len(payload),
            "sha256": digest,
        },
    }
    if detected == "kicad":
        records, omissions = _extract_kicad(root, digest)
        result.update({"records": records, "documents": [], "omissions": omissions})
    else:
        entity, records, documents, omissions = _extract_librepcb(root, digest)
        result.update(
            {"entity": entity, "records": records, "documents": documents, "omissions": omissions}
        )
    return result


__all__ = ["DeterministicEcadError", "extract_ecad"]
