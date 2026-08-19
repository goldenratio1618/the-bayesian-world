"""Deterministic, evidence-preserving procurement extraction.

This module deliberately recognizes only explicit component-input fields and a
small set of unambiguous identifier spellings.  It does not search the web,
infer a manufacturer from a URL/domain, or turn a general source URL into a
price/availability offer.  When no evidenced identifier is present it returns
no record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Any, TYPE_CHECKING
from urllib.parse import unquote, urlparse

from ..catalog.procurement import (
    ProcurementDocumentSpec,
    ProcurementEvidenceSpec,
    ProcurementIdentifierSpec,
    ProcurementLifecycleSpec,
    ProcurementProvisionSpec,
    ProcurementRecord,
    ProcurementRegistry,
    ProcurementSpecError,
)


if TYPE_CHECKING:
    from ..catalog.instantiations import StaticPartSpec


class ProcurementExtractionError(ValueError):
    """Component procurement evidence is malformed or ambiguous."""


@dataclass(frozen=True, slots=True)
class ProcurementTextFallbackConfig:
    """Explicit opt-in for non-agentic PDF-text identity extraction.

    The fallback is always a raw Chat Completions call to Luna at low reasoning
    effort.  It consumes only canonical deterministic extraction JSON and is
    disabled unless a caller supplies this configuration.
    """

    cache_directory: Path
    model: str = "gpt-5.6-luna"
    max_attempts: int = 3
    max_completion_tokens: int = 2_000
    api_key: str | None = None

    def __post_init__(self) -> None:
        cache = Path(self.cache_directory).expanduser().resolve()
        if not str(self.model).strip() or self.model != self.model.strip():
            raise ProcurementExtractionError("fallback model must be nonempty and trimmed")
        if "luna" not in self.model.casefold():
            raise ProcurementExtractionError("PDF procurement fallback must use a Luna model")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
            or self.max_attempts > 3
        ):
            raise ProcurementExtractionError("fallback max_attempts must be from 1 through 3")
        if (
            isinstance(self.max_completion_tokens, bool)
            or not isinstance(self.max_completion_tokens, int)
            or self.max_completion_tokens < 256
            or self.max_completion_tokens > 8_192
        ):
            raise ProcurementExtractionError(
                "fallback max_completion_tokens must be from 256 through 8192"
            )
        object.__setattr__(self, "cache_directory", cache)


HOST_PROCUREMENT_CONTEXT_FORMAT = "host-procurement-context-1"
HOST_PROCUREMENT_RECEIPT_FORMAT = "host-procurement-receipt-1"
HOST_PROCUREMENT_CONTEXT_FILENAME = ".host-procurement-context.json"
HOST_PROCUREMENT_RECEIPT_FILENAME = ".host-procurement-receipt.json"


_COMPACT_PART_NUMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+:-]{1,127}$")
_ITEM_SUFFIX = re.compile(
    r"(?:,\s*|\s+)item\s+(?P<item>[A-Za-z0-9][A-Za-z0-9_.-]*)\s*$",
    re.IGNORECASE,
)
_URL_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SLUG_RUN = re.compile(r"[^a-z0-9]+")
_PDF_MANUFACTURER = re.compile(
    r"^(?:manufacturer|mfr)\s*[:=]\s*(?P<value>\S(?:.*\S)?)$", re.IGNORECASE
)
_PDF_PART_NUMBER = re.compile(
    r"^(?:manufacturer\s+part\s+number|manufacturer_part_number|mpn)\s*[:=]\s*(?P<value>\S(?:.*\S)?)$",
    re.IGNORECASE,
)
_PDF_IDENTITY_MAX_LINE_DISTANCE = 8

_PDF_FALLBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "manufacturer": {"type": "string"},
                    "manufacturer_part_number": {"type": "string"},
                    "manufacturer_page": {"type": "integer", "minimum": 1},
                    "manufacturer_line": {"type": "string"},
                    "part_number_page": {"type": "integer", "minimum": 1},
                    "part_number_line": {"type": "string"},
                },
                "required": [
                    "manufacturer",
                    "manufacturer_part_number",
                    "manufacturer_page",
                    "manufacturer_line",
                    "part_number_page",
                    "part_number_line",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}
_PDF_FALLBACK_PROMPT = (
    "Extract purchasing identity only when the canonical PDF text literally states both "
    "a manufacturer and its manufacturer part number. Return exact substrings and exact "
    "complete evidence lines from the supplied page text. Do not infer from filenames, "
    "logos, URLs, product families, package codes, dimensions, or general context. If "
    "either value is absent or ambiguous, return records=[]. Do not return offers, prices, "
    "availability, lifecycle, or guessed identifiers."
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProcurementExtractionError(
                f"duplicate component-input JSON field {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ProcurementExtractionError(
        f"non-finite component-input JSON number {value!r} is forbidden"
    )


def _parse_component_input(source: bytes) -> Mapping[str, Any]:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProcurementExtractionError(
            "component input must be UTF-8 JSON"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ProcurementExtractionError:
        raise
    except json.JSONDecodeError as exc:
        raise ProcurementExtractionError(
            f"invalid component-input JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ProcurementExtractionError("component input must be a JSON object")
    return value


def _optional_string(
    data: Mapping[str, Any], name: str, *, allow_null: bool = True
) -> str | None:
    if name not in data or (allow_null and data[name] is None):
        return None
    value = data[name]
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ProcurementExtractionError(
            f"component input {name!r} must be a nonempty trimmed string"
        )
    return value


def _string_array(data: Mapping[str, Any], name: str) -> tuple[str, ...]:
    if name not in data:
        return ()
    value = data[name]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProcurementExtractionError(
            f"component input {name!r} must be an array of strings"
        )
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ProcurementExtractionError(
                f"component input {name}[{index}] must be a nonempty trimmed string"
            )
        result.append(item)
    if len(result) != len(set(result)):
        raise ProcurementExtractionError(
            f"component input {name!r} must not contain duplicates"
        )
    return tuple(result)


def _products(data: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    product = _optional_string(data, "product")
    products = _string_array(data, "products")
    if product is not None and products:
        raise ProcurementExtractionError(
            "component input cannot contain both 'product' and 'products'"
        )
    if product is not None:
        return ((product, "$.product"),)
    return tuple((value, f"$.products[{index}]") for index, value in enumerate(products))


def _is_compact_part_number(value: str) -> bool:
    return (
        _COMPACT_PART_NUMBER.fullmatch(value) is not None
        and any(character.isalpha() for character in value)
        and any(character.isdigit() for character in value)
        and not any(character.isspace() for character in value)
    )


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    result = _SLUG_RUN.sub("-", ascii_value.lower()).strip("-")
    if not result:
        result = "record"
    if not result[0].isalpha():
        result = "record-" + result
    return result[:120].rstrip("-")


def _record_id(
    manufacturer: str | None,
    product: str,
    identifiers: Sequence[ProcurementIdentifierSpec],
) -> str:
    preferred = next(
        (
            item.value
            for item in identifiers
            if item.scheme
            in {"manufacturer_part_number", "manufacturer_item_number"}
        ),
        product,
    )
    prefix = manufacturer if manufacturer is not None else "product"
    return f"{_slug(prefix)}.{_slug(preferred)}"


def _document_kind(url: str, *, explicit_datasheet: bool) -> str:
    if explicit_datasheet:
        return "datasheet"
    path = urlparse(url).path.lower()
    if path.endswith((".pdf", ".ashx")) or "/datasheet" in path:
        return "datasheet"
    return "product_page"


def _product_url_match(product: str, url: str) -> bool:
    """Return true only for a strong, text-local product/URL match."""

    decoded = unquote(url).lower()
    item = _ITEM_SUFFIX.search(product)
    if item is not None:
        token = item.group("item").lower()
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", decoded):
            return True
    tokens = [token.lower() for token in _URL_TOKEN.findall(product)]
    if len(tokens) < 2:
        return False
    normalized_url = "-".join(_URL_TOKEN.findall(decoded))
    for length in range(len(tokens), 1, -1):
        for start in range(0, len(tokens) - length + 1):
            phrase = "-".join(tokens[start : start + length])
            if len(phrase) >= 8 and phrase in normalized_url:
                return True
    return False


def _selected_urls(
    product: str,
    products_count: int,
    urls: Sequence[tuple[str, str, bool]],
) -> tuple[tuple[str, str, bool], ...]:
    if products_count == 1:
        return tuple(urls)
    return tuple(item for item in urls if _product_url_match(product, item[0]))


def _provision_sequence(
    value: Sequence[ProcurementProvisionSpec] | ProcurementProvisionSpec,
    context: str,
) -> tuple[ProcurementProvisionSpec, ...]:
    if isinstance(value, ProcurementProvisionSpec):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProcurementExtractionError(
            f"{context} must be a procurement provision or an array of provisions"
        )
    result = tuple(value)
    if any(not isinstance(item, ProcurementProvisionSpec) for item in result):
        raise ProcurementExtractionError(f"{context} contains an invalid provision")
    return result


def static_part_provision(
    static: "StaticPartSpec", *, quantity: int = 1
) -> ProcurementProvisionSpec:
    """Create an exact, hash-bound provision for an already-parsed static part."""

    return ProcurementProvisionSpec(
        part=static.id,
        version=static.version,
        static_sha256=static.sha256,
        quantity=quantity,
    )


def extract_component_procurement(
    source: bytes,
    *,
    source_name: str,
    provisions_by_product: Mapping[
        str, Sequence[ProcurementProvisionSpec] | ProcurementProvisionSpec
    ]
    | None = None,
) -> tuple[ProcurementRecord, ...]:
    """Extract procurement records from one component-input JSON document.

    Provision bindings are deliberately explicit and keyed by the exact product
    string in the source.  Omitting the mapping produces valid unmodeled records
    with ``provides: []``; the extractor never guesses a catalog part from a
    filename, URL domain, product family, or similar spelling.
    """

    if not isinstance(source, bytes):
        raise TypeError("source must be bytes so evidence hashes exact input content")
    if not isinstance(source_name, str) or not source_name.strip() or source_name != source_name.strip():
        raise ProcurementExtractionError("source_name must be a nonempty trimmed string")
    data = _parse_component_input(source)
    manufacturer = _optional_string(data, "manufacturer")
    products = _products(data)
    explicit_mpn = _optional_string(data, "manufacturer_part_number")
    explicit_item = _optional_string(data, "manufacturer_item_number")
    if explicit_item is None:
        explicit_item = _optional_string(data, "item_number")
        item_locator = "$.item_number"
    else:
        item_locator = "$.manufacturer_item_number"
    if (explicit_mpn is not None or explicit_item is not None) and manufacturer is None:
        raise ProcurementExtractionError(
            "explicit manufacturer part/item numbers require an explicit manufacturer"
        )
    if len(products) > 1 and (explicit_mpn is not None or explicit_item is not None):
        raise ProcurementExtractionError(
            "top-level part/item numbers are ambiguous when 'products' has multiple entries"
        )
    if not products:
        identity = explicit_mpn if explicit_mpn is not None else explicit_item
        if identity is None:
            return ()
        locator = "$.manufacturer_part_number" if explicit_mpn is not None else item_locator
        products = ((identity, locator),)

    supplied = {} if provisions_by_product is None else dict(provisions_by_product)
    product_values = {item[0] for item in products}
    unknown_bindings = sorted(set(supplied) - product_values)
    if unknown_bindings:
        raise ProcurementExtractionError(
            "provision bindings do not exactly match an evidenced product: "
            + ", ".join(repr(item) for item in unknown_bindings)
        )

    source_urls = _string_array(data, "source_urls")
    datasheet_urls = _string_array(data, "datasheet_urls")
    url_values: list[tuple[str, str, bool]] = [
        (url, f"$.source_urls[{index}]", False)
        for index, url in enumerate(source_urls)
    ] + [
        (url, f"$.datasheet_urls[{index}]", True)
        for index, url in enumerate(datasheet_urls)
    ]
    source_digest = "sha256:" + hashlib.sha256(source).hexdigest()

    records: list[ProcurementRecord] = []
    seen_ids: set[str] = set()
    for product, product_locator in products:
        # A maker-looking first token in a free-form product name is not an
        # explicit issuer declaration. Without the dedicated manufacturer
        # field, retain only the evidenced product name.
        record_manufacturer = manufacturer
        identifiers: list[ProcurementIdentifierSpec] = [
            ProcurementIdentifierSpec("product_name", product)
        ]
        identifier_locators = [product_locator]
        if record_manufacturer is not None and _is_compact_part_number(product):
            identifiers.append(
                ProcurementIdentifierSpec(
                    "manufacturer_part_number", product, record_manufacturer
                )
            )
            identifier_locators.append(product_locator)
        item_match = _ITEM_SUFFIX.search(product)
        if record_manufacturer is not None and item_match is not None:
            identifiers.append(
                ProcurementIdentifierSpec(
                    "manufacturer_item_number",
                    item_match.group("item"),
                    record_manufacturer,
                )
            )
            identifier_locators.append(product_locator)
        if explicit_mpn is not None:
            candidate = ProcurementIdentifierSpec(
                "manufacturer_part_number", explicit_mpn, manufacturer
            )
            if candidate not in identifiers:
                identifiers.append(candidate)
                identifier_locators.append("$.manufacturer_part_number")
        if explicit_item is not None:
            candidate = ProcurementIdentifierSpec(
                "manufacturer_item_number", explicit_item, manufacturer
            )
            if candidate not in identifiers:
                identifiers.append(candidate)
                identifier_locators.append(item_locator)

        selected = _selected_urls(product, len(products), url_values)
        documents_by_url: dict[str, ProcurementDocumentSpec] = {}
        document_locators: dict[str, str] = {}
        for url, locator, explicit_datasheet in selected:
            try:
                document = ProcurementDocumentSpec(
                    kind=_document_kind(url, explicit_datasheet=explicit_datasheet),
                    url=url,
                )
            except ProcurementSpecError as exc:
                raise ProcurementExtractionError(
                    f"component input {locator} is not a valid source URL: {exc}"
                ) from exc
            existing = documents_by_url.get(url)
            if existing is None or document.kind == "datasheet":
                documents_by_url[url] = document
                document_locators[url] = locator

        evidence_locators = identifier_locators[:]
        if manufacturer is not None:
            evidence_locators.append("$.manufacturer")
        evidence_locators.extend(document_locators.values())
        unique_locators = tuple(dict.fromkeys(evidence_locators))
        evidence = tuple(
            ProcurementEvidenceSpec(source_name, source_digest, locator)
            for locator in unique_locators
        )
        provisions = _provision_sequence(
            supplied.get(product, ()),
            f"provisions_by_product[{product!r}]",
        )
        record_id = _record_id(record_manufacturer, product, identifiers)
        if record_id in seen_ids:
            raise ProcurementExtractionError(
                f"component input produces duplicate procurement id {record_id!r}"
            )
        seen_ids.add(record_id)
        records.append(
            ProcurementRecord(
                format="procurement-record-1",
                id=record_id,
                version="1.0.0",
                manufacturer=record_manufacturer,
                identifiers=tuple(identifiers),
                documents=tuple(documents_by_url.values()),
                offers=(),
                lifecycle=ProcurementLifecycleSpec("unknown"),
                provides=provisions,
                evidence=evidence,
            )
        )
    return tuple(records)


def _parse_extraction_json(source: bytes, context: str) -> Mapping[str, Any]:
    if not isinstance(source, bytes):
        raise TypeError("structured extraction source must be bytes")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProcurementExtractionError(f"{context} must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ProcurementExtractionError:
        raise
    except json.JSONDecodeError as exc:
        raise ProcurementExtractionError(
            f"invalid {context} JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ProcurementExtractionError(f"{context} must be a JSON object")
    return value


def _extraction_source(
    data: Mapping[str, Any], context: str
) -> tuple[str, str]:
    source = data.get("source")
    if not isinstance(source, Mapping):
        raise ProcurementExtractionError(f"{context}.source must be an object")
    name = source.get("name")
    digest = source.get("sha256")
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ProcurementExtractionError(f"{context}.source.name is invalid")
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ProcurementExtractionError(f"{context}.source.sha256 is invalid")
    return name, digest


def _structured_fact(
    raw: Any, *, context: str, source_sha256: str
) -> tuple[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != {"value", "evidence"}:
        raise ProcurementExtractionError(f"{context} must be an exact extracted fact")
    value = raw.get("value")
    evidence = raw.get("evidence")
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ProcurementExtractionError(f"{context}.value is invalid")
    if not isinstance(evidence, Mapping):
        raise ProcurementExtractionError(f"{context}.evidence must be an object")
    locator = evidence.get("locator")
    if not isinstance(locator, str) or not locator.strip() or locator != locator.strip():
        raise ProcurementExtractionError(f"{context}.evidence.locator is invalid")
    if evidence.get("source_sha256") != source_sha256:
        raise ProcurementExtractionError(f"{context}.evidence source hash changed")
    expected_value = "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    if evidence.get("value_sha256") != expected_value:
        raise ProcurementExtractionError(f"{context}.evidence value hash changed")
    for name in ("line", "column"):
        number = evidence.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ProcurementExtractionError(f"{context}.evidence.{name} is invalid")
    if set(evidence) != {"source_sha256", "locator", "line", "column", "value_sha256"}:
        raise ProcurementExtractionError(f"{context}.evidence has unexpected fields")
    return value, locator


def _ecad_document(
    raw: Any, *, context: str, source_sha256: str
) -> tuple[ProcurementDocumentSpec, str]:
    if not isinstance(raw, Mapping):
        raise ProcurementExtractionError(f"{context} must be an object")
    allowed = {"kind", "url", "title", "media_type", "evidence"}
    if set(raw) - allowed or not {"kind", "url", "evidence"}.issubset(raw):
        raise ProcurementExtractionError(f"{context} has invalid fields")
    kind = raw.get("kind")
    url = raw.get("url")
    evidence = raw.get("evidence")
    if kind not in {"datasheet", "product_document"} or not isinstance(url, str):
        raise ProcurementExtractionError(f"{context} has an invalid document identity")
    if not isinstance(evidence, Mapping):
        raise ProcurementExtractionError(f"{context}.evidence must be an object")
    if evidence.get("source_sha256") != source_sha256:
        raise ProcurementExtractionError(f"{context}.evidence source hash changed")
    locator = evidence.get("locator")
    if not isinstance(locator, str) or not locator.strip():
        raise ProcurementExtractionError(f"{context}.evidence.locator is invalid")
    title = None
    if raw.get("title") is not None:
        title, _title_locator = _structured_fact(
            raw["title"], context=f"{context}.title", source_sha256=source_sha256
        )
    media_type = None
    if raw.get("media_type") is not None:
        media_type, _media_locator = _structured_fact(
            raw["media_type"],
            context=f"{context}.media_type",
            source_sha256=source_sha256,
        )
    try:
        document = ProcurementDocumentSpec(
            "datasheet" if kind == "datasheet" else "other",
            url,
            media_type=media_type,
            source_sha256=source_sha256,
            title=title,
        )
    except ProcurementSpecError as exc:
        raise ProcurementExtractionError(f"{context} is invalid: {exc}") from exc
    return document, locator


def _identity_record(
    manufacturer: str,
    part_number: str,
    *,
    evidence_source: str,
    evidence_sha256: str,
    locators: Sequence[str],
    documents: Sequence[ProcurementDocumentSpec] = (),
) -> ProcurementRecord:
    identifier = ProcurementIdentifierSpec(
        "manufacturer_part_number", part_number, manufacturer
    )
    return ProcurementRecord(
        format="procurement-record-1",
        id=_record_id(manufacturer, part_number, (identifier,)),
        version="1.0.0",
        manufacturer=manufacturer,
        identifiers=(identifier,),
        documents=tuple(documents),
        offers=(),
        lifecycle=ProcurementLifecycleSpec("unknown"),
        provides=(),
        evidence=tuple(
            ProcurementEvidenceSpec(evidence_source, evidence_sha256, locator)
            for locator in dict.fromkeys(locators)
        ),
    )


def extract_ecad_procurement(
    source: bytes, *, source_name: str
) -> tuple[ProcurementRecord, ...]:
    """Convert only explicit manufacturer/MPN fields from canonical ECAD JSON."""

    data = _parse_extraction_json(source, "deterministic ECAD extraction")
    if data.get("format") != "deterministic-ecad-extraction-1":
        raise ProcurementExtractionError("unsupported deterministic ECAD extraction format")
    _embedded_name, source_sha256 = _extraction_source(
        data, "deterministic ECAD extraction"
    )
    raw_records = data.get("records")
    if not isinstance(raw_records, list):
        raise ProcurementExtractionError("deterministic ECAD records must be an array")
    records: list[ProcurementRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ProcurementExtractionError(f"deterministic ECAD record[{index}] is invalid")
        manufacturer_raw = raw.get("manufacturer")
        part_number_raw = raw.get("manufacturer_part_number")
        # An MPN without an explicit manufacturer cannot satisfy the typed
        # manufacturer-issued identifier contract and is deliberately omitted.
        if manufacturer_raw is None or part_number_raw is None:
            continue
        manufacturer, manufacturer_locator = _structured_fact(
            manufacturer_raw,
            context=f"deterministic ECAD record[{index}].manufacturer",
            source_sha256=source_sha256,
        )
        part_number, part_number_locator = _structured_fact(
            part_number_raw,
            context=f"deterministic ECAD record[{index}].manufacturer_part_number",
            source_sha256=source_sha256,
        )
        raw_documents = raw.get("documents", [])
        if not isinstance(raw_documents, list):
            raise ProcurementExtractionError(
                f"deterministic ECAD record[{index}].documents must be an array"
            )
        documents: list[ProcurementDocumentSpec] = []
        document_locators: list[str] = []
        for document_index, item in enumerate(raw_documents):
            document, locator = _ecad_document(
                item,
                context=f"deterministic ECAD record[{index}].documents[{document_index}]",
                source_sha256=source_sha256,
            )
            documents.append(document)
            document_locators.append(locator)
        try:
            record = _identity_record(
                manufacturer,
                part_number,
                evidence_source=source_name,
                evidence_sha256=source_sha256,
                locators=(manufacturer_locator, part_number_locator, *document_locators),
                documents=documents,
            )
        except ProcurementSpecError as exc:
            raise ProcurementExtractionError(
                f"deterministic ECAD record[{index}] is not a valid identity: {exc}"
            ) from exc
        if record.id in seen:
            raise ProcurementExtractionError(
                f"deterministic ECAD extraction produces duplicate id {record.id!r}"
            )
        seen.add(record.id)
        records.append(record)
    return tuple(records)


def _pdf_extraction(
    source: bytes,
) -> tuple[Mapping[str, Any], str, tuple[tuple[str, ...], ...]]:
    data = _parse_extraction_json(source, "deterministic PDF extraction")
    if data.get("format") != "deterministic-pdf-extraction-1":
        raise ProcurementExtractionError("unsupported deterministic PDF extraction format")
    _source_name, source_sha256 = _extraction_source(
        data, "deterministic PDF extraction"
    )
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ProcurementExtractionError("deterministic PDF pages must be nonempty")
    if data.get("page_count") != len(pages):
        raise ProcurementExtractionError("deterministic PDF page count changed")
    page_lines: list[tuple[str, ...]] = []
    page_texts: list[str] = []
    for index, page in enumerate(pages):
        if not isinstance(page, Mapping) or page.get("number") != index + 1:
            raise ProcurementExtractionError(f"deterministic PDF page[{index}] is invalid")
        text = page.get("text")
        if not isinstance(text, str):
            raise ProcurementExtractionError(f"deterministic PDF page[{index}].text is invalid")
        encoded = text.encode("utf-8")
        if page.get("characters") != len(text):
            raise ProcurementExtractionError(f"deterministic PDF page[{index}] character count changed")
        if page.get("text_sha256") != "sha256:" + hashlib.sha256(encoded).hexdigest():
            raise ProcurementExtractionError(f"deterministic PDF page[{index}] text hash changed")
        page_lines.append(tuple(text.splitlines()))
        page_texts.append(text)
    joined = "\f".join(page_texts).encode("utf-8")
    if data.get("text_sha256") != "sha256:" + hashlib.sha256(joined).hexdigest():
        raise ProcurementExtractionError("deterministic PDF joined text hash changed")
    return data, source_sha256, tuple(page_lines)


def _literal_pdf_identities(
    page_lines: Sequence[Sequence[str]], *, source_name: str, source_sha256: str
) -> tuple[ProcurementRecord, ...]:
    manufacturers: list[tuple[str, int, str]] = []
    part_numbers: list[tuple[str, int, str]] = []
    for page_index, lines in enumerate(page_lines):
        for line in lines:
            manufacturer = _PDF_MANUFACTURER.fullmatch(line)
            if manufacturer is not None:
                manufacturers.append(
                    (manufacturer.group("value"), page_index + 1, line)
                )
            part_number = _PDF_PART_NUMBER.fullmatch(line)
            if part_number is not None:
                part_numbers.append(
                    (part_number.group("value"), page_index + 1, line)
                )
    # Repeated or multiple labels are not a singleton deterministic identity.
    # The optional bounded fallback may enumerate multiple locally paired
    # records, but it must pass the same exact label-pair validator below.
    if len(manufacturers) != 1 or len(part_numbers) != 1:
        return ()
    manufacturer, manufacturer_page, manufacturer_line = manufacturers[0]
    part_number, part_number_page, part_number_line = part_numbers[0]
    try:
        locators = _validated_pdf_label_pair(
            page_lines=page_lines,
            manufacturer=manufacturer,
            part_number=part_number,
            manufacturer_page=manufacturer_page,
            manufacturer_line=manufacturer_line,
            part_number_page=part_number_page,
            part_number_line=part_number_line,
            context="deterministic PDF identity",
        )
    except ProcurementExtractionError:
        return ()
    try:
        return (
            _identity_record(
                manufacturer,
                part_number,
                evidence_source=source_name,
                evidence_sha256=source_sha256,
                locators=locators,
            ),
        )
    except ProcurementSpecError as exc:
        raise ProcurementExtractionError(
            f"deterministic PDF labels contain an invalid identity: {exc}"
        ) from exc


def _validated_pdf_label_pair(
    *,
    page_lines: Sequence[Sequence[str]],
    manufacturer: str,
    part_number: str,
    manufacturer_page: Any,
    manufacturer_line: Any,
    part_number_page: Any,
    part_number_line: Any,
    context: str,
) -> tuple[str, str]:
    """Validate one exact, local manufacturer/MPN label pair.

    Literal occurrence alone is insufficient evidence that two values form one
    issued identity.  The labels must occur uniquely on the same page, in
    manufacturer-then-MPN order, within a small source window, with no other
    identity label between them.
    """

    for name, page in (
        ("manufacturer_page", manufacturer_page),
        ("part_number_page", part_number_page),
    ):
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
            or page > len(page_lines)
        ):
            raise ProcurementExtractionError(f"{context} {name} is invalid")
    if manufacturer_page != part_number_page:
        raise ProcurementExtractionError(
            f"{context} manufacturer and MPN labels must be on the same page"
        )
    if not isinstance(manufacturer_line, str) or not manufacturer_line:
        raise ProcurementExtractionError(
            f"{context} manufacturer evidence line is invalid"
        )
    if not isinstance(part_number_line, str) or not part_number_line:
        raise ProcurementExtractionError(
            f"{context} MPN evidence line is invalid"
        )
    manufacturer_match = _PDF_MANUFACTURER.fullmatch(manufacturer_line)
    if (
        manufacturer_match is None
        or manufacturer_match.group("value") != manufacturer
    ):
        raise ProcurementExtractionError(
            f"{context} manufacturer line is not an exact manufacturer label"
        )
    part_number_match = _PDF_PART_NUMBER.fullmatch(part_number_line)
    if part_number_match is None or part_number_match.group("value") != part_number:
        raise ProcurementExtractionError(
            f"{context} part-number line is not an exact MPN label"
        )

    lines = page_lines[manufacturer_page - 1]
    manufacturer_matches = [
        index for index, line in enumerate(lines, 1) if line == manufacturer_line
    ]
    part_number_matches = [
        index for index, line in enumerate(lines, 1) if line == part_number_line
    ]
    if len(manufacturer_matches) != 1:
        raise ProcurementExtractionError(
            f"{context} manufacturer evidence line is not unique"
        )
    if len(part_number_matches) != 1:
        raise ProcurementExtractionError(
            f"{context} MPN evidence line is not unique"
        )
    manufacturer_index = manufacturer_matches[0]
    part_number_index = part_number_matches[0]
    distance = part_number_index - manufacturer_index
    if distance < 1 or distance > _PDF_IDENTITY_MAX_LINE_DISTANCE:
        raise ProcurementExtractionError(
            f"{context} manufacturer/MPN labels are not an adjacent ordered pair"
        )
    intervening = lines[manufacturer_index:part_number_index - 1]
    if any(
        _PDF_MANUFACTURER.fullmatch(line) is not None
        or _PDF_PART_NUMBER.fullmatch(line) is not None
        for line in intervening
    ):
        raise ProcurementExtractionError(
            f"{context} has another identity label inside the evidence pair"
        )
    return (
        f"$.pages[{manufacturer_page - 1}].text#line={manufacturer_index}",
        f"$.pages[{part_number_page - 1}].text#line={part_number_index}",
    )


def _fallback_hashes(
    source: bytes, source_sha256: str, config: ProcurementTextFallbackConfig
) -> dict[str, str]:
    canonical_schema = json.dumps(
        _PDF_FALLBACK_SCHEMA, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "source_sha256": source_sha256,
        "extraction_sha256": "sha256:" + hashlib.sha256(source).hexdigest(),
        "prompt_sha256": "sha256:" + hashlib.sha256(_PDF_FALLBACK_PROMPT.encode("utf-8")).hexdigest(),
        "schema_sha256": "sha256:" + hashlib.sha256(canonical_schema).hexdigest(),
        "model_sha256": "sha256:" + hashlib.sha256(config.model.encode("utf-8")).hexdigest(),
    }


def procurement_fallback_identity(
    config: ProcurementTextFallbackConfig | None,
) -> Mapping[str, Any]:
    if config is None:
        return {"enabled": False}
    hashes = _fallback_hashes(b"", "sha256:" + "0" * 64, config)
    return {
        "enabled": True,
        "model": config.model,
        "reasoning_effort": "low",
        "max_attempts": config.max_attempts,
        "max_completion_tokens": config.max_completion_tokens,
        "prompt_sha256": hashes["prompt_sha256"],
        "schema_sha256": hashes["schema_sha256"],
        "model_sha256": hashes["model_sha256"],
    }


def _validate_fallback_value(
    value: Any,
    *,
    page_lines: Sequence[Sequence[str]],
    source_name: str,
    source_sha256: str,
) -> tuple[ProcurementRecord, ...]:
    if not isinstance(value, Mapping) or set(value) != {"records"}:
        raise ProcurementExtractionError("PDF fallback response must contain only records")
    raw_records = value.get("records")
    if not isinstance(raw_records, list) or len(raw_records) > 16:
        raise ProcurementExtractionError("PDF fallback records must be an array of at most 16 items")
    required = {
        "manufacturer",
        "manufacturer_part_number",
        "manufacturer_page",
        "manufacturer_line",
        "part_number_page",
        "part_number_line",
    }
    records: list[ProcurementRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ProcurementExtractionError(f"PDF fallback record[{index}] has invalid fields")
        manufacturer = raw.get("manufacturer")
        part_number = raw.get("manufacturer_part_number")
        if (
            not isinstance(manufacturer, str)
            or not manufacturer.strip()
            or manufacturer != manufacturer.strip()
            or not isinstance(part_number, str)
            or not _is_compact_part_number(part_number)
        ):
            raise ProcurementExtractionError(f"PDF fallback record[{index}] identity is invalid")
        locators = _validated_pdf_label_pair(
            page_lines=page_lines,
            manufacturer=manufacturer,
            part_number=part_number,
            manufacturer_page=raw.get("manufacturer_page"),
            manufacturer_line=raw.get("manufacturer_line"),
            part_number_page=raw.get("part_number_page"),
            part_number_line=raw.get("part_number_line"),
            context=f"PDF fallback record[{index}]",
        )
        try:
            record = _identity_record(
                manufacturer,
                part_number,
                evidence_source=source_name,
                evidence_sha256=source_sha256,
                locators=locators,
            )
        except ProcurementSpecError as exc:
            raise ProcurementExtractionError(
                f"PDF fallback record[{index}] is not a valid identity: {exc}"
            ) from exc
        if record.id in seen:
            raise ProcurementExtractionError(
                f"PDF fallback produces duplicate identity {record.id!r}"
            )
        seen.add(record.id)
        records.append(record)
    return tuple(records)


def _fallback_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ProcurementExtractionError("PDF fallback response has no message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ProcurementExtractionError("PDF fallback response content is empty")
    return content


def _fallback_pdf_procurement(
    source: bytes,
    *,
    source_name: str,
    source_sha256: str,
    page_lines: Sequence[Sequence[str]],
    config: ProcurementTextFallbackConfig,
    client: Any | None,
) -> tuple[ProcurementRecord, ...]:
    hashes = _fallback_hashes(source, source_sha256, config)
    key_payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(key_payload).hexdigest()
    cache_path = config.cache_directory / f"{cache_key}.json"
    if cache_path.exists():
        if cache_path.is_symlink() or not cache_path.is_file():
            raise ProcurementExtractionError(f"invalid PDF fallback cache entry: {cache_path}")
        cached = _parse_extraction_json(cache_path.read_bytes(), "PDF fallback cache")
        if cached.get("format") != "procurement-pdf-fallback-cache-1" or cached.get("hashes") != hashes:
            raise ProcurementExtractionError("PDF fallback cache identity changed")
        return _validate_fallback_value(
            cached.get("response"),
            page_lines=page_lines,
            source_name=source_name,
            source_sha256=source_sha256,
        )
    if client is None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ProcurementExtractionError(
                "install the 'agents' extra to enable PDF procurement fallback"
            ) from exc
        client = OpenAI(api_key=config.api_key) if config.api_key else OpenAI()
    canonical_text = source.decode("utf-8")
    last_error: Exception | None = None
    response_value: Any = None
    records: tuple[ProcurementRecord, ...] | None = None
    for _attempt in range(1, config.max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=config.model,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": _PDF_FALLBACK_PROMPT},
                    {"role": "user", "content": canonical_text},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "procurement_pdf_identity",
                        "strict": True,
                        "schema": _PDF_FALLBACK_SCHEMA,
                    },
                },
                max_completion_tokens=config.max_completion_tokens,
            )
            response_value = _parse_extraction_json(
                _fallback_content(response).encode("utf-8"),
                "PDF fallback response",
            )
            records = _validate_fallback_value(
                response_value,
                page_lines=page_lines,
                source_name=source_name,
                source_sha256=source_sha256,
            )
            break
        except Exception as exc:  # Validation/API failures are bounded by max_attempts.
            last_error = exc
    if records is None:
        raise ProcurementExtractionError(
            f"PDF fallback failed validation after {config.max_attempts} attempts: {last_error}"
        ) from last_error
    config.cache_directory.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "format": "procurement-pdf-fallback-cache-1",
                "hashes": hashes,
                "response": response_value,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.", suffix=".tmp", dir=str(config.cache_directory)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return records


def extract_pdf_procurement(
    source: bytes,
    *,
    source_name: str,
    fallback: ProcurementTextFallbackConfig | None = None,
    client: Any | None = None,
) -> tuple[ProcurementRecord, ...]:
    """Extract explicit PDF-text identity, optionally using bounded Luna-low parsing."""

    _data, source_sha256, page_lines = _pdf_extraction(source)
    records = _literal_pdf_identities(
        page_lines, source_name=source_name, source_sha256=source_sha256
    )
    if records or fallback is None:
        return records
    return _fallback_pdf_procurement(
        source,
        source_name=source_name,
        source_sha256=source_sha256,
        page_lines=page_lines,
        config=fallback,
        client=client,
    )


def extract_structured_procurement(
    source: bytes,
    *,
    source_name: str,
    pdf_fallback: ProcurementTextFallbackConfig | None = None,
    client: Any | None = None,
) -> tuple[ProcurementRecord, ...]:
    """Dispatch canonical host extraction JSON without accepting raw documents."""

    data = _parse_extraction_json(source, "deterministic document extraction")
    format_name = data.get("format")
    if format_name == "deterministic-ecad-extraction-1":
        return extract_ecad_procurement(source, source_name=source_name)
    if format_name == "deterministic-pdf-extraction-1":
        return extract_pdf_procurement(
            source,
            source_name=source_name,
            fallback=pdf_fallback,
            client=client,
        )
    raise ProcurementExtractionError(
        f"unsupported deterministic document extraction format {format_name!r}"
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _below(root: Path, relative: str, context: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ProcurementExtractionError(f"{context} must be a safe relative POSIX path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ProcurementExtractionError(f"{context} must be a safe relative POSIX path")
    result = root.joinpath(*parts).resolve()
    if root != result and root not in result.parents:
        raise ProcurementExtractionError(f"{context} escapes {root}")
    return result


def write_host_procurement_context(
    run_root: str | Path,
    *,
    component_input: str | Path,
    source_name: str,
    deterministic_plan: str | Path | None = None,
) -> Path:
    """Write protected host context used after Luna's candidate is validated."""

    root = Path(run_root).resolve()
    component = Path(component_input).resolve()
    if root != component and root not in component.parents:
        raise ProcurementExtractionError("component snapshot escapes modeling run root")
    if component.is_symlink() or not component.is_file():
        raise ProcurementExtractionError("component snapshot is not a regular file")
    if not isinstance(source_name, str) or not source_name.strip() or source_name != source_name.strip():
        raise ProcurementExtractionError("component evidence source name is invalid")
    payload: dict[str, Any] = {
        "format": HOST_PROCUREMENT_CONTEXT_FORMAT,
        "component_input": {
            "path": component.relative_to(root).as_posix(),
            "source": source_name,
            "sha256": "sha256:" + hashlib.sha256(component.read_bytes()).hexdigest(),
        },
        "deterministic_plan": None,
    }
    if deterministic_plan is not None:
        plan = Path(deterministic_plan).resolve()
        if root != plan and root not in plan.parents:
            raise ProcurementExtractionError("deterministic plan escapes modeling run root")
        if plan.is_symlink() or not plan.is_file():
            raise ProcurementExtractionError("deterministic plan is not a regular file")
        payload["deterministic_plan"] = {
            "path": plan.relative_to(root).as_posix(),
            "sha256": "sha256:" + hashlib.sha256(plan.read_bytes()).hexdigest(),
        }
    path = root / HOST_PROCUREMENT_CONTEXT_FILENAME
    _write_text_atomic(path, _canonical_json(payload))
    return path


def proposal_procurement_receipt_path(candidate_root: str | Path) -> Path:
    candidate = Path(candidate_root).resolve()
    return candidate.parent.parent / HOST_PROCUREMENT_RECEIPT_FILENAME


def _load_host_context(path: Path) -> tuple[Mapping[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ProcurementExtractionError(f"host procurement context is missing: {path}")
    source = path.read_bytes()
    data = _parse_extraction_json(source, "host procurement context")
    if set(data) != {"format", "component_input", "deterministic_plan"}:
        raise ProcurementExtractionError("host procurement context has invalid fields")
    if data.get("format") != HOST_PROCUREMENT_CONTEXT_FORMAT:
        raise ProcurementExtractionError("unsupported host procurement context format")
    return data, source


def _context_sources(
    context_path: Path,
) -> tuple[bytes, str, tuple[tuple[bytes, str], ...], tuple[dict[str, Any], ...]]:
    context, context_source = _load_host_context(context_path)
    run_root = context_path.parent.resolve()
    component = context.get("component_input")
    if not isinstance(component, Mapping) or set(component) != {"path", "source", "sha256"}:
        raise ProcurementExtractionError("host component context is invalid")
    component_path = _below(run_root, component.get("path"), "component_input.path")
    if component_path.is_symlink() or not component_path.is_file():
        raise ProcurementExtractionError("host component snapshot is missing")
    component_source = component.get("source")
    component_sha = component.get("sha256")
    if not isinstance(component_source, str) or not component_source.strip():
        raise ProcurementExtractionError("host component source is invalid")
    component_payload = component_path.read_bytes()
    actual_component_sha = "sha256:" + hashlib.sha256(component_payload).hexdigest()
    if component_sha != actual_component_sha:
        raise ProcurementExtractionError("host component snapshot hash changed")
    sources: list[dict[str, Any]] = [
        {
            "kind": "component_input",
            "source": component_source,
            "path": component_path.relative_to(run_root).as_posix(),
            "sha256": actual_component_sha,
        }
    ]
    structured: list[tuple[bytes, str]] = []
    plan_entry = context.get("deterministic_plan")
    if plan_entry is not None:
        if not isinstance(plan_entry, Mapping) or set(plan_entry) != {"path", "sha256"}:
            raise ProcurementExtractionError("host deterministic plan context is invalid")
        plan_path = _below(run_root, plan_entry.get("path"), "deterministic_plan.path")
        if plan_path.is_symlink() or not plan_path.is_file():
            raise ProcurementExtractionError("host deterministic plan is missing")
        plan_payload = plan_path.read_bytes()
        plan_sha = "sha256:" + hashlib.sha256(plan_payload).hexdigest()
        if plan_entry.get("sha256") != plan_sha:
            raise ProcurementExtractionError("host deterministic plan hash changed")
        from .deterministic_assets import modeling_context_paths

        for extraction_path in modeling_context_paths(plan_path):
            payload = extraction_path.read_bytes()
            extraction = _parse_extraction_json(
                payload, "host deterministic document extraction"
            )
            embedded_name, raw_sha = _extraction_source(
                extraction, "host deterministic document extraction"
            )
            structured.append((payload, embedded_name))
            sources.append(
                {
                    "kind": "structured_document_extraction",
                    "source": embedded_name,
                    "path": extraction_path.relative_to(run_root).as_posix(),
                    "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                    "raw_source_sha256": raw_sha,
                }
            )
    return component_payload, component_source, tuple(structured), tuple(sources)


def materialize_proposal_procurement(
    candidate_root: str | Path,
    context_path: str | Path,
    *,
    pdf_fallback: ProcurementTextFallbackConfig | None = None,
    client: Any | None = None,
    _allow_unbound_without_static: bool = False,
) -> tuple[tuple[Path, ...], dict[str, Any] | None]:
    """Add evidence-only central records to a validated temporary proposal.

    The caller must run this after deterministic host assets have finalized the
    candidate static part.  A singleton identity source is bound only when the
    proposal contains one static part; multi-product/multi-record sources remain
    valid but unbound.
    """

    candidate = Path(candidate_root).resolve()
    context_file = Path(context_path).resolve()
    component_payload, component_source, structured, sources = _context_sources(
        context_file
    )
    from ..catalog.instantiations import StaticPartSpec

    static_specs: list[StaticPartSpec] = []
    for path in sorted(candidate.rglob("static.part")):
        if path.is_symlink() or not path.is_file():
            raise ProcurementExtractionError(f"candidate static part is unsafe: {path}")
        try:
            static_specs.append(
                StaticPartSpec.from_json(path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            raise ProcurementExtractionError(
                f"candidate static part is invalid: {path}: {exc}"
            ) from exc
    # Procurement proposal artifacts belong to a concrete, already-validated
    # static-part candidate.  Identity evidence on its own is still useful to
    # the offline catalog migration, but must not make a modeling proposal
    # procurement-bearing when there is no physical part to provision.
    if not static_specs and not _allow_unbound_without_static:
        return (), None
    if _allow_unbound_without_static and static_specs:
        raise ProcurementExtractionError(
            "deferred procurement materialization must not contain static parts"
        )
    singleton_provision = (
        static_part_provision(static_specs[0]) if len(static_specs) == 1 else None
    )
    records: list[ProcurementRecord] = []
    component_records = extract_component_procurement(
        component_payload, source_name=component_source
    )
    if len(component_records) == 1 and singleton_provision is not None:
        component_records = (
            bind_record(component_records[0], (singleton_provision,)),
        )
    records.extend(component_records)
    for extraction_payload, extraction_source in structured:
        extracted = extract_structured_procurement(
            extraction_payload,
            source_name=extraction_source,
            pdf_fallback=pdf_fallback,
            client=client,
        )
        if len(extracted) == 1 and singleton_provision is not None:
            extracted = (bind_record(extracted[0], (singleton_provision,)),)
        records.extend(extracted)
    if not records:
        return (), None
    try:
        registry = ProcurementRegistry(records)
        registry.validate_provisions(
            {item.id: (item.version, item.sha256) for item in static_specs}
        )
    except ProcurementSpecError as exc:
        raise ProcurementExtractionError(
            f"proposal procurement registry is invalid: {exc}"
        ) from exc
    records_root = candidate / "procurement" / "records"
    if records_root.exists() and (records_root.is_symlink() or not records_root.is_dir()):
        raise ProcurementExtractionError("candidate procurement records root is unsafe")
    written: list[Path] = []
    artifacts: list[dict[str, Any]] = []
    for record in registry.values():
        target = records_root / f"{record.id}.procurement"
        if target.exists():
            raise ProcurementExtractionError(
                f"candidate already contains procurement artifact {target}"
            )
        payload = record.to_json()
        _write_text_atomic(target, payload)
        encoded = payload.encode("utf-8")
        written.append(target)
        artifacts.append(
            {
                "path": target.relative_to(candidate).as_posix(),
                "sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
                "record_sha256": record.sha256,
                "provides": [item.to_dict() for item in record.provides],
            }
        )
    context_payload = context_file.read_bytes()
    receipt = {
        "format": HOST_PROCUREMENT_RECEIPT_FORMAT,
        "context": {
            "path": context_file.name,
            "sha256": "sha256:" + hashlib.sha256(context_payload).hexdigest(),
        },
        "sources": list(sources),
        "artifacts": artifacts,
        "fallback": dict(procurement_fallback_identity(pdf_fallback)),
    }
    return tuple(written), receipt


def materialize_deferred_procurement(
    candidate_root: str | Path,
    context_path: str | Path,
    *,
    pdf_fallback: ProcurementTextFallbackConfig | None = None,
    client: Any | None = None,
) -> tuple[tuple[Path, ...], dict[str, Any] | None]:
    """Materialize evidence-only, unbound records for a deferred model import.

    This is the explicit zero-agent counterpart to proposal materialization.
    It is intentionally valid only for a candidate with no ``static.part``;
    every record therefore retains ``provides: []``.  Keeping this as a named
    host pathway prevents the ordinary post-validation proposal hook from
    producing free-floating procurement artifacts when Luna failed to produce
    a physical part.
    """

    return materialize_proposal_procurement(
        candidate_root,
        context_path,
        pdf_fallback=pdf_fallback,
        client=client,
        _allow_unbound_without_static=True,
    )


def verify_proposal_procurement_receipt(
    candidate_root: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return hash-verified host-owned procurement paths for validation/promotion."""

    candidate = Path(candidate_root).resolve()
    actual_files: list[Path] = []
    normalized_paths: dict[str, Path] = {}
    for path in sorted(candidate.rglob("*")):
        if path.suffix.casefold() != ".procurement":
            continue
        if path.is_symlink() or not path.is_file():
            raise ProcurementExtractionError(
                f"procurement artifact is not a regular file: {path}"
            )
        relative = path.relative_to(candidate).as_posix()
        normalized = relative.casefold()
        existing = normalized_paths.get(normalized)
        if existing is not None:
            raise ProcurementExtractionError(
                "case-colliding procurement artifact paths: "
                f"{existing.relative_to(candidate).as_posix()}, {relative}"
            )
        normalized_paths[normalized] = path
        actual_files.append(path)
    actual = tuple(actual_files)
    receipt_path = (
        proposal_procurement_receipt_path(candidate)
        if receipt_path is None
        else Path(receipt_path).resolve()
    )
    if receipt_path == candidate or candidate in receipt_path.parents:
        raise ProcurementExtractionError(
            "host procurement receipt may not be inside the candidate"
        )
    if not actual:
        if receipt_path.exists():
            raise ProcurementExtractionError(
                "host procurement receipt exists without procurement artifacts"
            )
        return ()
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ProcurementExtractionError(
            "procurement artifacts require a protected host receipt"
        )
    receipt = _parse_extraction_json(
        receipt_path.read_bytes(), "host procurement receipt"
    )
    if set(receipt) != {"format", "context", "sources", "artifacts", "fallback"}:
        raise ProcurementExtractionError("host procurement receipt has invalid fields")
    if receipt.get("format") != HOST_PROCUREMENT_RECEIPT_FORMAT:
        raise ProcurementExtractionError("unsupported host procurement receipt format")
    context = receipt.get("context")
    if not isinstance(context, Mapping) or set(context) != {"path", "sha256"}:
        raise ProcurementExtractionError("host procurement receipt context is invalid")
    context_path = _below(receipt_path.parent, context.get("path"), "receipt.context.path")
    if context_path.parent != receipt_path.parent:
        raise ProcurementExtractionError("host procurement context must be beside its receipt")
    context_payload = context_path.read_bytes()
    if context.get("sha256") != "sha256:" + hashlib.sha256(context_payload).hexdigest():
        raise ProcurementExtractionError("host procurement receipt context hash changed")
    _component, _source_name, _structured, expected_sources = _context_sources(
        context_path
    )
    if receipt.get("sources") != list(expected_sources):
        raise ProcurementExtractionError("host procurement receipt sources changed")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise ProcurementExtractionError("host procurement receipt artifacts are invalid")
    expected_paths: list[Path] = []
    evidence_pairs = {
        (item["source"], item.get("raw_source_sha256", item["sha256"]))
        for item in expected_sources
    }
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "path", "sha256", "record_sha256", "provides"
        }:
            raise ProcurementExtractionError(
                f"host procurement receipt artifact[{index}] is invalid"
            )
        path = _below(candidate, artifact.get("path"), f"receipt.artifacts[{index}].path")
        if path.parent != candidate / "procurement" / "records" or path.suffix != ".procurement":
            raise ProcurementExtractionError("host procurement artifact path is not central")
        payload = path.read_bytes()
        if artifact.get("sha256") != "sha256:" + hashlib.sha256(payload).hexdigest():
            raise ProcurementExtractionError("host procurement artifact hash changed")
        try:
            record = ProcurementRecord.from_json(payload.decode("utf-8"))
        except (UnicodeDecodeError, ProcurementSpecError) as exc:
            raise ProcurementExtractionError(
                f"host procurement artifact is invalid: {path}: {exc}"
            ) from exc
        if path.stem != record.id or artifact.get("record_sha256") != record.sha256:
            raise ProcurementExtractionError("host procurement record identity changed")
        if artifact.get("provides") != [item.to_dict() for item in record.provides]:
            raise ProcurementExtractionError("host procurement provision receipt changed")
        if any((item.source, item.sha256) not in evidence_pairs for item in record.evidence):
            raise ProcurementExtractionError("host procurement evidence is outside its source context")
        expected_paths.append(path)
    if tuple(sorted(expected_paths)) != actual:
        raise ProcurementExtractionError("host procurement receipt artifact set changed")
    return tuple(expected_paths)


def extract_component_procurement_file(
    path: str | Path,
    *,
    source_name: str | None = None,
    provisions_by_product: Mapping[
        str, Sequence[ProcurementProvisionSpec] | ProcurementProvisionSpec
    ]
    | None = None,
) -> tuple[ProcurementRecord, ...]:
    component_path = Path(path)
    if component_path.is_symlink() or not component_path.is_file():
        raise ProcurementExtractionError(
            f"component input is not a regular file: {component_path}"
        )
    try:
        source = component_path.read_bytes()
    except OSError as exc:
        raise ProcurementExtractionError(
            f"cannot read component input {component_path}: {exc}"
        ) from exc
    return extract_component_procurement(
        source,
        source_name=component_path.as_posix() if source_name is None else source_name,
        provisions_by_product=provisions_by_product,
    )


def bind_record(
    record: ProcurementRecord,
    provisions: Iterable[ProcurementProvisionSpec],
) -> ProcurementRecord:
    """Return a record with caller-supplied exact provision bindings."""

    values = tuple(provisions)
    if any(not isinstance(item, ProcurementProvisionSpec) for item in values):
        raise ProcurementExtractionError("provisions contain an invalid record")
    return replace(record, provides=values)


def write_procurement_records(
    catalog_root: str | Path,
    records: Iterable[ProcurementRecord],
    *,
    overwrite: bool = False,
) -> ProcurementRegistry:
    """Write canonical central ``.procurement`` files with collision checks."""

    registry = ProcurementRegistry(records)
    records_root = Path(catalog_root) / "procurement" / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    for record in registry.values():
        path = records_root / f"{record.id}.procurement"
        rendered = record.to_json()
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ProcurementExtractionError(
                    f"procurement output is not a regular file: {path}"
                )
            existing = path.read_text(encoding="utf-8")
            if existing == rendered:
                continue
            if not overwrite:
                raise ProcurementExtractionError(
                    f"refusing to overwrite different procurement record: {path}"
                )
        path.write_text(rendered, encoding="utf-8")
    return registry


def migrate_component_inputs(
    paths: Iterable[str | Path],
    catalog_root: str | Path,
    *,
    source_root: str | Path | None = None,
    provisions_by_source: Mapping[
        str,
        Mapping[
            str, Sequence[ProcurementProvisionSpec] | ProcurementProvisionSpec
        ],
    ]
    | None = None,
    overwrite: bool = False,
) -> ProcurementRegistry:
    """Extract and write records from component inputs using explicit bindings.

    ``provisions_by_source`` keys are exact normalized evidence source names;
    unspecified sources remain unmodeled.  This makes a batch migration safe for
    mixed modeled/unmodeled corpora such as the Yageo/Murata import session.
    """

    root = None if source_root is None else Path(source_root).resolve()
    supplied = {} if provisions_by_source is None else dict(provisions_by_source)
    used_sources: set[str] = set()
    records: list[ProcurementRecord] = []
    for raw_path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        path = raw_path.resolve()
        if root is None:
            source_name = raw_path.as_posix()
        else:
            try:
                source_name = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ProcurementExtractionError(
                    f"component input escapes source_root: {path}"
                ) from exc
        bindings = supplied.get(source_name)
        if bindings is not None:
            used_sources.add(source_name)
        records.extend(
            extract_component_procurement_file(
                path,
                source_name=source_name,
                provisions_by_product=bindings,
            )
        )
    unused = sorted(set(supplied) - used_sources)
    if unused:
        raise ProcurementExtractionError(
            "provision bindings reference inputs outside the migration set: "
            + ", ".join(unused)
        )
    return write_procurement_records(
        catalog_root, records, overwrite=overwrite
    )


__all__ = [
    "HOST_PROCUREMENT_CONTEXT_FILENAME",
    "HOST_PROCUREMENT_CONTEXT_FORMAT",
    "HOST_PROCUREMENT_RECEIPT_FILENAME",
    "HOST_PROCUREMENT_RECEIPT_FORMAT",
    "ProcurementExtractionError",
    "ProcurementTextFallbackConfig",
    "bind_record",
    "extract_component_procurement",
    "extract_component_procurement_file",
    "extract_ecad_procurement",
    "extract_pdf_procurement",
    "extract_structured_procurement",
    "materialize_deferred_procurement",
    "materialize_proposal_procurement",
    "migrate_component_inputs",
    "procurement_fallback_identity",
    "proposal_procurement_receipt_path",
    "static_part_provision",
    "verify_proposal_procurement_receipt",
    "write_host_procurement_context",
    "write_procurement_records",
]
