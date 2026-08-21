"""First-class procurement and supplier identity records.

Procurement records intentionally live outside ``static.part``.  They may be
updated as lifecycle, supplier, price, or availability evidence changes without
changing the immutable physical assembly hash.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse


class ProcurementSpecError(ValueError):
    """A procurement record is malformed or not evidence-backed."""


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_SCHEMES = frozenset(
    {
        "manufacturer_part_number",
        "manufacturer_item_number",
        "supplier_sku",
        "product_name",
        "gtin",
        "upc",
        "ean",
        "standard_designation",
        "extension",
    }
)
_DOCUMENT_KINDS = frozenset(
    {"datasheet", "product_page", "purchase_page", "drawing", "certificate", "lifecycle_notice", "other"}
)
_AVAILABILITY = frozenset(
    {"in_stock", "backorder", "preorder", "out_of_stock", "discontinued", "unknown"}
)
_LIFECYCLE = frozenset(
    {"active", "not_recommended_for_new_design", "last_time_buy", "obsolete", "unknown"}
)


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ProcurementSpecError(f"{context} must be an object with string keys")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProcurementSpecError(f"{context} must be an array")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], required: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ProcurementSpecError(f"{context} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ProcurementSpecError(f"{context} is missing fields: {', '.join(missing)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ProcurementSpecError(f"{context} must be a nonempty trimmed string")
    return value


def _optional_text(value: Any, context: str) -> str | None:
    return None if value is None else _text(value, context)


def _id(value: Any, context: str) -> str:
    result = _text(value, context)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ProcurementSpecError(f"{context} must be an identifier")
    return result


def _digest(value: Any, context: str) -> str:
    result = _text(value, context)
    if _SHA256.fullmatch(result) is None:
        raise ProcurementSpecError(f"{context} must be 'sha256:' plus 64 lowercase hex digits")
    return result


def _url(value: Any, context: str) -> str:
    result = _text(value, context)
    parsed = urlparse(result)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProcurementSpecError(f"{context} must be an absolute HTTP(S) URL")
    return result


def _timestamp(value: Any, context: str) -> str:
    result = _text(value, context)
    normalized = result[:-1] + "+00:00" if result.endswith("Z") else result
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ProcurementSpecError(f"{context} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ProcurementSpecError(f"{context} must include a timezone")
    return result


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProcurementSpecError(f"{context} must be a positive integer")
    return value


def _positive_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProcurementSpecError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ProcurementSpecError(f"{context} must be positive and finite")
    return result


def _gtin_valid(value: str) -> bool:
    if not value.isdigit() or len(value) not in {8, 12, 13, 14}:
        return False
    digits = [int(item) for item in value]
    total = sum(
        digit * (3 if (len(digits) - index) % 2 == 0 else 1)
        for index, digit in enumerate(digits[:-1])
    )
    return (10 - total % 10) % 10 == digits[-1]


@dataclass(frozen=True, slots=True)
class ProcurementIdentifierSpec:
    scheme: str
    value: str
    issuer: str | None = None
    scheme_uri: str | None = None

    def __post_init__(self) -> None:
        if self.scheme not in _IDENTIFIER_SCHEMES:
            raise ProcurementSpecError(
                f"identifier.scheme must be one of {sorted(_IDENTIFIER_SCHEMES)}"
            )
        _text(self.value, "identifier.value")
        if self.issuer is not None:
            _text(self.issuer, "identifier.issuer")
        if self.scheme_uri is not None:
            _url(self.scheme_uri, "identifier.scheme_uri")
        if self.scheme in {"manufacturer_part_number", "manufacturer_item_number", "supplier_sku"} and self.issuer is None:
            raise ProcurementSpecError(f"{self.scheme} requires issuer")
        if self.scheme == "extension" and self.scheme_uri is None:
            raise ProcurementSpecError("extension identifiers require scheme_uri")
        if self.scheme in {"gtin", "upc", "ean"} and not _gtin_valid(self.value):
            raise ProcurementSpecError(f"{self.scheme} value has an invalid length or check digit")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcurementIdentifierSpec":
        data = _object(value, "procurement identifier")
        _keys(data, {"scheme", "value", "issuer", "scheme_uri"}, {"scheme", "value"}, "procurement identifier")
        return cls(
            _text(data["scheme"], "identifier.scheme"),
            _text(data["value"], "identifier.value"),
            _optional_text(data.get("issuer"), "identifier.issuer"),
            None if data.get("scheme_uri") is None else _url(data["scheme_uri"], "identifier.scheme_uri"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"scheme": self.scheme, "value": self.value}
        if self.issuer is not None:
            result["issuer"] = self.issuer
        if self.scheme_uri is not None:
            result["scheme_uri"] = self.scheme_uri
        return result


@dataclass(frozen=True, slots=True)
class ProcurementDocumentSpec:
    kind: str
    url: str
    media_type: str | None = None
    source_sha256: str | None = None
    extracted_text_sha256: str | None = None
    retrieved_at: str | None = None
    title: str | None = None
    page_count: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _DOCUMENT_KINDS:
            raise ProcurementSpecError(f"document.kind must be one of {sorted(_DOCUMENT_KINDS)}")
        _url(self.url, "document.url")
        for name in ("media_type", "title"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), f"document.{name}")
        for name in ("source_sha256", "extracted_text_sha256"):
            if getattr(self, name) is not None:
                _digest(getattr(self, name), f"document.{name}")
        if self.retrieved_at is not None:
            _timestamp(self.retrieved_at, "document.retrieved_at")
        if self.page_count is not None:
            _positive_integer(self.page_count, "document.page_count")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcurementDocumentSpec":
        data = _object(value, "procurement document")
        names = {"kind", "url", "media_type", "source_sha256", "extracted_text_sha256", "retrieved_at", "title", "page_count"}
        _keys(data, names, {"kind", "url"}, "procurement document")
        return cls(
            _text(data["kind"], "document.kind"),
            _url(data["url"], "document.url"),
            _optional_text(data.get("media_type"), "document.media_type"),
            None if data.get("source_sha256") is None else _digest(data["source_sha256"], "document.source_sha256"),
            None if data.get("extracted_text_sha256") is None else _digest(data["extracted_text_sha256"], "document.extracted_text_sha256"),
            None if data.get("retrieved_at") is None else _timestamp(data["retrieved_at"], "document.retrieved_at"),
            _optional_text(data.get("title"), "document.title"),
            None if data.get("page_count") is None else _positive_integer(data["page_count"], "document.page_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "url": self.url}
        for name in ("media_type", "source_sha256", "extracted_text_sha256", "retrieved_at", "title", "page_count"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class ProcurementOfferSpec:
    supplier: str
    supplier_part_number: str
    purchase_url: str
    observed_at: str
    availability: str
    currency: str | None = None
    unit_price: float | None = None
    minimum_order_quantity: int | None = None

    def __post_init__(self) -> None:
        _text(self.supplier, "offer.supplier")
        _text(self.supplier_part_number, "offer.supplier_part_number")
        _url(self.purchase_url, "offer.purchase_url")
        _timestamp(self.observed_at, "offer.observed_at")
        if self.availability not in _AVAILABILITY:
            raise ProcurementSpecError(f"offer.availability must be one of {sorted(_AVAILABILITY)}")
        if (self.currency is None) != (self.unit_price is None):
            raise ProcurementSpecError("offer.currency and unit_price must be supplied together")
        if self.currency is not None and re.fullmatch(r"[A-Z]{3}", self.currency) is None:
            raise ProcurementSpecError("offer.currency must be an ISO-4217-style code")
        if self.unit_price is not None:
            _positive_number(self.unit_price, "offer.unit_price")
        if self.minimum_order_quantity is not None:
            _positive_integer(self.minimum_order_quantity, "offer.minimum_order_quantity")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcurementOfferSpec":
        data = _object(value, "procurement offer")
        names = {"supplier", "supplier_part_number", "purchase_url", "observed_at", "availability", "currency", "unit_price", "minimum_order_quantity"}
        _keys(data, names, {"supplier", "supplier_part_number", "purchase_url", "observed_at", "availability"}, "procurement offer")
        return cls(
            _text(data["supplier"], "offer.supplier"),
            _text(data["supplier_part_number"], "offer.supplier_part_number"),
            _url(data["purchase_url"], "offer.purchase_url"),
            _timestamp(data["observed_at"], "offer.observed_at"),
            _text(data["availability"], "offer.availability"),
            _optional_text(data.get("currency"), "offer.currency"),
            None if data.get("unit_price") is None else _positive_number(data["unit_price"], "offer.unit_price"),
            None if data.get("minimum_order_quantity") is None else _positive_integer(data["minimum_order_quantity"], "offer.minimum_order_quantity"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "supplier": self.supplier,
            "supplier_part_number": self.supplier_part_number,
            "purchase_url": self.purchase_url,
            "observed_at": self.observed_at,
            "availability": self.availability,
        }
        for name in ("currency", "unit_price", "minimum_order_quantity"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class ProcurementLifecycleSpec:
    status: str
    observed_at: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _LIFECYCLE:
            raise ProcurementSpecError(f"lifecycle.status must be one of {sorted(_LIFECYCLE)}")
        if self.status != "unknown" and (self.observed_at is None or self.source_url is None):
            raise ProcurementSpecError("a known lifecycle status requires observed_at and source_url")
        if self.observed_at is not None:
            _timestamp(self.observed_at, "lifecycle.observed_at")
        if self.source_url is not None:
            _url(self.source_url, "lifecycle.source_url")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcurementLifecycleSpec":
        data = _object(value, "procurement lifecycle")
        _keys(data, {"status", "observed_at", "source_url"}, {"status"}, "procurement lifecycle")
        return cls(
            _text(data["status"], "lifecycle.status"),
            None if data.get("observed_at") is None else _timestamp(data["observed_at"], "lifecycle.observed_at"),
            None if data.get("source_url") is None else _url(data["source_url"], "lifecycle.source_url"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status}
        if self.observed_at is not None:
            result["observed_at"] = self.observed_at
        if self.source_url is not None:
            result["source_url"] = self.source_url
        return result


@dataclass(frozen=True, slots=True)
class ProcurementProvisionSpec:
    part: str
    version: str
    static_sha256: str
    quantity: int

    def __post_init__(self) -> None:
        _id(self.part, "provision.part")
        _text(self.version, "provision.version")
        _digest(self.static_sha256, "provision.static_sha256")
        _positive_integer(self.quantity, "provision.quantity")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcurementProvisionSpec":
        data = _object(value, "procurement provision")
        names = {"part", "version", "static_sha256", "quantity"}
        _keys(data, names, names, "procurement provision")
        return cls(
            _id(data["part"], "provision.part"),
            _text(data["version"], "provision.version"),
            _digest(data["static_sha256"], "provision.static_sha256"),
            _positive_integer(data["quantity"], "provision.quantity"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "version": self.version,
            "static_sha256": self.static_sha256,
            "quantity": self.quantity,
        }


@dataclass(frozen=True, slots=True)
class ProcurementEvidenceSpec:
    source: str
    sha256: str
    locator: str | None = None

    def __post_init__(self) -> None:
        _text(self.source, "procurement evidence.source")
        _digest(self.sha256, "procurement evidence.sha256")
        if self.locator is not None:
            _text(self.locator, "procurement evidence.locator")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcurementEvidenceSpec":
        data = _object(value, "procurement evidence")
        _keys(data, {"source", "sha256", "locator"}, {"source", "sha256"}, "procurement evidence")
        return cls(
            _text(data["source"], "procurement evidence.source"),
            _digest(data["sha256"], "procurement evidence.sha256"),
            _optional_text(data.get("locator"), "procurement evidence.locator"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"source": self.source, "sha256": self.sha256}
        if self.locator is not None:
            result["locator"] = self.locator
        return result


@dataclass(frozen=True, slots=True)
class ProcurementRecord:
    format: str
    id: str
    version: str
    manufacturer: str | None
    identifiers: tuple[ProcurementIdentifierSpec, ...]
    documents: tuple[ProcurementDocumentSpec, ...]
    offers: tuple[ProcurementOfferSpec, ...]
    lifecycle: ProcurementLifecycleSpec
    provides: tuple[ProcurementProvisionSpec, ...]
    evidence: tuple[ProcurementEvidenceSpec, ...]

    def __post_init__(self) -> None:
        if self.format != "procurement-record-1":
            raise ProcurementSpecError("procurement format must be procurement-record-1")
        _id(self.id, "procurement.id")
        _text(self.version, "procurement.version")
        if self.manufacturer is not None:
            _text(self.manufacturer, "procurement.manufacturer")
        for name, kind in (
            ("identifiers", ProcurementIdentifierSpec),
            ("documents", ProcurementDocumentSpec),
            ("offers", ProcurementOfferSpec),
            ("provides", ProcurementProvisionSpec),
            ("evidence", ProcurementEvidenceSpec),
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, kind) for item in values):
                raise ProcurementSpecError(f"procurement.{name} has an invalid record")
            object.__setattr__(self, name, values)
        if not self.identifiers:
            raise ProcurementSpecError("procurement records require at least one evidenced identifier")
        if not self.evidence:
            raise ProcurementSpecError("procurement records require source evidence")
        identifier_keys = [(item.scheme, item.issuer, item.value) for item in self.identifiers]
        if len(identifier_keys) != len(set(identifier_keys)):
            raise ProcurementSpecError("procurement identifiers must be unique")
        provision_parts = [item.part for item in self.provides]
        if len(provision_parts) != len(set(provision_parts)):
            raise ProcurementSpecError("a procurement record may provide each part only once")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcurementRecord":
        data = _object(value, "procurement record")
        names = {"format", "id", "version", "manufacturer", "identifiers", "documents", "offers", "lifecycle", "provides", "evidence"}
        _keys(data, names, names, "procurement record")
        return cls(
            _text(data["format"], "procurement.format"),
            _id(data["id"], "procurement.id"),
            _text(data["version"], "procurement.version"),
            _optional_text(data["manufacturer"], "procurement.manufacturer"),
            tuple(ProcurementIdentifierSpec.from_dict(_object(item, f"procurement.identifiers[{index}]")) for index, item in enumerate(_sequence(data["identifiers"], "procurement.identifiers"))),
            tuple(ProcurementDocumentSpec.from_dict(_object(item, f"procurement.documents[{index}]")) for index, item in enumerate(_sequence(data["documents"], "procurement.documents"))),
            tuple(ProcurementOfferSpec.from_dict(_object(item, f"procurement.offers[{index}]")) for index, item in enumerate(_sequence(data["offers"], "procurement.offers"))),
            ProcurementLifecycleSpec.from_dict(_object(data["lifecycle"], "procurement.lifecycle")),
            tuple(ProcurementProvisionSpec.from_dict(_object(item, f"procurement.provides[{index}]")) for index, item in enumerate(_sequence(data["provides"], "procurement.provides"))),
            tuple(ProcurementEvidenceSpec.from_dict(_object(item, f"procurement.evidence[{index}]")) for index, item in enumerate(_sequence(data["evidence"], "procurement.evidence"))),
        )

    @classmethod
    def from_json(cls, source: str) -> "ProcurementRecord":
        try:
            value = json.loads(source, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise ProcurementSpecError(
                f"invalid procurement JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        return cls.from_dict(_object(value, "procurement record"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "id": self.id,
            "version": self.version,
            "manufacturer": self.manufacturer,
            "identifiers": [item.to_dict() for item in self.identifiers],
            "documents": [item.to_dict() for item in self.documents],
            "offers": [item.to_dict() for item in self.offers],
            "lifecycle": self.lifecycle.to_dict(),
            "provides": [item.to_dict() for item in self.provides],
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProcurementSpecError(f"duplicate procurement JSON field {key!r}")
        result[key] = value
    return result


class ProcurementRegistry(Mapping[str, ProcurementRecord]):
    def __init__(self, records: Iterable[ProcurementRecord] = ()) -> None:
        values: dict[str, ProcurementRecord] = {}
        for record in records:
            if record.id in values:
                raise ProcurementSpecError(f"duplicate procurement record id {record.id!r}")
            values[record.id] = record
        self._records = values

    def __getitem__(self, key: str) -> ProcurementRecord:
        return self._records[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def sha256(self) -> str:
        payload = [self._records[key].to_dict() for key in sorted(self._records)]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def for_part(self, part_id: str) -> tuple[ProcurementRecord, ...]:
        return tuple(
            record
            for record in self._records.values()
            if any(item.part == part_id for item in record.provides)
        )

    def validate_provisions(
        self, parts: Mapping[str, tuple[str, str]]
    ) -> None:
        """Validate every provision against an exact static part closure.

        ``parts`` maps a static part ID to ``(version, canonical_sha256)``.
        Keeping that small projection here avoids a catalog import cycle while
        still making stale or cross-catalog procurement bindings fail closed.
        Empty ``provides`` arrays are valid: an evidenced product may be useful
        for classification or future modeling without claiming a physical part.
        """

        for record in self._records.values():
            for provision in record.provides:
                try:
                    version, digest = parts[provision.part]
                except KeyError as exc:
                    raise ProcurementSpecError(
                        f"procurement record {record.id!r} provides unknown static "
                        f"part {provision.part!r}"
                    ) from exc
                if provision.version != version or provision.static_sha256 != digest:
                    raise ProcurementSpecError(
                        f"procurement record {record.id!r} provision for "
                        f"{provision.part!r} is stale: expected {version} {digest}, "
                        f"got {provision.version} {provision.static_sha256}"
                    )

    @classmethod
    def load_catalog(cls, root: str | Path) -> "ProcurementRegistry":
        catalog_root = Path(root).resolve()
        records: list[ProcurementRecord] = []
        for path in sorted(catalog_root.rglob("*.procurement")):
            if path.is_symlink() or not path.is_file():
                raise ProcurementSpecError(f"procurement record is not a regular file: {path}")
            try:
                relative = path.relative_to(catalog_root)
            except ValueError as exc:  # pragma: no cover - rglob guarantees this.
                raise ProcurementSpecError(f"procurement record escapes catalog: {path}") from exc
            if path.parent.parent.name != "instantiations":
                raise ProcurementSpecError(
                    f"{relative.as_posix()}: procurement records must be directly "
                    "inside a part instantiation directory"
                )
            record = ProcurementRecord.from_json(path.read_text(encoding="utf-8"))
            if path.stem != record.id:
                raise ProcurementSpecError(f"{path.name}: filename stem must equal record id {record.id!r}")
            records.append(record)
        return cls(records)


__all__ = [
    "ProcurementDocumentSpec",
    "ProcurementEvidenceSpec",
    "ProcurementIdentifierSpec",
    "ProcurementLifecycleSpec",
    "ProcurementOfferSpec",
    "ProcurementProvisionSpec",
    "ProcurementRecord",
    "ProcurementRegistry",
    "ProcurementSpecError",
]
