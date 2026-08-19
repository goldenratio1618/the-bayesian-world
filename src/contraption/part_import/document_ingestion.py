"""Host-owned deterministic extraction for documents used by part import.

Binary document bytes are inspected and reduced to a bounded, hashed JSON
record.  Only that extracted record is suitable for a Luna workspace; a raw PDF
is evidence retained by the host and is never modeling-agent input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import hashlib
import importlib
import json
from pathlib import Path
import unicodedata
from typing import Any

from .dxf_ingestion import DXFLimits, DeterministicDxfError, extract_dxf
from .ecad_ingestion import DeterministicEcadError, extract_ecad


class DeterministicDocumentError(ValueError):
    """Raised when deterministic document ingestion must fail closed."""


@dataclass(frozen=True, slots=True)
class PDFLimits:
    max_bytes: int = 32 * 1024 * 1024
    max_pages: int = 512
    max_characters_per_page: int = 2_000_000
    max_total_characters: int = 16_000_000
    max_object_nodes: int = 250_000
    max_object_depth: int = 128

    def __post_init__(self) -> None:
        for name in (
            "max_bytes",
            "max_pages",
            "max_characters_per_page",
            "max_total_characters",
            "max_object_nodes",
            "max_object_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DeterministicDocumentError(f"{name} must be a positive integer")


_BLOCKED_KEYS = {
    "/JavaScript",
    "/JS",
    "/EmbeddedFiles",
    "/EmbeddedFile",
    "/EF",
    "/XFA",
    "/RichMediaContent",
}
_BLOCKED_TYPES = {
    "/EmbeddedFile",
    "/Filespec",
    "/FileAttachment",
    "/RichMedia",
    "/Movie",
    "/Sound",
}
_BLOCKED_ACTIONS = {
    "/JavaScript",
    "/Launch",
    "/SubmitForm",
    "/ImportData",
    "/GoToE",
    "/Rendition",
}


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_regular_file(source: str | Path, *, max_bytes: int) -> tuple[Path, bytes]:
    path = Path(source).resolve()
    if not path.is_file() or path.is_symlink():
        raise DeterministicDocumentError(
            f"document source is missing or not a regular file: {path}"
        )
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise DeterministicDocumentError(
            f"document source size must be between 1 and {max_bytes} bytes: {path}"
        )
    payload = path.read_bytes()
    if len(payload) != size:
        raise DeterministicDocumentError(f"document source changed while being read: {path}")
    return path, payload


def normalize_extracted_text(value: str) -> str:
    """Return stable plain text suitable for hashing and inert model context."""

    if not isinstance(value, str):
        raise DeterministicDocumentError("PDF text extractor returned a non-string value")
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    cleaned: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if char == "\n":
            cleaned.append(char)
        elif char == "\t" or category == "Zs":
            cleaned.append(" ")
        elif category in {"Cc", "Cf"}:
            # Layout controls, NULs, bidi overrides, and zero-width characters
            # have no evidence value in normalized datasheet text.
            cleaned.append(" ")
        else:
            cleaned.append(char)
    lines = [" ".join(line.split()) for line in "".join(cleaned).split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    compact: list[str] = []
    for line in lines:
        if line or not compact or compact[-1]:
            compact.append(line)
    return "\n".join(compact)


def _load_pypdf() -> tuple[Any, str]:
    try:
        module = importlib.import_module("pypdf")
    except ImportError as exc:
        raise DeterministicDocumentError(
            "PDF ingestion requires the optional 'documents' dependency; "
            "install the project with [documents]"
        ) from exc
    reader = getattr(module, "PdfReader", None)
    if reader is None:
        raise DeterministicDocumentError("installed pypdf does not expose PdfReader")
    return reader, str(getattr(module, "__version__", "unknown"))


def _indirect_key(value: Any) -> tuple[int, int] | None:
    identifier = getattr(value, "idnum", None)
    generation = getattr(value, "generation", None)
    if isinstance(identifier, int) and isinstance(generation, int):
        return identifier, generation
    return None


def _resolve_pdf_object(value: Any) -> Any:
    if _indirect_key(value) is None:
        return value
    try:
        return value.get_object()
    except Exception as exc:  # pypdf exposes several parse-specific exceptions
        raise DeterministicDocumentError("PDF contains an unreadable indirect object") from exc


def _scan_pdf_objects(roots: Sequence[Any], limits: PDFLimits) -> None:
    pending: list[tuple[Any, int, str]] = [
        (root, 0, "catalog" if index == 0 else f"page[{index}]")
        for index, root in enumerate(roots)
    ]
    visited_indirect: set[tuple[int, int]] = set()
    visited_direct: set[int] = set()
    nodes = 0
    while pending:
        value, depth, context = pending.pop()
        if depth > limits.max_object_depth:
            raise DeterministicDocumentError("PDF object graph exceeds the depth safety limit")
        indirect = _indirect_key(value)
        if indirect is not None:
            if indirect in visited_indirect:
                continue
            visited_indirect.add(indirect)
        value = _resolve_pdf_object(value)
        if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
            continue
        direct = id(value)
        if direct in visited_direct:
            continue
        visited_direct.add(direct)
        nodes += 1
        if nodes > limits.max_object_nodes:
            raise DeterministicDocumentError("PDF object graph exceeds the node safety limit")
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                if key in _BLOCKED_KEYS:
                    raise DeterministicDocumentError(
                        f"PDF active or embedded content is forbidden ({key} at {context})"
                    )
                resolved_child = _resolve_pdf_object(child)
                child_name = str(resolved_child)
                if key in {"/Type", "/Subtype"} and child_name in _BLOCKED_TYPES:
                    raise DeterministicDocumentError(
                        f"PDF embedded or multimedia object is forbidden ({child_name} at {context})"
                    )
                if key == "/S" and child_name in _BLOCKED_ACTIONS:
                    raise DeterministicDocumentError(
                        f"PDF active action is forbidden ({child_name} at {context})"
                    )
                # Parent links only point back up the already-scanned page tree.
                if key != "/Parent":
                    pending.append((child, depth + 1, context + key))
        elif isinstance(value, Sequence):
            pending.extend(
                (child, depth + 1, f"{context}[{index}]")
                for index, child in enumerate(value)
            )


def extract_pdf(source: str | Path, *, limits: PDFLimits = PDFLimits()) -> dict[str, Any]:
    """Validate a PDF and return normalized, per-page hashed plain text.

    Encryption, JavaScript/actions, embedded files, multimedia, malformed
    structure, and configured size/page/text limits all fail closed.
    """

    path, payload = _read_regular_file(source, max_bytes=limits.max_bytes)
    if not payload.startswith(b"%PDF-"):
        raise DeterministicDocumentError("PDF source does not begin with the PDF magic header")
    reader_class, parser_version = _load_pypdf()
    try:
        reader = reader_class(BytesIO(payload), strict=True)
    except Exception as exc:
        raise DeterministicDocumentError(f"PDF parser rejected {path.name}: {exc}") from exc
    try:
        if bool(reader.is_encrypted):
            raise DeterministicDocumentError("encrypted PDFs are not accepted")
        page_count = len(reader.pages)
    except DeterministicDocumentError:
        raise
    except Exception as exc:
        raise DeterministicDocumentError("PDF page tree is unreadable") from exc
    if page_count <= 0 or page_count > limits.max_pages:
        raise DeterministicDocumentError(
            f"PDF page count must be between 1 and {limits.max_pages}"
        )
    try:
        root = reader.trailer["/Root"]
    except Exception as exc:
        raise DeterministicDocumentError("PDF catalog root is missing or unreadable") from exc
    _scan_pdf_objects((root, *tuple(reader.pages)), limits)

    pages: list[dict[str, Any]] = []
    total_characters = 0
    for index, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise DeterministicDocumentError(
                f"PDF text extraction failed on page {index + 1}"
            ) from exc
        text = normalize_extracted_text(raw_text)
        characters = len(text)
        if characters > limits.max_characters_per_page:
            raise DeterministicDocumentError(
                f"PDF page {index + 1} exceeds the extracted-text safety limit"
            )
        total_characters += characters
        if total_characters > limits.max_total_characters:
            raise DeterministicDocumentError("PDF exceeds the total extracted-text safety limit")
        encoded = text.encode("utf-8")
        pages.append(
            {
                "number": index + 1,
                "characters": characters,
                "text_sha256": _sha256(encoded),
                "text": text,
            }
        )
    joined = "\f".join(page["text"] for page in pages).encode("utf-8")
    return {
        "format": "deterministic-pdf-extraction-1",
        "source": {
            "name": path.name,
            "media_type": "application/pdf",
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        "parser": {"name": "pypdf", "version": parser_version, "strict": True},
        "page_count": page_count,
        "text_sha256": _sha256(joined),
        "pages": pages,
    }


def extract_supported_document(
    source: str | Path,
    *,
    source_format: str = "auto",
    pdf_limits: PDFLimits = PDFLimits(),
    dxf_limits: DXFLimits = DXFLimits(),
) -> dict[str, Any]:
    """Dispatch one declared source to a deterministic parser.

    Supported values are auto, pdf, dxf, kicad, and librepcb.
    Automatic detection uses PDF magic, then the two structured ECAD roots;
    unsupported formats fail instead of being sent through as opaque bytes.
    """

    if source_format not in {"auto", "pdf", "dxf", "kicad", "librepcb"}:
        raise DeterministicDocumentError(
            "document source_format must be auto, pdf, dxf, kicad, or librepcb"
        )
    path = Path(source).resolve()
    if source_format == "pdf":
        return extract_pdf(path, limits=pdf_limits)
    if source_format == "dxf":
        try:
            return extract_dxf(path, limits=dxf_limits)
        except DeterministicDxfError as exc:
            raise DeterministicDocumentError(str(exc)) from exc
    if source_format in {"kicad", "librepcb"}:
        try:
            return extract_ecad(path, source_format=source_format)
        except DeterministicEcadError as exc:
            raise DeterministicDocumentError(str(exc)) from exc
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
    except OSError as exc:
        raise DeterministicDocumentError(f"document source cannot be read: {path}") from exc
    if prefix.startswith(b"%PDF-") or path.suffix.casefold() == ".pdf":
        return extract_pdf(path, limits=pdf_limits)
    if path.suffix.casefold() == ".dxf" or prefix.startswith(b"AutoCAD Binary DXF"):
        try:
            return extract_dxf(path, limits=dxf_limits)
        except DeterministicDxfError as exc:
            raise DeterministicDocumentError(str(exc)) from exc
    try:
        return extract_ecad(path, source_format="auto")
    except DeterministicEcadError as exc:
        raise DeterministicDocumentError(
            f"unsupported deterministic document format for {path.name}: {exc}"
        ) from exc


def canonical_extraction_bytes(extraction: Mapping[str, Any]) -> bytes:
    """Serialize an extracted record canonically for hashing and staging."""

    return (
        json.dumps(
            dict(extraction),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "DXFLimits",
    "DeterministicDocumentError",
    "PDFLimits",
    "canonical_extraction_bytes",
    "extract_pdf",
    "extract_supported_document",
    "normalize_extracted_text",
]
