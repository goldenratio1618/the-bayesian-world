from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from contraption.catalog.interfaces import interface_paths
from contraption.part_import.agents import ModelingAgent, ModelingInputs
from contraption.part_import.budget import BudgetLedger
from contraption.part_import.deterministic_assets import (
    DeterministicAssetError,
    bundle_staged_plan,
    modeling_context_paths,
    stage_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = PROJECT_ROOT / "model_catalog"
from contraption.part_import.document_ingestion import (
    DeterministicDocumentError,
    PDFLimits,
    extract_pdf,
    extract_supported_document,
    normalize_extracted_text,
)
from contraption.part_import.ecad_ingestion import extract_ecad


def _minimal_text_pdf(text: str = "Part 123") -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode("ascii"))
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(payload)


def test_pdf_extraction_validates_normalizes_and_hashes_each_page(tmp_path: Path) -> None:
    source = tmp_path / "vendor.pdf"
    source.write_bytes(_minimal_text_pdf("RC0603FR-0710KL"))

    extracted = extract_pdf(source)

    assert extracted["format"] == "deterministic-pdf-extraction-1"
    assert extracted["source"]["sha256"] == "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    assert extracted["page_count"] == 1
    assert extracted["pages"][0]["text"] == "RC0603FR-0710KL"
    assert extracted["pages"][0]["text_sha256"] == "sha256:" + hashlib.sha256(
        b"RC0603FR-0710KL"
    ).hexdigest()
    assert extracted["parser"]["strict"] is True


def test_pdf_text_normalization_removes_layout_controls_without_merging_lines() -> None:
    assert normalize_extracted_text(" \u200f A\t B \r\n\r\n\r\n C\x00 ") == "A B\n\nC"


def test_pdf_reports_clear_optional_dependency_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contraption.part_import import document_ingestion

    source = tmp_path / "vendor.pdf"
    source.write_bytes(_minimal_text_pdf())
    real_import = document_ingestion.importlib.import_module

    def import_without_pypdf(name: str):
        if name == "pypdf":
            raise ImportError("fixture")
        return real_import(name)

    monkeypatch.setattr(document_ingestion.importlib, "import_module", import_without_pypdf)
    with pytest.raises(DeterministicDocumentError, match=r"optional 'documents' dependency"):
        extract_pdf(source)


def test_pdf_preflight_rejects_wrong_magic(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.pdf"
    source.write_bytes(b"not a pdf")
    with pytest.raises(DeterministicDocumentError, match="magic"):
        extract_pdf(source)


def test_pdf_rejects_structural_javascript_action(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    source = tmp_path / "javascript.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_js("app.alert('unsafe')")
    with source.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(DeterministicDocumentError, match="forbidden"):
        extract_pdf(source)


def test_pdf_rejects_encryption_and_page_limit(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    encrypted = tmp_path / "encrypted.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    with encrypted.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(DeterministicDocumentError, match="encrypted"):
        extract_pdf(encrypted)

    too_many = tmp_path / "too-many.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with too_many.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(DeterministicDocumentError, match="page count"):
        extract_pdf(too_many, limits=PDFLimits(max_pages=1))


def test_pdf_rejects_pypdf_embedded_attachment(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    source = tmp_path / "attachment.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_attachment("payload.txt", b"not allowed")
    with source.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(DeterministicDocumentError, match="forbidden"):
        extract_pdf(source)


def test_librepcb_extraction_preserves_explicit_mpn_manufacturer_and_resource(
    tmp_path: Path,
) -> None:
    source = tmp_path / "device.lp"
    source.write_text(
        """(librepcb_device 07af776f-e2f9-4e0c-9a66-f0f25159bace
 (name "1N4006")
 (resource "Datasheet 1N4006" (mediatype "application/pdf")
  (url "https://example.test/1n4006.pdf"))
 (part "1N4006RLG" (manufacturer "onsemi"))
 (part "1N4006-E3/54" (manufacturer "Vishay"))
)\n""",
        encoding="utf-8",
    )

    extracted = extract_ecad(source)

    assert extracted["source"]["kind"] == "librepcb"
    assert [item["manufacturer_part_number"]["value"] for item in extracted["records"]] == [
        "1N4006RLG",
        "1N4006-E3/54",
    ]
    assert [item["manufacturer"]["value"] for item in extracted["records"]] == [
        "onsemi",
        "Vishay",
    ]
    assert extracted["documents"][0]["url"] == "https://example.test/1n4006.pdf"
    assert extracted["records"][0]["manufacturer_part_number"]["evidence"]["source_sha256"] == extracted["source"]["sha256"]
    # Device display names are not silently promoted to purchasable identifiers.
    assert all("name" not in item for item in extracted["records"])


def test_kicad_extraction_uses_only_explicit_identity_properties(tmp_path: Path) -> None:
    source = tmp_path / "vendor.kicad_sym"
    source.write_text(
        """(kicad_symbol_lib (version 20251024)
 (symbol "14528"
  (property "Value" "tempting-but-not-an-mpn")
  (property "Footprint" "")
  (property "Manufacturer" "onsemi")
  (property "MPN" "MC14528BDR2G")
  (property "Datasheet" "https://example.test/MC14528B-D.PDF"))
 (symbol "generic"
  (property "Value" "generic")
  (property "Datasheet" "local-file.pdf"))
)\n""",
        encoding="utf-8",
    )

    extracted = extract_ecad(source, source_format="kicad")

    first, second = extracted["records"]
    assert first["catalog_identifier"]["value"] == "14528"
    assert first["manufacturer"]["value"] == "onsemi"
    assert first["manufacturer_part_number"]["value"] == "MC14528BDR2G"
    assert first["documents"][0]["url"].startswith("https://")
    assert "manufacturer_part_number" not in second
    assert second["documents"] == []
    assert extracted["omissions"] == [
        {
            "reason": "datasheet value is not an absolute HTTP(S) URL",
            "locator": "symbol[1].property[1]",
        }
    ]


def test_document_plan_stages_extraction_not_raw_pdf_and_detects_tamper(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "vendor.pdf").write_bytes(_minimal_text_pdf("evidence"))
    component = input_root / "component.json"
    component.write_text(
        json.dumps(
            {
                "deterministic_ingestion": {
                    "format": "deterministic-part-ingestion-1",
                    "documents": [{"source": "vendor.pdf", "source_format": "pdf"}],
                }
            }
        ),
        encoding="utf-8",
    )

    plan = stage_plan(component, tmp_path / "staged")

    assert plan is not None
    assert not tuple(plan.parent.rglob("*.pdf"))
    contexts = modeling_context_paths(plan)
    assert len(contexts) == 1
    assert json.loads(contexts[0].read_text())["pages"][0]["text"] == "evidence"
    assert bundle_staged_plan(tmp_path / "candidate", plan) == ()

    contexts[0].write_text("{}", encoding="utf-8")
    with pytest.raises(DeterministicAssetError, match="hash mismatch"):
        modeling_context_paths(plan)


def test_modeling_workspace_exposes_only_verified_document_extraction(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    raw_pdf = input_root / "vendor.pdf"
    raw_pdf.write_bytes(_minimal_text_pdf("RC0603FR-0710KL"))
    component = input_root / "pdf_component.json"
    component.write_text(
        json.dumps(
            {
                "domains": ["electrical"],
                "deterministic_ingestion": {
                    "format": "deterministic-part-ingestion-1",
                    "documents": [{"source": "vendor.pdf", "source_format": "pdf"}],
                },
            }
        ),
        encoding="utf-8",
    )
    inputs = ModelingInputs(
        constraints=PROJECT_ROOT / "prompts" / "model_constraints.md",
        gold_templates=(CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl",),
        interfaces=interface_paths(CATALOG_ROOT),
        direct_hierarchy=(CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl",),
        component_information=component,
    )
    agent = ModelingAgent(
        BudgetLedger(tmp_path / "ledger.json", 1.0), tmp_path / "staging"
    )

    workspace = agent.prepare_workspace(inputs, "document-context")

    instructions = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    manifest = json.loads((workspace / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    assert "deterministic-pdf-extraction-1" in instructions
    assert "RC0603FR-0710KL" in instructions
    assert "%PDF-1.4" not in instructions
    assert all(item["source_label"] != "vendor.pdf" for item in manifest)
    assert any(item["source_label"] == "extraction.json" for item in manifest)
    assert not tuple(workspace.parent.rglob("*.pdf"))


def test_unknown_document_format_never_falls_through_as_opaque_input(tmp_path: Path) -> None:
    source = tmp_path / "vendor.bin"
    source.write_bytes(b"opaque binary")
    with pytest.raises(DeterministicDocumentError, match="unsupported"):
        extract_supported_document(source)
