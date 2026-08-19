from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from contraption.catalog.instantiations import (
    ModelInstantiationSpec,
    PartInstantiation,
    PartInstantiationRegistry,
    StaticPartSpec,
)
from contraption.catalog.procurement import (
    ProcurementProvisionSpec,
    ProcurementRecord,
    ProcurementRegistry,
    ProcurementSpecError,
)
from contraption.part_import.procurement_extraction import (
    ProcurementExtractionError,
    ProcurementTextFallbackConfig,
    extract_component_procurement,
    extract_component_procurement_file,
    extract_ecad_procurement,
    extract_pdf_procurement,
    migrate_component_inputs,
    static_part_provision,
    write_procurement_records,
)
from contraption.physics.physical import PhysicalSpecError
from scripts.generate_procurement_records import generate


ROOT = Path(__file__).resolve().parents[1]
PRIOR_INPUTS = (
    ROOT
    / "assembled_contraptions"
    / "part_import_2026_08_18"
    / "component_inputs"
)
RESISTOR_INSTANTIATIONS = (
    ROOT
    / "model_catalog"
    / "electrical"
    / "resistors"
    / "fixed_resistors"
    / "instantiations"
)
RESERVED_IDENTITY = {
    "datasheet_url",
    "datasheet_urls",
    "manufacturer",
    "manufacturer_item_number",
    "manufacturer_part_number",
    "mpn",
    "part_number",
    "product",
    "purchase_url",
    "purchase_urls",
    "source_urls",
    "supplier",
    "supplier_sku",
}


def _fact(value: str, locator: str, source_sha256: str) -> dict:
    return {
        "value": value,
        "evidence": {
            "source_sha256": source_sha256,
            "locator": locator,
            "line": 1,
            "column": 1,
            "value_sha256": "sha256:"
            + hashlib.sha256(value.encode("utf-8")).hexdigest(),
        },
    }


def _ecad_extraction(*, manufacturer: str | None = "Acme", mpn: str = "AX-100") -> bytes:
    source_sha256 = "sha256:" + hashlib.sha256(b"canonical-symbol.kicad_sym").hexdigest()
    record = {
        "manufacturer_part_number": _fact(
            mpn, "$.symbols[0].properties.MPN", source_sha256
        ),
        "documents": [],
    }
    if manufacturer is not None:
        record["manufacturer"] = _fact(
            manufacturer, "$.symbols[0].properties.Manufacturer", source_sha256
        )
    return json.dumps(
        {
            "format": "deterministic-ecad-extraction-1",
            "source": {
                "name": "canonical-symbol.kicad_sym",
                "sha256": source_sha256,
            },
            "records": [record],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pdf_extraction_pages(*texts: str) -> bytes:
    source_sha256 = "sha256:" + hashlib.sha256(b"raw-pdf-placeholder").hexdigest()
    pages = []
    for index, text in enumerate(texts, 1):
        pages.append(
            {
                "number": index,
                "text": text,
                "characters": len(text),
                "text_sha256": "sha256:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    joined = "\f".join(texts).encode("utf-8")
    return json.dumps(
        {
            "format": "deterministic-pdf-extraction-1",
            "source": {"name": "datasheet.pdf", "sha256": source_sha256},
            "page_count": len(pages),
            "pages": pages,
            "text_sha256": "sha256:" + hashlib.sha256(joined).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pdf_extraction(text: str) -> bytes:
    return _pdf_extraction_pages(text)


class _MockChatCompletions:
    def __init__(self, *values: dict) -> None:
        self.values = list(values)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))
            ]
        )


def _mock_chat_client(*values: dict) -> tuple[SimpleNamespace, _MockChatCompletions]:
    completions = _MockChatCompletions(*values)
    return (
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        completions,
    )


def _static_v2(path: Path) -> StaticPartSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format"] = "static-part-2"
    data.pop("purchasing", None)
    metadata = data.setdefault("metadata", {})
    for name in RESERVED_IDENTITY:
        metadata.pop(name, None)
    return StaticPartSpec.from_dict(data)


def _exact_yageo_static(component: dict) -> StaticPartSpec:
    product = component["product"]
    source_urls = set(component["source_urls"])
    candidates: list[Path] = []
    for path in RESISTOR_INSTANTIATIONS.glob("*/static.part"):
        data = json.loads(path.read_text(encoding="utf-8"))
        explicit_products = {
            section.get("product")
            for section in (data.get("purchasing", {}), data.get("metadata", {}))
            if isinstance(section, dict) and isinstance(section.get("product"), str)
        }
        reference = data.get("provenance", {}).get("reference")
        if product in explicit_products or reference in source_urls:
            candidates.append(path)
    assert len(candidates) == 1, (product, candidates)
    return _static_v2(candidates[0])


def test_evidence_only_extraction_preserves_identifiers_and_documents() -> None:
    payload = {
        "manufacturer": "Yageo",
        "product": "RC0603FR-071KL",
        "source_urls": [
            "https://www.yageo.com/en/ProductSearch/PartNumberSearch?partNo=RC0603FR-071KL"
        ],
    }
    source = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    (record,) = extract_component_procurement(
        source, source_name="component_inputs/yageo_rc0603_1k.json"
    )

    assert record.id == "yageo.rc0603fr-071kl"
    assert record.manufacturer == "Yageo"
    assert {(item.scheme, item.value, item.issuer) for item in record.identifiers} == {
        ("product_name", "RC0603FR-071KL", None),
        ("manufacturer_part_number", "RC0603FR-071KL", "Yageo"),
    }
    assert [item.kind for item in record.documents] == ["product_page"]
    assert record.offers == ()
    assert record.provides == ()
    assert {item.locator for item in record.evidence} == {
        "$.manufacturer",
        "$.product",
        "$.source_urls[0]",
    }
    assert {item.sha256 for item in record.evidence} == {
        "sha256:" + hashlib.sha256(source).hexdigest()
    }


def test_missing_or_malformed_identity_is_not_invented() -> None:
    source = b'{"manufacturer":"Acme","purpose":"unknown sample"}'
    assert extract_component_procurement(source, source_name="sample.json") == ()

    with pytest.raises(ProcurementExtractionError, match="require an explicit manufacturer"):
        extract_component_procurement(
            b'{"manufacturer_part_number":"ABC123"}',
            source_name="sample.json",
        )
    with pytest.raises(ProcurementExtractionError, match="both 'product' and 'products'"):
        extract_component_procurement(
            b'{"product":"one","products":["two"]}',
            source_name="sample.json",
        )

    (maker_like,) = extract_component_procurement(
        b'{"product":"Robot KIT123, item 1"}', source_name="sample.json"
    )
    assert maker_like.manufacturer is None
    assert [
        (item.scheme, item.value, item.issuer)
        for item in maker_like.identifiers
    ] == [("product_name", "Robot KIT123, item 1", None)]


def test_structured_ecad_identity_is_explicit_and_manufacturer_scoped() -> None:
    (record,) = extract_ecad_procurement(
        _ecad_extraction(), source_name="extractions/canonical-symbol.json"
    )
    assert record.manufacturer == "Acme"
    assert [(item.scheme, item.value, item.issuer) for item in record.identifiers] == [
        ("manufacturer_part_number", "AX-100", "Acme")
    ]
    assert record.provides == ()
    assert record.offers == ()
    assert {item.locator for item in record.evidence} == {
        "$.symbols[0].properties.Manufacturer",
        "$.symbols[0].properties.MPN",
    }

    # An MPN is not globally meaningful without an explicit issuing maker.
    assert (
        extract_ecad_procurement(
            _ecad_extraction(manufacturer=None),
            source_name="extractions/canonical-symbol.json",
        )
        == ()
    )


def test_pdf_labels_are_deterministic_and_unlabeled_text_defaults_to_missing() -> None:
    labeled = _pdf_extraction("Manufacturer: Acme\nMPN: AX-100\n")
    (record,) = extract_pdf_procurement(labeled, source_name="datasheet.pdf.json")
    assert record.manufacturer == "Acme"
    assert record.identifiers[0].value == "AX-100"
    assert record.provides == ()

    unlabeled = _pdf_extraction("Acme Corporation\nOrdering code AX-100\n")
    assert extract_pdf_procurement(unlabeled, source_name="datasheet.pdf.json") == ()


def test_pdf_luna_low_fallback_retries_validates_evidence_and_caches(tmp_path: Path) -> None:
    extraction = _pdf_extraction(
        "Manufacturer: NoiseCo\n"
        "MPN: NC-1\n"
        "description\n"
        "Manufacturer: Acme\n"
        "MPN: AX-100\n"
    )
    invalid = {
        "records": [
            {
                "manufacturer": "NoiseCo",
                "manufacturer_part_number": "AX-100",
                "manufacturer_page": 1,
                "manufacturer_line": "Manufacturer: NoiseCo",
                "part_number_page": 1,
                "part_number_line": "MPN: AX-100",
            }
        ]
    }
    valid = {
        "records": [
            {
                "manufacturer": "Acme",
                "manufacturer_part_number": "AX-100",
                "manufacturer_page": 1,
                "manufacturer_line": "Manufacturer: Acme",
                "part_number_page": 1,
                "part_number_line": "MPN: AX-100",
            }
        ]
    }
    client, completions = _mock_chat_client(invalid, valid)
    config = ProcurementTextFallbackConfig(
        cache_directory=tmp_path / "cache", max_attempts=3
    )
    (record,) = extract_pdf_procurement(
        extraction,
        source_name="datasheet.pdf.json",
        fallback=config,
        client=client,
    )
    assert record.manufacturer == "Acme"
    assert record.identifiers[0].value == "AX-100"
    assert len(completions.calls) == 2
    assert all(call["model"] == "gpt-5.6-luna" for call in completions.calls)
    assert all(call["reasoning_effort"] == "low" for call in completions.calls)
    assert all(call["messages"][1]["content"] == extraction.decode("utf-8") for call in completions.calls)
    assert all("%PDF" not in call["messages"][1]["content"] for call in completions.calls)
    assert len(tuple((tmp_path / "cache").glob("*.json"))) == 1

    class NeverCall:
        def create(self, **_kwargs):
            raise AssertionError("cache replay attempted a paid call")

    cached_client = SimpleNamespace(
        chat=SimpleNamespace(completions=NeverCall())
    )
    assert extract_pdf_procurement(
        extraction,
        source_name="datasheet.pdf.json",
        fallback=config,
        client=cached_client,
    ) == (record,)


@pytest.mark.parametrize(
    ("extraction", "response"),
    (
        (
            _pdf_extraction("Acme Corporation\nOrdering code AX-100\n"),
            {
                "records": [
                    {
                        "manufacturer": "Acme",
                        "manufacturer_part_number": "AX-100",
                        "manufacturer_page": 1,
                        "manufacturer_line": "Acme Corporation",
                        "part_number_page": 1,
                        "part_number_line": "Ordering code AX-100",
                    }
                ]
            },
        ),
        (
            _pdf_extraction_pages(
                "Manufacturer: Acme\nManufacturer: Other\n",
                "MPN: AX-100\nMPN: OT-200\n",
            ),
            {
                "records": [
                    {
                        "manufacturer": "Acme",
                        "manufacturer_part_number": "AX-100",
                        "manufacturer_page": 1,
                        "manufacturer_line": "Manufacturer: Acme",
                        "part_number_page": 2,
                        "part_number_line": "MPN: AX-100",
                    }
                ]
            },
        ),
        (
            _pdf_extraction(
                "Manufacturer: Acme\nMPN: AX-100\n"
                "Manufacturer: Beta\nMPN: B-200\n"
            ),
            {
                "records": [
                    {
                        "manufacturer": "Acme",
                        "manufacturer_part_number": "B-200",
                        "manufacturer_page": 1,
                        "manufacturer_line": "Manufacturer: Acme",
                        "part_number_page": 1,
                        "part_number_line": "MPN: B-200",
                    }
                ]
            },
        ),
    ),
    ids=("unlabeled", "cross-page", "cross-pair"),
)
def test_pdf_fallback_rejects_unrelated_literal_lines(
    tmp_path: Path, extraction: bytes, response: dict
) -> None:
    client, completions = _mock_chat_client(response)
    config = ProcurementTextFallbackConfig(
        cache_directory=tmp_path / "cache", max_attempts=1
    )
    with pytest.raises(
        ProcurementExtractionError, match="failed validation after 1 attempts"
    ):
        extract_pdf_procurement(
            extraction,
            source_name="datasheet.pdf.json",
            fallback=config,
            client=client,
        )
    assert len(completions.calls) == 1
    assert not tuple((tmp_path / "cache").glob("*.json"))


def test_pdf_fallback_accepts_multiple_adjacent_labeled_pairs(tmp_path: Path) -> None:
    extraction = _pdf_extraction(
        "Manufacturer: Acme\nMPN: AX-100\n"
        "description\n"
        "Manufacturer: Beta\nMPN: B-200\n"
    )
    response = {
        "records": [
            {
                "manufacturer": "Acme",
                "manufacturer_part_number": "AX-100",
                "manufacturer_page": 1,
                "manufacturer_line": "Manufacturer: Acme",
                "part_number_page": 1,
                "part_number_line": "MPN: AX-100",
            },
            {
                "manufacturer": "Beta",
                "manufacturer_part_number": "B-200",
                "manufacturer_page": 1,
                "manufacturer_line": "Manufacturer: Beta",
                "part_number_page": 1,
                "part_number_line": "MPN: B-200",
            },
        ]
    }
    client, completions = _mock_chat_client(response)
    records = extract_pdf_procurement(
        extraction,
        source_name="datasheet.pdf.json",
        fallback=ProcurementTextFallbackConfig(
            cache_directory=tmp_path / "cache", max_attempts=1
        ),
        client=client,
    )
    assert {(item.manufacturer, item.identifiers[0].value) for item in records} == {
        ("Acme", "AX-100"),
        ("Beta", "B-200"),
    }
    assert len(completions.calls) == 1


def test_item_number_and_product_list_paths_are_conservative() -> None:
    control = ROOT / "assembled_contraptions" / "scanner" / "component_inputs" / "romi_control.json"
    (record,) = extract_component_procurement_file(control)
    assert ("manufacturer_item_number", "3544", "Pololu") in {
        (item.scheme, item.value, item.issuer) for item in record.identifiers
    }

    power = ROOT / "assembled_contraptions" / "scanner" / "component_inputs" / "power.json"
    records = extract_component_procurement_file(power)
    assert len(records) == 2
    cells = next(item for item in records if "NiMH cells" in item.identifiers[0].value)
    regulator = next(item for item in records if "S13V25F6" in item.identifiers[0].value)
    assert cells.manufacturer is None
    assert cells.documents == ()
    assert regulator.manufacturer is None
    assert [item.url for item in regulator.documents] == [
        "https://www.pololu.com/product/4981"
    ]
    assert [(item.scheme, item.value, item.issuer) for item in regulator.identifiers] == [
        (
            "product_name",
            "Pololu 6V 2.5A step-up/step-down regulator S13V25F6, item 4981",
            None,
        )
    ]


def test_prior_twenty_inputs_yield_ten_exact_modeled_provisions() -> None:
    records: list[ProcurementRecord] = []
    exact_parts: dict[str, tuple[str, str]] = {}
    paths = sorted(PRIOR_INPUTS.glob("*.json"))
    assert len(paths) == 20
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        bindings = None
        if data["manufacturer"] == "Yageo":
            static = _exact_yageo_static(data)
            provision = static_part_provision(static)
            exact_parts[static.id] = (static.version, static.sha256)
            bindings = {data["product"]: provision}
        extracted = extract_component_procurement_file(
            path,
            source_name=f"component_inputs/{path.name}",
            provisions_by_product=bindings,
        )
        assert len(extracted) == 1
        records.extend(extracted)

    assert len(records) == 20
    assert sum(bool(record.provides) for record in records) == 10
    assert all(
        bool(record.provides) == (record.manufacturer == "Yageo")
        for record in records
    )
    ProcurementRegistry(records).validate_provisions(exact_parts)


def test_static_part_v2_rejects_inline_procurement() -> None:
    path = RESISTOR_INSTANTIATIONS / "generic-100ohm-resistor" / "static.part"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format"] = "static-part-2"
    data["purchasing"] = {"manufacturer": "Acme"}
    with pytest.raises(PhysicalSpecError, match="unknown static part field.*purchasing"):
        StaticPartSpec.from_dict(data)

    data.pop("purchasing")
    data.setdefault("metadata", {})["manufacturer"] = "Acme"
    with pytest.raises(PhysicalSpecError, match="belong in a .procurement record"):
        StaticPartSpec.from_dict(data)

    data["metadata"] = {"procurement": {"manufacturer": "Acme"}}
    with pytest.raises(
        PhysicalSpecError,
        match=r"static_part\.metadata\.procurement\.manufacturer",
    ):
        StaticPartSpec.from_dict(data)

    data["metadata"] = {"Procurement": {"Manufacturer": "Acme"}}
    with pytest.raises(
        PhysicalSpecError,
        match=r"static_part\.metadata\.Procurement\.Manufacturer",
    ):
        StaticPartSpec.from_dict(data)


def test_model_metadata_rejects_top_level_and_nested_procurement_identity() -> None:
    path = RESISTOR_INSTANTIATIONS / "generic-100ohm-resistor" / "v1.model"
    base = json.loads(path.read_text(encoding="utf-8"))
    for metadata, expected in (
        ({"manufacturer": "Acme"}, r"model_instance\.metadata\.manufacturer"),
        (
            {"procurement": {"manufacturer": "Acme"}},
            r"model_instance\.metadata\.procurement\.manufacturer",
        ),
        (
            {"Procurement": {"MPN": "AX-100"}},
            r"model_instance\.metadata\.Procurement\.MPN",
        ),
        (
            {"Identity": {"Vendor_SKU": "SKU-100"}},
            r"model_instance\.metadata\.Identity\.Vendor_SKU",
        ),
    ):
        data = dict(base)
        data["metadata"] = metadata
        with pytest.raises(PhysicalSpecError, match=expected):
            ModelInstantiationSpec.from_dict(data)


def test_registry_exposes_procurement_without_changing_resolved_part() -> None:
    directory = RESISTOR_INSTANTIATIONS / "generic-100ohm-resistor"
    static = _static_v2(directory / "static.part")
    model = ModelInstantiationSpec.from_json(
        (directory / "v1.model").read_text(encoding="utf-8")
    )
    instantiation = PartInstantiation(static, model, directory)
    source = b'{"manufacturer":"Acme","product":"ACME100R"}'
    provision = static_part_provision(static)
    (record,) = extract_component_procurement(
        source,
        source_name="component_inputs/acme.json",
        provisions_by_product={"ACME100R": provision},
    )

    empty = PartInstantiationRegistry((instantiation,))
    with_procurement = PartInstantiationRegistry(
        (instantiation,), procurement=ProcurementRegistry((record,))
    )

    assert with_procurement.procurement.for_part(static.id) == (record,)
    assert (
        empty.resolved_parts[instantiation.id].to_dict()
        == with_procurement.resolved_parts[instantiation.id].to_dict()
    )
    stale = replace(
        record,
        provides=(replace(provision, static_sha256="sha256:" + "0" * 64),),
    )
    with pytest.raises(PhysicalSpecError, match="provision.*stale"):
        PartInstantiationRegistry(
            (instantiation,), procurement=ProcurementRegistry((stale,))
        )


def test_central_record_io_and_explicit_batch_migration(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    evidenced = input_root / "evidenced.json"
    evidenced.write_text('{"product":"Explicit Product"}', encoding="utf-8")
    unidentified = input_root / "unidentified.json"
    unidentified.write_text('{"purpose":"fixture"}', encoding="utf-8")
    catalog = tmp_path / "catalog"

    registry = migrate_component_inputs(
        (evidenced, unidentified),
        catalog,
        source_root=input_root,
    )

    assert tuple(registry) == ("product.explicit-product",)
    record_path = (
        catalog
        / "procurement"
        / "records"
        / "product.explicit-product.procurement"
    )
    assert record_path.is_file()
    loaded = ProcurementRegistry.load_catalog(catalog)
    assert loaded["product.explicit-product"].to_dict() == registry[
        "product.explicit-product"
    ].to_dict()
    write_procurement_records(catalog, registry.values())

    changed = replace(registry["product.explicit-product"], version="1.0.1")
    with pytest.raises(ProcurementExtractionError, match="refusing to overwrite"):
        write_procurement_records(catalog, (changed,))
    with pytest.raises(ProcurementExtractionError, match="outside the migration set"):
        migrate_component_inputs(
            (evidenced,),
            tmp_path / "other-catalog",
            source_root=input_root,
            provisions_by_source={"missing.json": {}},
        )


def test_procurement_registry_rejects_unknown_and_stale_provisions() -> None:
    record = ProcurementRecord.from_dict(
        {
            "format": "procurement-record-1",
            "id": "acme.widget-1",
            "version": "1.0.0",
            "manufacturer": "Acme",
            "identifiers": [
                {
                    "scheme": "manufacturer_part_number",
                    "value": "WIDGET-1",
                    "issuer": "Acme",
                }
            ],
            "documents": [],
            "offers": [],
            "lifecycle": {"status": "unknown"},
            "provides": [
                {
                    "part": "widget",
                    "version": "2.0.0",
                    "static_sha256": "sha256:" + "1" * 64,
                    "quantity": 1,
                }
            ],
            "evidence": [
                {
                    "source": "component_inputs/widget.json",
                    "sha256": "sha256:" + "2" * 64,
                    "locator": "$.product",
                }
            ],
        }
    )
    registry = ProcurementRegistry((record,))
    with pytest.raises(ProcurementSpecError, match="unknown static part"):
        registry.validate_provisions({})
    with pytest.raises(ProcurementSpecError, match="is stale"):
        registry.validate_provisions(
            {"widget": ("2.0.0", "sha256:" + "3" * 64)}
        )


def test_checked_in_procurement_catalog_is_deterministic_and_exact() -> None:
    report = generate(ROOT, check=True)
    assert report["changed"] == []
    assert report["record_count"] == 29
    assert report["bound_record_count"] == 18
    assert report["unbound_record_count"] == 11
    assert report["provision_count"] == 21
    assert report["provided_quantity"] == 24

    registry = PartInstantiationRegistry.load_catalog(ROOT / "model_catalog")
    assert len(registry.procurement) == 29
    assert registry.procurement.sha256 == report["registry_sha256"]
    assert all(record.offers == () for record in registry.procurement.values())
    assert all(
        record.lifecycle.status == "unknown"
        for record in registry.procurement.values()
    )

    yageo = [
        record
        for record in registry.procurement.values()
        if record.manufacturer == "Yageo"
    ]
    murata = [
        record
        for record in registry.procurement.values()
        if record.manufacturer == "Murata"
    ]
    assert len(yageo) == 10
    assert len(murata) == 10
    assert all(len(record.provides) == 1 for record in yageo)
    assert all(record.provides == () for record in murata)

    expected_scanner = {
        "pololu.record-3500": {
            ("scanner.romi_chassis", 1),
            ("scanner.wheel", 2),
            ("scanner.gearmotor", 2),
        },
        "pololu.record-3550": {
            ("scanner.arm_linkage", 1),
            ("scanner.position_servo", 2),
        },
        "pololu.record-3544": {("scanner.control_board", 1)},
        "pololu.record-3542": {("scanner.encoder_pair", 1)},
        "product.pololu-6v-2-5a-step-up-step-down-regulator-s13v25f6-item-4981": {
            ("scanner.servo_regulator", 1)
        },
        "raspberry-pi.raspberry-pi-zero-2-w": {("scanner.compute", 1)},
        "raspberry-pi.camera-module-3-wide": {("scanner.camera", 1)},
    }
    for record_id, expected in expected_scanner.items():
        assert {
            (item.part, item.quantity)
            for item in registry.procurement[record_id].provides
        } == expected

    cells = registry.procurement[
        "product.six-rechargeable-aa-nimh-cells-and-matched-charger"
    ]
    assert cells.provides == ()
    assert all(
        "battery" not in provision.part
        for record in registry.procurement.values()
        for provision in record.provides
    )

    kemet = registry.procurement["kemet.c1210c476k8rac"]
    assert [(item.part, item.quantity) for item in kemet.provides] == [
        ("C1210C476K8RAC", 1)
    ]
    assert any(
        evidence.source.startswith("git:96e9baa46c8219982a1a798408bfcf3f638ec63e:")
        and evidence.sha256
        == "sha256:10503c7f8b52f8d9663203427547a9c3d1f49752e105b535995a8b79d909cf0a"
        for evidence in kemet.evidence
    )
