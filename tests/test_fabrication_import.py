from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from contraption.part_import.fabrication_extraction import (
    FabricationExtractionError,
    extract_connector_fabrication,
    materialize_proposal_fabrication,
    write_host_fabrication_context,
)


REPOSITORY = Path(__file__).resolve().parents[1]
GENERIC_PART = (
    REPOSITORY
    / "model_catalog/electrical/resistors/fixed_resistors/instantiations"
    / "generic-100ohm-resistor/static.part"
)


def _component(record: dict | None = None) -> bytes:
    value: dict[str, object] = {"manufacturer": "Evidence only"}
    if record is not None:
        value["connector_fabrication"] = [record]
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _partial_record(*, connector: str = "p") -> dict:
    return {
        "part": "generic-100ohm-resistor",
        "connector": connector,
        "fabrication": {
            "kind": "electrical_termination",
            "status": "partial",
            "missing": [
                "conductor.standard",
                "conductor.material",
                "conductor.cross_section_m2",
                "conductor.insulation_standard",
                "conductor.voltage_rating_v",
                "conductor.temperature_rating_k",
                "termination",
            ],
            "conductor": {"conductor_count": 1},
        },
    }


def _stage(component: bytes, *, mutate_static=None):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    inputs = root / "inputs"
    inputs.mkdir()
    source = inputs / "00_component.json"
    source.write_bytes(component)
    context = write_host_fabrication_context(
        root, component_input=source, source_name="component_inputs/component.json"
    )
    candidate = root / "candidate/electrical/resistors/instantiations/generic"
    candidate.mkdir(parents=True)
    static = json.loads(GENERIC_PART.read_text(encoding="utf-8"))
    if mutate_static is not None:
        mutate_static(static)
    (candidate / "static.part").write_text(
        json.dumps(static, indent=2) + "\n", encoding="utf-8"
    )
    return temporary, root / "candidate", context, candidate / "static.part"


def test_absent_source_facts_replace_agent_claims_with_missing():
    def invent(data):
        data["connectors"][0]["fabrication"] = copy.deepcopy(
            {
                **_partial_record()["fabrication"],
                "evidence": [
                    {"kind": "datasheet", "source": "invented.pdf"}
                ],
            }
        )

    temporary, candidate, context, static_path = _stage(_component(), mutate_static=invent)
    with temporary:
        materialize_proposal_fabrication(candidate, context)
        data = json.loads(static_path.read_text(encoding="utf-8"))
        assert [item["fabrication"]["status"] for item in data["connectors"]] == [
            "missing",
            "missing",
        ]
        assert data["connectors"][0]["fabrication"]["missing"] == [
            "conductor",
            "termination",
        ]


def test_explicit_partial_fact_is_bound_to_exact_component_evidence():
    source = _component(_partial_record())
    temporary, candidate, context, static_path = _stage(source)
    with temporary:
        materialize_proposal_fabrication(candidate, context)
        data = json.loads(static_path.read_text(encoding="utf-8"))
        first, second = (item["fabrication"] for item in data["connectors"])
        assert first["status"] == "partial"
        assert first["conductor"] == {"conductor_count": 1}
        assert first["evidence"] == [
            {
                "kind": "component_input",
                "source": "component_inputs/component.json",
                "locator": "$.connector_fabrication[0].fabrication",
                "sha256": "sha256:" + hashlib.sha256(source).hexdigest(),
            }
        ]
        assert second == {
            "kind": "electrical_termination",
            "status": "missing",
            "missing": ["conductor", "termination"],
        }


def test_source_cannot_author_its_own_evidence():
    record = _partial_record()
    record["fabrication"]["evidence"] = [
        {"kind": "datasheet", "source": "unverified.pdf"}
    ]
    with pytest.raises(FabricationExtractionError, match="cannot author evidence"):
        extract_connector_fabrication(
            _component(record), source_name="component_inputs/component.json"
        )


def test_unmatched_explicit_selector_fails_closed():
    temporary, candidate, context, _static_path = _stage(
        _component(_partial_record(connector="absent"))
    )
    with temporary, pytest.raises(FabricationExtractionError, match="absent from the candidate"):
        materialize_proposal_fabrication(candidate, context)


def test_context_is_rederived_from_protected_component_snapshot():
    source = _component(_partial_record())
    temporary, candidate, context, _static_path = _stage(source)
    with temporary:
        value = json.loads(context.read_text(encoding="utf-8"))
        value["records"] = []
        context.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(FabricationExtractionError, match="records changed"):
            materialize_proposal_fabrication(candidate, context)
