from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from contraption.part_import.agents import (
    AgentLimits,
    CLASSIFICATION_SCHEMA,
    MAX_MODELING_VALIDATION_ATTEMPTS,
    MODELING_SCHEMA,
    ClassificationAgent,
    ModelingAgent,
    ModelingInputs,
    _validate_modeling_value,
    _validate_strict_output_schema,
    _validate_shape,
    build_modeling_import_plan,
    modeling_preflight,
    run_classification_batch,
    run_modeling_proposal,
    validate_classification_proposal,
)
from contraption.part_import.fabrication_extraction import (
    HOST_FABRICATION_CONTEXT_FILENAME,
    proposal_fabrication_receipt_path,
    write_host_fabrication_context,
)
from contraption.part_import.budget import (
    BudgetExceeded,
    BudgetLedger,
    ProvenPreInferenceProviderRejection,
    TokenPricing,
    Usage,
)
from contraption.catalog.instantiations import StaticPartSpec
from contraption.catalog.interfaces import ModelInterfaceCatalog, interface_paths, load_interface_catalog
from contraption.catalog.procurement import ProcurementRecord
from contraption.cli import (
    _agent_key,
    _default_dotenv_path,
    _full_modeling_inputs,
    _load_agent_job_bundle,
    _torch_diagnostics,
    build_parser,
)
from contraption.part_import.model_validation_tool import (
    assert_workspace_integrity,
    validate_candidate,
    validation_activity,
    write_validation_context,
)
from contraption.part_import.reference_docs import (
    STRUCTURED_FORMAT_GUIDES,
    structured_format_guides,
)
from contraption.part_import.procurement_extraction import (
    HOST_PROCUREMENT_CONTEXT_FILENAME,
    proposal_procurement_receipt_path,
    verify_proposal_procurement_receipt,
    write_host_procurement_context,
)
from contraption.part_import.deterministic_assets import (
    build_proposal_shape_receipt,
    proposal_shape_receipt_path,
    verify_proposal_shape_receipt,
)
from contraption.shape import TessellatedShape, TriangleMesh, import_shape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = PROJECT_ROOT / "model_catalog"
INTERFACE_DATA = load_interface_catalog(CATALOG_ROOT).to_dict()


def _materialize_linked_shape(
    root: Path,
    *,
    linked_name: str = "scene.bin",
    write_receipt: bool = True,
) -> tuple[Path, Path]:
    run_root = root / "run"
    source_root = root / "source"
    source_root.mkdir()
    gltf = source_root / "scene.gltf"
    linked = source_root / linked_name
    gltf.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": linked_name}],
            }
        ),
        encoding="utf-8",
    )
    linked.write_bytes(b"verified-linked-resource")

    def tessellate(path: Path, _scale: float) -> TessellatedShape:
        mesh = TriangleMesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
        ).with_computed_normals()
        return TessellatedShape(mesh, (), (path.parent / linked_name,))

    artifacts = run_root / "workspace" / "proposed"
    import_shape(
        gltf,
        artifacts / "shape",
        artifact_id="fixture-linked-gltf",
        metres_per_source_unit=1.0,
        tessellator=tessellate,
    )
    plan = run_root / "deterministic-assets" / "plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        '{"format":"deterministic-part-ingestion-staged-1"}\n',
        encoding="utf-8",
    )
    host_paths = tuple(path for path in artifacts.rglob("*") if path.is_file())
    receipt = build_proposal_shape_receipt(artifacts, plan, host_paths)
    if receipt is None:
        raise AssertionError("shape materialization did not produce a receipt")
    if write_receipt:
        proposal_shape_receipt_path(artifacts).write_text(
            json.dumps(receipt, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifacts, linked


class _UsageDetails:
    cached_tokens = 10


class _Usage:
    input_tokens = 100
    output_tokens = 20
    input_tokens_details = _UsageDetails()


def _classification_value(**overrides) -> dict:
    value = {
        "canonical_name": "Fixed resistor",
        "domains": ["electrical"],
        "reuse_path": ["resistor", "fixed-resistor"],
        "new_nodes": [],
        "category": "resistor",
        "device": "fixed-resistor",
        "rationale": "existing contract fits",
        "uncertainties": [],
    }
    value.update(overrides)
    return value


class _Response:
    usage = _Usage()

    def __init__(self, value: dict | None = None):
        self.output_text = json.dumps(value or _classification_value())


class _Responses:
    def __init__(self, values: list[dict] | None = None):
        self.values = list(values or [])
        self.calls = 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls += 1
        return _Response(self.values.pop(0) if self.values else None)


class _Client:
    def __init__(self, values: list[dict] | None = None):
        self.responses = _Responses(values)


def _modeling_value(
    *, path: str = "components/target.json", content: str = '{"id":"target"}'
) -> dict:
    return {
        "summary": "modeled the authoritative target",
        "artifacts": [{"path": path, "content": content}],
        "assumptions": [],
        "evidence": ["fixture evidence"],
    }


def _canary_modeling_value() -> dict:
    device = Path("electrical/resistors/fixed_resistors")
    source_instance = device / "instantiations/generic-100ohm-resistor"
    part_id = "fixture-100ohm-resistor"
    instance = device / f"instantiations/{part_id}"
    static_part = json.loads(
        (CATALOG_ROOT / source_instance / "static.part").read_text(encoding="utf-8")
    )
    static_part["id"] = part_id
    static_part["name"] = "Fixture 100 ohm resistor"
    geometry = static_part["bodies"][0]["solids"][0]["geometry"]
    static_part["bodies"][0]["solids"][0]["geometry"] = {
        "kind": "box",
        "dimensions_m": geometry["dimensions_m"],
    }
    model_instance = json.loads(
        (CATALOG_ROOT / source_instance / "v1.model").read_text(encoding="utf-8")
    )
    model_instance["id"] = f"{part_id}.v1"
    model_instance["part"] = part_id
    return {
        "summary": "modeled the authoritative target",
        "artifacts": [
            {
                "path": (instance / "static.part").as_posix(),
                "content": json.dumps(static_part, indent=2, sort_keys=True) + "\n",
            },
            {
                "path": (instance / "v1.model").as_posix(),
                "content": json.dumps(model_instance, indent=2, sort_keys=True) + "\n",
            },
        ],
        "assumptions": [],
        "evidence": ["fixture evidence"],
    }


def _event_stream(*messages: dict, include_usage: bool = False) -> str:
    events = [
        {"type": "thread.started", "thread_id": "fixture-thread"},
        {"type": "turn.started"},
    ]
    for index, message in enumerate(messages):
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": f"item_{index}",
                    "type": "agent_message",
                    "text": json.dumps(message, sort_keys=True),
                },
            }
        )
    if include_usage:
        events.append(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 25,
                    "output_tokens": 50,
                },
            }
        )
    events.extend(
        (
            {"type": "error", "message": "shared rollout token budget exhausted"},
            {
                "type": "turn.failed",
                "error": {"message": "shared rollout token budget exhausted"},
            },
        )
    )
    return "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n"


def _invalid_schema_provider_message() -> str:
    return json.dumps(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_json_schema",
                "message": "fixture schema rejected before inference",
                "param": "text.format.schema",
            },
            "status": 400,
        },
        sort_keys=True,
    )


def _provider_rejection_stream(
    *,
    include_usage: bool = False,
    include_agent_message: bool = False,
    include_command_execution: bool = False,
    include_in_turn_error_item: bool = False,
    include_known_pre_turn_warning: bool = False,
    include_prefixed_pre_turn_error: bool = False,
    include_later_event: bool = False,
) -> str:
    message = _invalid_schema_provider_message()
    events: list[dict] = [
        {"type": "thread.started", "thread_id": "fixture-thread"},
    ]
    if include_known_pre_turn_warning:
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_warning",
                    "type": "error",
                    "message": (
                        "Under-development features enabled: rollout_budget. "
                        "Under-development features are incomplete and may behave "
                        "unpredictably. To suppress this warning, set "
                        "`suppress_unstable_features_warning = true` in "
                        "/tmp/contraption-codex-auth-fixture1/config.toml."
                    ),
                },
            }
        )
    if include_prefixed_pre_turn_error:
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_forged_warning",
                    "type": "error",
                    "message": (
                        "Under-development features enabled: rollout_budget. "
                        "different unknown provider activity"
                    ),
                },
            }
        )
    events.append({"type": "turn.started"})
    if include_agent_message:
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_inference",
                    "type": "agent_message",
                    "text": '{"summary":"inference happened"}',
                },
            }
        )
    if include_command_execution:
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_command",
                    "type": "command_execution",
                    "command": "true",
                    "exit_code": 0,
                },
            }
        )
    if include_in_turn_error_item:
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_error",
                    "type": "error",
                    "message": "unknown in-turn provider activity",
                },
            }
        )
    if include_usage:
        events.append(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 25,
                    "output_tokens": 50,
                },
            }
        )
    events.extend(
        (
            {"type": "error", "message": message},
            {"type": "turn.failed", "error": {"message": message}},
        )
    )
    if include_later_event:
        events.append({"type": "thread.idle"})
    return "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n"


def _prepared_workspace(staging_root: Path, run_name: str = "modeling-existing") -> Path:
    workspace = staging_root / run_name / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "candidate").mkdir()
    (workspace / "INPUT_MANIFEST.json").write_text("[]\n", encoding="utf-8")
    (workspace / "output-schema.json").write_text("{}\n", encoding="utf-8")
    write_validation_context(
        workspace.parent,
        (workspace / "INPUT_MANIFEST.json", workspace / "output-schema.json"),
    )
    return workspace


def _real_modeling_inputs() -> ModelingInputs:
    return ModelingInputs(
        constraints=PROJECT_ROOT / "prompts" / "model_constraints.md",
        gold_templates=(
            CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl",
            CATALOG_ROOT / "mechanical" / "inert_objects" / "planar_rigid_bodies" / "rigid_body_planar.pmdl",
        ),
        interfaces=interface_paths(CATALOG_ROOT),
        direct_hierarchy=(CATALOG_ROOT / "electromechanical" / "motors" / "brushed_dc_motors" / "dc_motor.pmdl",),
        component_information=(
            PROJECT_ROOT
            / "assembled_contraptions"
            / "scanner"
            / "component_inputs"
            / "romi_drive.json"
        ),
    )


class BudgetTests(unittest.TestCase):
    def test_provider_output_schemas_are_recursively_strict(self):
        def assert_strict(node: dict, path: str) -> None:
            raw_type = node.get("type")
            types = set(raw_type) if isinstance(raw_type, list) else {raw_type}
            if "object" in types:
                properties = node.get("properties")
                self.assertIsInstance(properties, dict, path)
                self.assertIs(node.get("additionalProperties"), False, path)
                required = node.get("required")
                self.assertIsInstance(required, list, path)
                self.assertEqual(len(required), len(set(required)), path)
                self.assertEqual(set(required), set(properties), path)
                for name, child in properties.items():
                    assert_strict(child, f"{path}.properties[{name!r}]")
            if "array" in types:
                self.assertIn("items", node, path)
                assert_strict(node["items"], f"{path}.items")
            for keyword in ("anyOf", "allOf", "oneOf"):
                for index, child in enumerate(node.get(keyword, [])):
                    assert_strict(child, f"{path}.{keyword}[{index}]")
            for keyword in ("$defs", "definitions"):
                for name, child in node.get(keyword, {}).items():
                    assert_strict(child, f"{path}.{keyword}[{name!r}]")

        for name, schema in (
            ("classification", CLASSIFICATION_SCHEMA),
            ("modeling", MODELING_SCHEMA),
        ):
            with self.subTest(schema=name):
                _validate_strict_output_schema(schema, source=name)
                assert_strict(schema, "$")

        artifact = MODELING_SCHEMA["properties"]["artifacts"]["items"]
        self.assertEqual(artifact["required"], ["path", "content"])
        self.assertEqual(artifact["properties"]["content"]["type"], ["string", "null"])

    def test_invalid_nested_modeling_schema_never_reserves_or_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = BudgetLedger(root / "ledger.json", limit_usd=1.0)
            agent = ModelingAgent(
                ledger,
                root / "staging",
                codex_binary="must-not-run",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
            )
            malformed = json.loads(json.dumps(MODELING_SCHEMA))
            malformed["properties"]["artifacts"]["items"]["required"] = ["path"]

            with mock.patch(
                "contraption.part_import.agents.MODELING_SCHEMA", malformed
            ), mock.patch(
                "contraption.part_import.agents.subprocess.run"
            ) as dispatch:
                with self.assertRaisesRegex(
                    ValueError, r"required must exactly match properties.*content"
                ):
                    agent.run(_real_modeling_inputs())

            dispatch.assert_not_called()
            self.assertEqual(ledger.snapshot()["events"], [])
            self.assertFalse((root / "staging").exists())

    def test_invalid_nested_modeling_schema_type_never_reserves_or_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = BudgetLedger(root / "ledger.json", limit_usd=1.0)
            agent = ModelingAgent(
                ledger,
                root / "staging",
                codex_binary="must-not-run",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
            )
            malformed = json.loads(json.dumps(MODELING_SCHEMA))
            malformed["properties"]["artifacts"]["items"]["properties"][
                "content"
            ]["type"] = "bogus-provider-type"

            with mock.patch(
                "contraption.part_import.agents.MODELING_SCHEMA", malformed
            ), mock.patch(
                "contraption.part_import.agents.subprocess.run"
            ) as dispatch:
                with self.assertRaisesRegex(ValueError, "unsupported names"):
                    agent.run(_real_modeling_inputs())

            dispatch.assert_not_called()
            self.assertEqual(ledger.snapshot()["events"], [])
            self.assertFalse((root / "staging").exists())

    def test_zero_settlement_rejects_subclass_spoof(self):
        class SpoofedProof(ProvenPreInferenceProviderRejection):
            def __post_init__(self) -> None:
                pass

        proof = SpoofedProof(
            source="codex_jsonl",
            terminal_event="turn.failed",
            provider_error_type="invalid_request_error",
            provider_error_code="invalid_json_schema",
            provider_status=400,
            provider_param="text.format.schema",
            usage_observed=False,
            completed_agent_message_observed=True,
            candidate_artifact_observed=False,
            validator_activity_observed=False,
            malformed_event_observed=False,
            unknown_failure_event_observed=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json", limit_usd=1.0)
            ledger.reserve("spoof", 0.5, {"kind": "test"})
            with self.assertRaisesRegex(TypeError, "required"):
                ledger.settle_proven_pre_inference_provider_rejection(
                    "spoof", proof=proof
                )
            snapshot = ledger.snapshot()
            self.assertIn("spoof", snapshot["reserved"])
            self.assertEqual(snapshot["events"], [])

    def test_pre_strict_path_only_receipt_remains_host_compatible(self):
        legacy = _modeling_value()
        del legacy["artifacts"][0]["content"]

        _validate_modeling_value(legacy)
        with self.assertRaisesRegex(ValueError, r"missing keys .*content"):
            _validate_shape(legacy, MODELING_SCHEMA)

    def test_reservation_and_settlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json", limit_usd=1.0)
            ledger.reserve("a", 0.5, {"kind": "test"})
            with self.assertRaises(BudgetExceeded):
                ledger.reserve("b", 0.6, {})
            charged = ledger.settle(
                "a", usage=Usage(1000, 0, 100), pricing=TokenPricing()
            )
            self.assertGreater(charged, 0)
            self.assertAlmostEqual(ledger.snapshot()["spent_usd"], charged)

    def test_classifier_is_structured_and_budgeted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json", limit_usd=100.0)
            agent = ClassificationAgent(ledger)
            result, usage, charged = agent.classify(
                {"name": "10 ohm resistor"},
                INTERFACE_DATA,
                client=_Client(),
            )
            self.assertEqual(result["category"], "resistor")
            self.assertEqual(usage.input_tokens, 100)
            self.assertGreater(charged, 0)
            _validate_shape(result, CLASSIFICATION_SCHEMA)

    def test_classification_canary_keeps_structured_output_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json", limit_usd=100.0)
            client = _Client()
            agent = ClassificationAgent(
                ledger,
                limits=AgentLimits(max_input_tokens=10_000, max_output_tokens=4_000),
            )
            agent.classify(
                {"name": "10 ohm resistor"},
                INTERFACE_DATA,
                client=client,
                canary=True,
            )
            self.assertEqual(client.responses.kwargs["max_output_tokens"], 4_000)

    def test_classifier_prompt_states_machine_identifier_grammar(self):
        prompt = ClassificationAgent.system_prompt(
            INTERFACE_DATA
        )
        self.assertIn("^[A-Za-z][A-Za-z0-9_.-]*$", prompt)
        self.assertIn("newly invented identifiers in lowercase kebab-case", prompt)
        self.assertIn("canonical_name is the separate human-readable display name", prompt)
        self.assertIn("must never be empty", prompt)
        self.assertIn("domain ids never belong in category, device, or reuse_path", prompt)
        self.assertIn("Allowed existing root category ids are exactly", prompt)
        self.assertIn("Domain ids are exactly", prompt)
        self.assertIn('"voltage-source"', prompt)
        self.assertIn("domains must contain every domain declared", prompt)
        self.assertIn("physical ports and behavior implemented by this component itself", prompt)
        self.assertIn("also include all of that domain's requires_physics entries", prompt)
        self.assertIn("never omit required physics domains", prompt)

    def test_generated_paths_cannot_escape_staging(self):
        for raw in (
            "../evil.py",
            "..\\evil.pmdl",
            "C:\\absolute\\model.pmdl",
            "/absolute/model.pmdl",
            "NUL.json",
            "host.py",
        ):
            with self.assertRaises(ValueError):
                ModelingAgent._safe_relative(raw)
        self.assertEqual(
            ModelingAgent._safe_relative("components/motor.pmdl"),
            Path("components") / "motor.pmdl",
        )

    def test_modeling_prompt_makes_catalog_relative_paths_unambiguous(self):
        prompt = ModelingAgent.prompt("target.json")
        self.assertIn("candidate/<physical-domain>/<category>[/<device>]", prompt)
        self.assertIn("never candidate/model_catalog/", prompt)
        self.assertIn("omit both candidate/ and model_catalog/", prompt)
        self.assertIn("static.part and .model files as exact record-shape examples", prompt)
        self.assertIn("Geometry and optical source ingestion are host-owned", prompt)
        self.assertIn("Never parse or emit CAD", prompt)
        self.assertIn("preserve only explicit opaque host references", prompt)
        self.assertIn("Do not create README.md", prompt)
        self.assertIn("Connections are optional", prompt)
        self.assertIn("at most 3 times total", prompt)
        self.assertIn("Do not copy file contents into the final response", prompt)

    def test_all_structured_format_guides_exist_in_declared_order(self):
        paths = structured_format_guides(PROJECT_ROOT)
        self.assertEqual(
            tuple(path.relative_to(PROJECT_ROOT).as_posix() for path in paths),
            STRUCTURED_FORMAT_GUIDES,
        )
        texts = [path.read_text(encoding="utf-8") for path in paths]
        self.assertTrue(all(text.startswith("# ") for text in texts))
        self.assertTrue(all(len(text.splitlines()) >= 20 for text in texts))
        by_name = {path.name: text for path, text in zip(paths, texts, strict=True)}
        self.assertIn("`artifact_type`", by_name["PMDL.md"])
        self.assertIn("`controller_stream`", by_name["PMDL.md"])
        self.assertIn("`deterministic-part-ingestion-staged-1`", by_name["DETERMINISTIC_INGESTION.md"])
        self.assertIn("`optical_sensors`", by_name["DETERMINISTIC_INGESTION.md"])
        self.assertIn("Luna must understand this contract", by_name["DETERMINISTIC_INGESTION.md"])
        self.assertIn("`shape_uri`", by_name["STATIC_PART.md"])
        self.assertIn("`optical_sensors`", by_name["STATIC_PART.md"])
        self.assertNotIn('"mesh_uri"', by_name["STATIC_PART.md"])
        self.assertIn("DifferentiableConstraint", by_name["OPTICAL_OBSERVATION.md"])
        self.assertIn("<4sBBHQQII", by_name["OPTICAL_SENSOR.md"])
        self.assertIn("`observation_manifest`", by_name["OPTICAL_SENSOR.md"])
        self.assertIn("`occupied_probability`", by_name["RECONSTRUCTION_STATE.md"])
        workflow = by_name["OPTICAL_WORKFLOWS.md"]
        self.assertIn("contraption simulate", workflow)
        self.assertIn("--optical-capture", workflow)
        self.assertIn("contraption optical-reconstruct", workflow)
        self.assertIn("assembly-optical-capture-1", workflow)
        self.assertIn("optical-reconstruction-run-1", workflow)
        self.assertIn("shape.artifact.json", workflow)
        self.assertIn("`contraption.render-bundle/v1`", by_name["RENDER_BUNDLE.md"])

    def test_luna_response_cannot_author_deterministic_shape_or_optical_payloads(self):
        formats = (
            "deterministic-part-ingestion-1",
            "deterministic-part-ingestion-staged-1",
            "shape-artifact-1",
            "optical-material-1",
            "optical-sensor-1",
            "optical-scene-1",
            "optical-observation-1",
            "reconstruction-state-1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for marker in formats:
                value = _modeling_value(
                    path=f"{marker}.json",
                    content=json.dumps({"format": marker}),
                )
                with self.assertRaisesRegex(ValueError, "host-owned artifact"):
                    ModelingAgent._materialize_artifacts(root / marker, value)
            for schema in ("contraption.triangle-mesh/v1", "contraption.ctmesh/v1"):
                value = _modeling_value(
                    path="canonical-mesh.json",
                    content=json.dumps({"schema": schema}),
                )
                with self.assertRaisesRegex(ValueError, "host-owned artifact"):
                    ModelingAgent._materialize_artifacts(root / schema.replace("/", "_"), value)

    def test_luna_candidate_recovery_enforces_deterministic_payload_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            candidate = workspace / "candidate"
            candidate.mkdir()
            (candidate / "shape.json").write_text(
                json.dumps({"format": "shape-artifact-1"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "host-owned artifact"):
                ModelingAgent._value_from_candidate_files(workspace)

    def test_doctor_reports_explicit_torch_cuda_details_and_errors(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            current_device=lambda: 1,
            get_device_name=lambda index: f"Fixture GPU {index}",
        )
        torch = types.SimpleNamespace(
            __version__="9.8.7", version=types.SimpleNamespace(cuda="13.1"), cuda=cuda
        )
        with mock.patch(
            "contraption.cli.importlib.util.find_spec", return_value=object()
        ), mock.patch.dict(sys.modules, {"torch": torch}):
            report = _torch_diagnostics()
        self.assertEqual(report["version"], "9.8.7")
        self.assertEqual(report["compiled_cuda_runtime"], "13.1")
        self.assertTrue(report["cuda_available"])
        self.assertEqual(report["cuda_device_count"], 2)
        self.assertEqual(report["cuda_selected_device_name"], "Fixture GPU 1")

        broken = types.SimpleNamespace(
            __version__="9.8.7",
            version=types.SimpleNamespace(cuda="13.1"),
            cuda=types.SimpleNamespace(
                is_available=mock.Mock(side_effect=RuntimeError("driver mismatch"))
            ),
        )
        with mock.patch(
            "contraption.cli.importlib.util.find_spec", return_value=object()
        ), mock.patch.dict(sys.modules, {"torch": broken}):
            failed = _torch_diagnostics()
        self.assertIn("driver mismatch", failed["cuda_runtime_error"])

    def test_actual_agent_cli_requires_a_declarative_job_file_and_target(self):
        job_file = (
            PROJECT_ROOT / "assembled_contraptions" / "scanner" / "agent_jobs.json"
        )
        args = build_parser().parse_args(
            [
                "agent-run",
                "modeling-one",
                "--job-file",
                str(job_file),
                "--target",
                "romi_drive",
            ]
        )
        self.assertEqual(args.agent_job, "modeling-one")
        self.assertEqual(args.target, "romi_drive")
        self.assertFalse(args.force)
        bundle = _load_agent_job_bundle(args.job_file)
        inputs = _full_modeling_inputs(bundle, args.target)
        self.assertEqual(
            inputs.component_information,
            PROJECT_ROOT
            / "assembled_contraptions"
            / "scanner"
            / "component_inputs"
            / "romi_drive.json",
        )
        self.assertEqual(inputs.interfaces, interface_paths(CATALOG_ROOT))

    def test_dotenv_resolution_supports_parent_workspace_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            repo = workspace / "the-bayesian-world"
            repo.mkdir()
            parent_env = workspace / ".env"
            parent_env.write_text("OPENAI_API_KEY=test-parent-key\n", encoding="utf-8")
            with mock.patch("contraption.cli.WORK_ROOT", repo), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(_default_dotenv_path(), parent_env)
                key, path = _agent_key(None)
            self.assertEqual(path, parent_env)
            self.assertEqual(key, "test-parent-key")

    def test_dotenv_resolution_rejects_ambiguous_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            repo = workspace / "the-bayesian-world"
            repo.mkdir()
            (workspace / ".env").write_text("OPENAI_API_KEY=parent\n", encoding="utf-8")
            (repo / ".env").write_text("OPENAI_API_KEY=repo\n", encoding="utf-8")
            with mock.patch("contraption.cli.WORK_ROOT", repo):
                with self.assertRaisesRegex(RuntimeError, "multiple dotenv files"):
                    _default_dotenv_path()

    def test_explicit_dotenv_resolves_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            repo = workspace / "the-bayesian-world"
            repo.mkdir()
            parent_env = workspace / ".env"
            repo_env = repo / ".env"
            parent_env.write_text("OPENAI_API_KEY=parent\n", encoding="utf-8")
            repo_env.write_text("OPENAI_API_KEY=repo\n", encoding="utf-8")
            with mock.patch("contraption.cli.WORK_ROOT", repo), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                key, path = _agent_key(str(parent_env))
            self.assertEqual(path, parent_env)
            self.assertEqual(key, "parent")


class ClassificationSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interface_catalog = INTERFACE_DATA

    def test_existing_and_genuinely_new_device_paths_are_valid(self):
        self.assertEqual(
            validate_classification_proposal(_classification_value(), self.interface_catalog),
            _classification_value(),
        )
        proposed = _classification_value(
            canonical_name="Thin-film fixed resistor",
            reuse_path=["resistor"],
            device="thin-film-resistor",
            new_nodes=[
                {
                    "parent": "resistor",
                    "label": "thin-film-resistor",
                    "contract_change": False,
                    "model_specificity_reason": "Adds thin-film temperature and noise parameters.",
                }
            ],
        )
        self.assertEqual(
            validate_classification_proposal(proposed, self.interface_catalog), proposed
        )

    def test_interface_catalog_rejects_semantically_parallel_categories(self):
        duplicate = json.loads(json.dumps(self.interface_catalog))
        clone = dict(duplicate["categories"][0])
        clone.update({"id": "resistor-canary-import", "name": "Resistor"})
        duplicate["categories"].append(clone)
        with self.assertRaisesRegex(ValueError, "parallel semantic identities"):
            ModelInterfaceCatalog.from_dict(duplicate)

    def test_semantic_mismatches_fail_deterministically(self):
        cases = {
            "unknown domain": _classification_value(domains=["rigid_mechanical"]),
            "missing intersection physics": _classification_value(
                domains=["electrical", "electromechanical"]
            ),
            "noncontiguous ancestry": _classification_value(reuse_path=["fixed-resistor"]),
            "empty canonical": _classification_value(canonical_name="  "),
            "undeclared new terminal": _classification_value(
                device="thin-film-resistor"
            ),
            "colliding new node": _classification_value(
                reuse_path=["resistor"],
                new_nodes=[
                    {
                        "parent": "resistor",
                        "label": "fixed_resistor",
                        "contract_change": False,
                        "model_specificity_reason": "Attempts to duplicate an existing node.",
                    }
                ],
                device="fixed_resistor",
            ),
            "invalid proposed parent": _classification_value(
                reuse_path=["resistor"],
                new_nodes=[
                    {
                        "parent": "motor",
                        "label": "thin-film-resistor",
                        "contract_change": False,
                        "model_specificity_reason": "Adds thin-film temperature and noise parameters.",
                    }
                ],
                device="thin-film-resistor",
            ),
            "multiple proposed devices": _classification_value(
                reuse_path=["resistor"],
                new_nodes=[
                    {
                        "parent": "resistor",
                        "label": "thin-film-resistor",
                        "contract_change": False,
                        "model_specificity_reason": "Adds thin-film temperature and noise parameters.",
                    },
                    {
                        "parent": "resistor",
                        "label": "carbon-film-resistor",
                        "contract_change": False,
                        "model_specificity_reason": "Adds carbon-film temperature and noise parameters.",
                    },
                ],
                device="thin-film-resistor",
            ),
        }
        for label, proposal in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_classification_proposal(proposal, self.interface_catalog)

    def test_invalid_provider_output_is_charged_and_not_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp) / "ledger.json", 100.0)
            agent = ClassificationAgent(ledger)
            with self.assertRaisesRegex(ValueError, "unknown physical domains"):
                agent.classify(
                    {"name": "bad"},
                    self.interface_catalog,
                    client=_Client([_classification_value(domains=["unknown"])]),
                )
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["reserved"], {})
            self.assertEqual(snapshot["events"][-1]["status"], "invalid_output")
            self.assertGreater(snapshot["events"][-1]["charged_usd"], 0)


class AgentWorkflowTests(unittest.TestCase):
    def _failed_modeling_event(
        self,
        stdout: str,
        *,
        stderr: str = "",
        candidate_artifact: bool = False,
        validator_activity_observed: bool = False,
    ) -> tuple[dict, float, TokenPricing]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            workspace = _prepared_workspace(staging, "modeling-accounting")
            if candidate_artifact:
                (workspace / "candidate" / "partial.txt").write_text(
                    "partial candidate output\n", encoding="utf-8"
                )
            if validator_activity_observed:
                (workspace / "validation-calls.jsonl").write_text(
                    json.dumps(
                        {
                            "schema": "contraption.model-validation/v1",
                            "call_number": 1,
                            "valid": False,
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            process = subprocess.CompletedProcess(
                args=["codex"], returncode=1, stdout=stdout, stderr=stderr
            )
            pricing = TokenPricing()
            ledger = BudgetLedger(root / "ledger.json", limit_usd=1.0)
            agent = ModelingAgent(
                ledger,
                staging,
                codex_binary="codex-fixture",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
                pricing=pricing,
            )
            component = root / "component.json"
            component.write_text('{"domains":["electrical"]}\n', encoding="utf-8")
            inputs = ModelingInputs(
                constraints=Path("constraints.md"),
                gold_templates=(Path("gold.pmdl"),),
                interfaces=(Path("interface.pmdl"),),
                direct_hierarchy=(),
                component_information=component,
            )
            reservation = pricing.worst_case(
                max_input_tokens=2_000, max_output_tokens=1_000
            )
            with mock.patch.object(
                agent, "prepare_workspace", return_value=workspace
            ), mock.patch(
                "contraption.part_import.agents.subprocess.run", return_value=process
            ):
                with self.assertRaises(RuntimeError):
                    agent.run(inputs)
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["reserved"], {})
            self.assertEqual(len(snapshot["events"]), 1)
            return dict(snapshot["events"][0]), reservation, pricing

    def test_exact_invalid_schema_rejection_releases_reservation_at_zero(self):
        event, _reservation, _pricing = self._failed_modeling_event(
            _provider_rejection_stream()
        )

        self.assertEqual(event["status"], "provider_rejected_before_inference")
        self.assertEqual(event["charged_usd"], 0.0)
        self.assertIsNone(event["usage"])
        self.assertEqual(event["cost_basis"], "proven_pre_inference_zero")
        self.assertEqual(event["proof"]["provider_error_code"], "invalid_json_schema")
        self.assertEqual(event["proof"]["provider_param"], "text.format.schema")

        warned, _reservation, _pricing = self._failed_modeling_event(
            _provider_rejection_stream(include_known_pre_turn_warning=True)
        )
        self.assertEqual(warned["status"], "provider_rejected_before_inference")
        self.assertEqual(warned["charged_usd"], 0.0)

    def test_stderr_invalid_schema_substring_cannot_authorize_zero(self):
        unknown = _event_stream()
        event, reservation, _pricing = self._failed_modeling_event(
            unknown,
            stderr=f"diagnostic only: {_invalid_schema_provider_message()}",
        )

        self.assertEqual(event["status"], "failed_after_dispatch")
        self.assertAlmostEqual(event["charged_usd"], reservation)
        self.assertEqual(event["cost_basis"], "full_reservation_conservative")

    def test_unknown_or_malformed_no_usage_failure_keeps_full_debit(self):
        exact_message = _invalid_schema_provider_message()
        cases = {
            "unknown": _event_stream(),
            "malformed": (
                json.dumps({"type": "error", "message": exact_message})
                + "\n"
                + '{"type":"turn.failed","error":'
                + "\n"
            ),
        }
        for label, stdout in cases.items():
            with self.subTest(label=label):
                event, reservation, _pricing = self._failed_modeling_event(stdout)
                self.assertEqual(event["status"], "failed_after_dispatch")
                self.assertAlmostEqual(event["charged_usd"], reservation)
                self.assertEqual(
                    event["cost_basis"], "full_reservation_conservative"
                )
                self.assertIsNone(event["usage"])

    def test_usage_bearing_provider_failure_charges_reported_usage(self):
        event, reservation, pricing = self._failed_modeling_event(
            _provider_rejection_stream(include_usage=True)
        )
        expected = pricing.cost(Usage(100, 25, 50))

        self.assertEqual(event["status"], "failed_after_dispatch")
        self.assertAlmostEqual(event["charged_usd"], expected)
        self.assertLess(event["charged_usd"], reservation)
        self.assertEqual(event["cost_basis"], "reported_usage")
        self.assertEqual(event["usage"]["input_tokens"], 100)

    def test_post_inference_activity_prevents_zero_settlement(self):
        cases = {
            "completed agent message": {
                "stdout": _provider_rejection_stream(include_agent_message=True),
            },
            "candidate artifact": {
                "stdout": _provider_rejection_stream(),
                "candidate_artifact": True,
            },
            "validator activity": {
                "stdout": _provider_rejection_stream(),
                "validator_activity_observed": True,
            },
            "completed command": {
                "stdout": _provider_rejection_stream(
                    include_command_execution=True
                ),
            },
            "event after terminal failure": {
                "stdout": _provider_rejection_stream(include_later_event=True),
            },
            "in-turn completed error item": {
                "stdout": _provider_rejection_stream(
                    include_in_turn_error_item=True
                ),
            },
            "prefixed but unknown pre-turn error": {
                "stdout": _provider_rejection_stream(
                    include_prefixed_pre_turn_error=True
                ),
            },
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label):
                event, reservation, _pricing = self._failed_modeling_event(**kwargs)
                self.assertEqual(event["status"], "failed_after_dispatch")
                self.assertAlmostEqual(event["charged_usd"], reservation)
                self.assertEqual(
                    event["cost_basis"], "full_reservation_conservative"
                )

    def test_classification_batch_persists_and_resumes_exact_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            components = []
            for name in ("a", "b"):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
                components.append(path)
            ledger = BudgetLedger(root / "ledger.json", 100.0)
            secret = "fixture-secret-never-persist"
            agent = ClassificationAgent(ledger, api_key=secret)
            client = _Client()
            output = root / "proposals"

            first = run_classification_batch(
                agent, components, CATALOG_ROOT, output, client=client
            )
            self.assertEqual(client.responses.calls, 2)
            self.assertTrue(all(item["status"] == "completed" for item in first))
            second = run_classification_batch(
                agent, components, CATALOG_ROOT, output, client=client
            )
            self.assertEqual(client.responses.calls, 2)
            self.assertTrue(
                all(item["status"] == "skipped_exact_input" for item in second)
            )

            components[1].write_text(
                json.dumps({"name": "b", "revision": 2}) + "\n", encoding="utf-8"
            )
            third = run_classification_batch(
                agent, components, CATALOG_ROOT, output, client=client
            )
            self.assertEqual(client.responses.calls, 3)
            self.assertEqual(
                [item["status"] for item in third],
                ["skipped_exact_input", "completed"],
            )
            forced = run_classification_batch(
                agent, components, CATALOG_ROOT, output, force=True, client=client
            )
            self.assertEqual(client.responses.calls, 5)
            self.assertTrue(all(item["status"] == "completed" for item in forced))
            persisted = "\n".join(
                path.read_text(encoding="utf-8") for path in root.rglob("*.json")
            )
            self.assertNotIn(secret, persisted)
            receipt = json.loads((output / "b.json").read_text(encoding="utf-8"))
            self.assertIn("usage", receipt)
            self.assertGreater(receipt["charged_usd"], 0)

    def test_classification_batch_stops_at_first_invalid_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            components = []
            for name in ("a", "b", "c"):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"name": name}), encoding="utf-8")
                components.append(path)
            client = _Client(
                [_classification_value(), _classification_value(domains=["unknown"])]
            )
            output = root / "proposals"
            with self.assertRaisesRegex(
                RuntimeError,
                "classification target 'b'.*unknown physical domains",
            ):
                run_classification_batch(
                    ClassificationAgent(BudgetLedger(root / "ledger.json", 100.0)),
                    components,
                    CATALOG_ROOT,
                    output,
                    client=client,
                )
            self.assertEqual(client.responses.calls, 2)
            self.assertTrue((output / "a.json").is_file())
            self.assertFalse((output / "b.json").exists())
            self.assertFalse((output / "c.json").exists())

    def test_full_modeling_proposal_stays_staged_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            output = root / "proposals"
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            secret = "modeling-secret-never-persist"
            agent = ModelingAgent(
                ledger,
                staging,
                codex_binary="codex-fixture",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
                api_key=secret,
            )
            inputs = _real_modeling_inputs()
            value = _modeling_value(
                path="components/romi-drive.json",
                content='{"id":"romi-drive"}',
            )
            process = subprocess.CompletedProcess(
                args=["codex"],
                returncode=0,
                stdout=_event_stream(value, include_usage=True),
                stderr=f"provider diagnostic accidentally echoed {secret}",
            )

            with mock.patch(
                "contraption.part_import.agents.subprocess.run", return_value=process
            ) as run_mock:
                first = run_modeling_proposal(
                    agent, inputs, "romi_drive", output
                )
                receipt_path = output / "romi_drive.json"
                legacy_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                for artifact in legacy_receipt["proposal"]["artifacts"]:
                    artifact.pop("content", None)
                receipt_path.write_text(
                    json.dumps(legacy_receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                events_path = (
                    Path(first["staging_artifacts"]).parent / "codex-events.jsonl"
                )
                legacy_events = []
                for raw in events_path.read_text(encoding="utf-8").splitlines():
                    event = json.loads(raw)
                    item = event.get("item")
                    if (
                        event.get("type") == "item.completed"
                        and isinstance(item, dict)
                        and item.get("type") == "agent_message"
                    ):
                        message = json.loads(item["text"])
                        for artifact in message["artifacts"]:
                            artifact.pop("content", None)
                        item["text"] = json.dumps(message, sort_keys=True)
                    legacy_events.append(json.dumps(event, sort_keys=True))
                events_path.write_text(
                    "\n".join(legacy_events) + "\n", encoding="utf-8"
                )
                second = run_modeling_proposal(
                    agent, inputs, "romi_drive", output
                )
                staged_model = (
                    Path(first["staging_artifacts"])
                    / "components"
                    / "romi-drive.json"
                )
                (staged_model.parent / "notes.md").write_text(
                    "validated but different recovered proposal\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    FileExistsError, "differ from recovered output"
                ):
                    run_modeling_proposal(agent, inputs, "romi_drive", output)

            self.assertEqual(run_mock.call_count, 2)
            login_call, execution_call = run_mock.call_args_list
            self.assertEqual(
                login_call.args[0], ["codex-fixture", "login", "--with-api-key"]
            )
            self.assertEqual(login_call.kwargs["input"], secret)
            command = execution_call.args[0]
            self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
            self.assertIn("python -I -m contraption.part_import.model_validation_tool", command[-1])
            child_env = execution_call.kwargs["env"]
            self.assertEqual(
                Path(child_env["PATH"].split(os.pathsep)[0]),
                Path(sys.executable).absolute().parent,
            )
            self.assertEqual(child_env["PYTHONSAFEPATH"], "1")
            self.assertNotIn("OPENAI_API_KEY", child_env)
            self.assertFalse(Path(child_env["CODEX_HOME"]).exists())
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "skipped_exact_input")
            self.assertEqual(first["validation_activity"]["logged_calls"], 0)
            self.assertFalse(first["promoted"])
            artifacts = Path(first["staging_artifacts"])
            self.assertTrue((artifacts / "components" / "romi-drive.json").is_file())
            receipt = json.loads(
                (output / "romi_drive.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["usage"]["input_tokens"], 100)
            self.assertGreater(receipt["charged_usd"], 0)
            self.assertFalse(receipt["promoted"])
            self.assertNotIn(
                secret,
                "\n".join(
                    path.read_text(encoding="utf-8", errors="ignore")
                    for path in root.rglob("*")
                    if path.is_file()
                ),
            )

    def test_modeling_api_key_login_is_ephemeral_and_pre_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = "temporary-login-secret"
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            agent = ModelingAgent(
                ledger,
                root / "staging",
                codex_binary="codex-fixture",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
                api_key=secret,
            )
            login_failure = subprocess.CompletedProcess(
                args=["codex-fixture", "login", "--with-api-key"],
                returncode=1,
                stdout="",
                stderr=f"login rejected {secret}",
            )

            with mock.patch(
                "contraption.part_import.agents.subprocess.run",
                return_value=login_failure,
            ) as run_mock:
                with self.assertRaisesRegex(RuntimeError, "API-key login failed") as caught:
                    agent.run(_real_modeling_inputs())

            self.assertNotIn(secret, str(caught.exception))
            self.assertEqual(run_mock.call_count, 1)
            call = run_mock.call_args
            self.assertEqual(call.kwargs["input"], secret)
            self.assertNotIn("OPENAI_API_KEY", call.kwargs["env"])
            self.assertFalse(Path(call.kwargs["env"]["CODEX_HOME"]).exists())
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["spent_usd"], 0.0)
            self.assertEqual(snapshot["events"][-1]["status"], "cancelled_before_dispatch")


class PriorTwentyOfflineReplayTests(unittest.TestCase):
    def test_prior_twenty_preflight_and_family_plan_are_deterministic(self):
        session = PROJECT_ROOT / "assembled_contraptions" / "part_import_2026_08_18"
        bundle = _load_agent_job_bundle(session / "agent_jobs.json")
        component_paths = sorted((session / "component_inputs").glob("*.json"))
        self.assertEqual(len(component_paths), 20)

        preflights = {path.stem: modeling_preflight(path) for path in component_paths}
        eligible = sorted(name for name, report in preflights.items() if report["eligible"])
        deferred = sorted(name for name, report in preflights.items() if not report["eligible"])
        self.assertEqual(len(eligible), 10)
        self.assertEqual(len(deferred), 10)
        self.assertTrue(all(name.startswith("yageo_") for name in eligible))
        self.assertTrue(all(name.startswith("murata_") for name in deferred))
        self.assertTrue(
            all(report["unsupported_physics"] == ["thermal"] for name, report in preflights.items() if name in deferred)
        )

        plans = [
            build_modeling_import_plan(_full_modeling_inputs(bundle, target))
            for target in eligible
        ]
        identities = {
            (plan["recommended_model"]["id"], plan["recommended_model"]["sha256"])
            for plan in plans
        }
        self.assertEqual(
            identities,
            {
                (
                    "electrical.resistor.ideal",
                    "sha256:10a026720ba1b0eec5a5cd2ea84f7c01bbcb030ac00ddbe5c5bf4251a08849c7",
                )
            },
        )
        self.assertTrue(
            all(
                plan["recommended_instantiation_root"].endswith(plan["target_id"])
                for plan in plans
            )
        )
        self.assertTrue(
            all(
                plan["validation"]["maximum_agent_calls"]
                == MAX_MODELING_VALIDATION_ATTEMPTS
                == 3
                for plan in plans
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            agent = ModelingAgent(ledger, root / "staging", codex_binary="must-not-run")
            with mock.patch.object(
                agent, "run", side_effect=AssertionError("physics agent dispatched")
            ) as dispatch:
                results = [
                    run_modeling_proposal(
                        agent,
                        _full_modeling_inputs(bundle, target),
                        target,
                        root / "proposals",
                    )
                    for target in deferred
                ]
            dispatch.assert_not_called()
            self.assertTrue(
                all(item["status"] == "deferred_unsupported_physics" for item in results)
            )
            self.assertTrue(all(item["charged_usd"] == 0.0 for item in results))
            self.assertEqual(ledger.snapshot()["events"], [])
            isolated_registry = root / "isolated-registry"
            record_ids: set[str] = set()
            for item in results:
                self.assertIn("staging_artifacts", item)
                self.assertIn("procurement_receipt", item)
                self.assertEqual(len(item["procurement_records"]), 1)
                artifacts = Path(item["staging_artifacts"])
                trusted = verify_proposal_procurement_receipt(artifacts)
                self.assertEqual(len(trusted), 1)
                record = ProcurementRecord.from_json(
                    trusted[0].read_text(encoding="utf-8")
                )
                self.assertEqual(record.manufacturer, "Murata")
                self.assertEqual(record.provides, ())
                record_ids.add(record.id)
                ModelingAgent.promote(artifacts, isolated_registry)
            self.assertEqual(len(record_ids), 10)
            replayed = tuple(
                sorted(isolated_registry.rglob("*.procurement"))
            )
            self.assertEqual(len(replayed), 10)
            self.assertTrue(
                all(
                    ProcurementRecord.from_json(path.read_text(encoding="utf-8")).provides
                    == ()
                    for path in replayed
                )
            )

    def test_deferred_input_without_identity_creates_no_procurement_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component = root / "thermal_unknown.json"
            component.write_text(
                json.dumps(
                    {
                        "purpose": "unidentified thermal sample",
                        "domains": ["thermal"],
                    }
                ),
                encoding="utf-8",
            )
            base = _real_modeling_inputs()
            inputs = ModelingInputs(
                constraints=base.constraints,
                gold_templates=base.gold_templates,
                interfaces=base.interfaces,
                direct_hierarchy=base.direct_hierarchy,
                component_information=component,
            )
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            agent = ModelingAgent(
                ledger, root / "staging", codex_binary="must-not-run"
            )
            with mock.patch.object(
                agent, "run", side_effect=AssertionError("physics agent dispatched")
            ) as dispatch:
                result = run_modeling_proposal(
                    agent, inputs, "thermal_unknown", root / "proposals"
                )
            dispatch.assert_not_called()
            self.assertEqual(result["status"], "deferred_unsupported_physics")
            self.assertNotIn("staging_artifacts", result)
            self.assertNotIn("procurement_receipt", result)
            self.assertFalse(any((root / "staging").rglob("*.procurement")))
            self.assertEqual(ledger.snapshot()["events"], [])

    def test_public_modeling_entry_rejects_bad_schema_before_deferred_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component = root / "thermal_unknown.json"
            component.write_text(
                '{"purpose":"unidentified thermal sample","domains":["thermal"]}\n',
                encoding="utf-8",
            )
            base = _real_modeling_inputs()
            inputs = ModelingInputs(
                constraints=base.constraints,
                gold_templates=base.gold_templates,
                interfaces=base.interfaces,
                direct_hierarchy=base.direct_hierarchy,
                component_information=component,
            )
            malformed = json.loads(json.dumps(MODELING_SCHEMA))
            malformed["properties"]["artifacts"]["items"]["properties"][
                "content"
            ]["type"] = "bogus-provider-type"
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            agent = ModelingAgent(ledger, root / "staging", codex_binary="must-not-run")
            with mock.patch(
                "contraption.part_import.agents.MODELING_SCHEMA", malformed
            ), mock.patch.object(
                agent, "run", side_effect=AssertionError("agent dispatched")
            ) as dispatch:
                with self.assertRaisesRegex(ValueError, "unsupported names"):
                    run_modeling_proposal(
                        agent, inputs, "thermal_unknown", root / "proposals"
                    )
            dispatch.assert_not_called()
            self.assertEqual(ledger.snapshot()["events"], [])
            self.assertFalse((root / "staging").exists())
            self.assertFalse((root / "proposals").exists())


class ModelingValidationToolTests(unittest.TestCase):
    def test_prepare_workspace_uses_one_authoritative_component_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component = root / "coherent.json"
            initial = {
                "manufacturer": "Initial Manufacturer",
                "product": "INITIAL-100",
                "purpose": "initial snapshot purpose",
                "domains": ["electrical"],
                "connector_fabrication": [
                    {
                        "part": "initial-part",
                        "connector": "p",
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
                ],
            }
            mutated = {
                "manufacturer": "Mutated Manufacturer",
                "product": "MUTATED-999",
                "purpose": "mutated source purpose",
                "domains": ["mechanical"],
            }
            initial_payload = (json.dumps(initial, sort_keys=True) + "\n").encode()
            component.write_bytes(initial_payload)
            base = _real_modeling_inputs()
            inputs = ModelingInputs(
                constraints=base.constraints,
                gold_templates=base.gold_templates,
                interfaces=base.interfaces,
                direct_hierarchy=base.direct_hierarchy,
                component_information=component,
            )
            agent = ModelingAgent(
                BudgetLedger(root / "ledger.json", 1.0), root / "staging"
            )
            from contraption.part_import import deterministic_assets

            real_stage_plan = deterministic_assets.stage_plan
            observed_component_paths: list[Path] = []

            def mutate_original_after_snapshot(component_information, destination, **kwargs):
                observed_component_paths.append(Path(component_information).resolve())
                component.write_text(
                    json.dumps(mutated, sort_keys=True) + "\n", encoding="utf-8"
                )
                return real_stage_plan(component_information, destination, **kwargs)

            with mock.patch(
                "contraption.part_import.deterministic_assets.stage_plan",
                side_effect=mutate_original_after_snapshot,
            ):
                workspace = agent.prepare_workspace(inputs, "snapshot-coherence")

            snapshot = workspace.parent / "inputs" / component.name
            self.assertEqual(observed_component_paths, [snapshot.resolve()])
            self.assertEqual(snapshot.read_bytes(), initial_payload)
            plan = json.loads((workspace / "IMPORT_PLAN.json").read_text())
            self.assertEqual(
                plan["source_identity_facts"]["manufacturer"],
                "Initial Manufacturer",
            )
            self.assertEqual(
                plan["source_identity_facts"]["product"], "INITIAL-100"
            )
            instructions = (workspace / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("Initial Manufacturer", instructions)
            self.assertNotIn("Mutated Manufacturer", instructions)
            procurement_context = json.loads(
                (workspace.parent / HOST_PROCUREMENT_CONTEXT_FILENAME).read_text()
            )
            fabrication_context = json.loads(
                (workspace.parent / HOST_FABRICATION_CONTEXT_FILENAME).read_text()
            )
            self.assertEqual(
                procurement_context["component_input"]["sha256"],
                "sha256:" + hashlib.sha256(initial_payload).hexdigest(),
            )
            self.assertEqual(
                procurement_context["component_input"]["path"],
                f"inputs/{component.name}",
            )
            self.assertEqual(
                fabrication_context["records"][0]["part"], "initial-part"
            )
            component_entries = [
                item
                for item in json.loads(
                    (workspace / "INPUT_MANIFEST.json").read_text()
                )
                if item["protected_path"] == f"../inputs/{component.name}"
            ]
            self.assertEqual(len(component_entries), 1)
            self.assertEqual(
                component_entries[0]["sha256"],
                hashlib.sha256(initial_payload).hexdigest(),
            )

    def test_recovery_rejects_tampered_protected_workspace_controls(self):
        for relative in (
            "IMPORT_PLAN.json",
            "AGENTS.md",
            "INPUT_MANIFEST.json",
            "output-schema.json",
        ):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                agent = ModelingAgent(
                    BudgetLedger(root / "ledger.json", 1.0), root / "staging"
                )
                workspace = agent.prepare_workspace(
                    _real_modeling_inputs(), f"tampered-{Path(relative).stem}"
                )
                target = workspace / relative
                target.write_bytes(target.read_bytes() + b"tampered\n")
                with self.assertRaisesRegex(
                    ValueError, "protected input integrity mismatch"
                ):
                    agent.recover_workspace(workspace)

    def test_agent_validator_refuses_a_fourth_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = ModelingAgent(
                BudgetLedger(root / "ledger.json", 1.0), root / "staging"
            )
            workspace = agent.prepare_workspace(_real_modeling_inputs(), "attempt-cap")
            candidate = workspace / "candidate" / "resistor.pmdl"
            candidate.write_bytes(
                (CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl").read_bytes()
            )

            first_three = [
                validate_candidate("candidate/resistor.pmdl", workspace=workspace)
                for _ in range(MAX_MODELING_VALIDATION_ATTEMPTS)
            ]
            refused = validate_candidate(
                "candidate/resistor.pmdl", workspace=workspace
            )

            self.assertTrue(all(item["valid"] for item in first_three))
            self.assertFalse(refused["valid"])
            self.assertEqual(
                refused["issues"][0]["code"], "validator.attempt_limit"
            )
            self.assertEqual(
                validation_activity(workspace)["logged_calls"],
                MAX_MODELING_VALIDATION_ATTEMPTS,
            )

    def test_drafts_validate_iteratively_with_detailed_feedback_and_call_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = ModelingAgent(BudgetLedger(root / "ledger.json", 1.0), root / "staging")
            workspace = agent.prepare_workspace(_real_modeling_inputs(), "validator-run")

            self.assertFalse((workspace / "inputs").exists())
            self.assertTrue((workspace.parent / "inputs").is_dir())
            self.assertTrue(
                (workspace.parent / HOST_FABRICATION_CONTEXT_FILENAME).is_file()
            )
            integrity = assert_workspace_integrity(workspace)
            self.assertGreaterEqual(len(integrity["checked"]), 8)
            instructions = (workspace / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn(
                "model_catalog/electrical/resistors/resistor.pmdl",
                instructions,
            )
            for name in (
                "README.md",
                "PMDL.md",
                "PMDL_INTERFACES.md",
                "STATIC_PART.md",
                "MODEL_INSTANCE.md",
                "VERIFICATION.md",
                "DETERMINISTIC_INGESTION.md",
            ):
                self.assertIn(f"docs/structured_formats/{name}", instructions)
            self.assertNotIn("docs/structured_formats/CONTRAPTION.md", instructions)
            self.assertIn("## BEGIN IMPORT_PLAN.json", instructions)
            plan = json.loads((workspace / "IMPORT_PLAN.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["validation"]["maximum_agent_calls"], 3)
            self.assertEqual(
                plan["recommended_model"]["id"], "electromechanical.dc_motor.ideal"
            )
            manifest = json.loads(
                (workspace / "INPUT_MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertTrue(all(item["source_label"] for item in manifest))

            candidate = workspace / "candidate" / "romi_drive.pmdl"
            candidate.write_text("{}\n", encoding="utf-8")
            invalid = validate_candidate("candidate/romi_drive.pmdl", workspace=workspace)
            self.assertFalse(invalid["valid"])
            self.assertEqual(invalid["call_number"], 1)
            self.assertEqual(invalid["issues"][0]["code"], "pmdl.parse")
            self.assertIn("candidate/romi_drive.pmdl", invalid["issues"][0]["path"])

            candidate.write_bytes(
                (CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl").read_bytes()
            )
            valid = validate_candidate("candidate/romi_drive.pmdl", workspace=workspace)
            self.assertTrue(valid["valid"])
            self.assertEqual(valid["call_number"], 2)
            self.assertEqual(valid["candidate_sha256"], hashlib.sha256(candidate.read_bytes()).hexdigest())
            activity = validation_activity(workspace)
            self.assertEqual(activity["logged_calls"], 2)
            self.assertEqual(activity["successful_calls"], 1)
            self.assertEqual(activity["failed_calls"], 1)

    def test_validator_rejects_escape_host_code_and_tampered_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = ModelingAgent(BudgetLedger(root / "ledger.json", 1.0), root / "staging")
            workspace = agent.prepare_workspace(_real_modeling_inputs(), "validator-run")
            outside = workspace / "outside.pmdl"
            outside.write_bytes(
                (CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl").read_bytes()
            )

            escaped = validate_candidate("outside.pmdl", workspace=workspace)
            self.assertFalse(escaped["valid"])
            self.assertEqual(escaped["issues"][0]["code"], "validator.contract")
            script = workspace / "candidate" / "host.py"
            script.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")
            host_code = validate_candidate("candidate/host.py", workspace=workspace)
            self.assertFalse(host_code["valid"])
            self.assertIn(".pmdl", host_code["issues"][0]["message"])

            protected = next((workspace.parent / "inputs").iterdir())
            protected.write_text("tampered\n", encoding="utf-8")
            candidate = workspace / "candidate" / "safe.pmdl"
            candidate.write_bytes(
                (CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl").read_bytes()
            )
            tampered = validate_candidate("candidate/safe.pmdl", workspace=workspace)
            self.assertFalse(tampered["valid"])
            self.assertIn("integrity mismatch", tampered["issues"][0]["message"])

    def test_host_rechecks_protected_inputs_after_codex_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            agent = ModelingAgent(
                ledger,
                root / "staging",
                codex_binary="codex-fixture",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
            )

            def tampering_run(*args, **kwargs):
                workspace = Path(kwargs["cwd"])
                protected = next((workspace.parent / "inputs").iterdir())
                protected.write_text("tampered by fixture\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    args=args[0],
                    returncode=0,
                    stdout=_event_stream(_modeling_value()),
                    stderr="",
                )

            with mock.patch("contraption.part_import.agents.subprocess.run", side_effect=tampering_run):
                with self.assertRaisesRegex(ValueError, "input integrity failed"):
                    agent.run(_real_modeling_inputs(), canary=True)
            self.assertEqual(ledger.snapshot()["events"][-1]["status"], "failed_after_dispatch")



class ModelingRecoveryTests(unittest.TestCase):
    def test_fabrication_receipt_rejects_post_materialization_invention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            component = run_root / "inputs" / "component.json"
            component.parent.mkdir(parents=True)
            component.write_text(
                '{"purpose":"fabrication facts intentionally absent"}',
                encoding="utf-8",
            )
            write_host_fabrication_context(
                run_root,
                component_input=component,
                source_name="component_inputs/component.json",
            )
            proposed = ModelingAgent._materialize_artifacts(
                run_root / "workspace", _canary_modeling_value()
            )
            self.assertTrue(proposal_fabrication_receipt_path(proposed).is_file())
            static_path = next(proposed.rglob("static.part"))
            data = json.loads(static_path.read_text(encoding="utf-8"))
            self.assertEqual(data["connectors"][0]["fabrication"]["status"], "missing")
            data["connectors"][0]["fabrication"] = {
                "kind": "electrical_termination",
                "status": "specified",
                "missing": [],
                "conductor": {
                    "standard": {
                        "family": "awg",
                        "authority": "ASTM",
                        "document": "ASTM B258",
                        "designation": "24 AWG",
                        "gauge_awg": 24,
                    },
                    "conductor_count": 1,
                    "material": "copper",
                    "cross_section_m2": 2.05e-7,
                    "insulation_standard": {
                        "family": "manufacturer_cable",
                        "authority": "Invented",
                        "document": "FAKE-WIRE",
                        "designation": "PVC-300V",
                    },
                    "voltage_rating_v": 300.0,
                    "temperature_rating_k": 378.15,
                },
                "termination": {
                    "method": "solder",
                    "installation_process": "invented qualified solder process",
                },
                "evidence": [
                    {
                        "kind": "manual",
                        "source": "invented-after-materialization.json",
                        "sha256": "sha256:" + "a" * 64,
                    }
                ],
            }
            static_path.write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
            # The mutation is schema-valid; rejection comes from host authority,
            # not from ordinary static.part validation.
            StaticPartSpec.from_json(static_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(
                ValueError, "fabrication receipt static bytes or fabrication payload changed"
            ):
                ModelingAgent.promote(proposed, root / "registry")
            self.assertFalse((root / "registry").exists())

    def test_mixed_case_procurement_cannot_bypass_host_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposed = root / "run" / "workspace" / "proposed"
            records = proposed / "procurement" / "records"
            records.mkdir(parents=True)
            source = next(
                (CATALOG_ROOT / "procurement" / "records").glob("*.procurement")
            )
            (records / "untrusted.PROCUREMENT").write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "protected host receipt"):
                verify_proposal_procurement_receipt(proposed)

            (records / "untrusted.procurement").write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "case-colliding"):
                verify_proposal_procurement_receipt(proposed)

    def test_promotion_reverifies_procurement_after_exact_snapshot_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            component = run_root / "inputs" / "component.json"
            component.parent.mkdir(parents=True)
            component.write_text(
                json.dumps(
                    {
                        "manufacturer": "Snapshot Example",
                        "product": "SNAP-100",
                        "domains": ["electrical"],
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            write_host_procurement_context(
                run_root,
                component_input=component,
                source_name="component_inputs/snapshot-example.json",
            )
            proposed = ModelingAgent._materialize_artifacts(
                run_root / "workspace", _canary_modeling_value()
            )
            self.assertEqual(len(verify_proposal_procurement_receipt(proposed)), 1)

            real_copyfile = __import__("shutil").copyfile

            def corrupt_procurement_snapshot(source, destination, *args, **kwargs):
                result = real_copyfile(source, destination, *args, **kwargs)
                target = Path(destination)
                if (
                    Path(source).suffix == ".procurement"
                    and any(
                        part.startswith("contraption-promotion-")
                        for part in target.parts
                    )
                ):
                    target.write_bytes(target.read_bytes() + b" ")
                return result

            with mock.patch(
                "contraption.part_import.agents.shutil.copyfile",
                side_effect=corrupt_procurement_snapshot,
            ), self.assertRaisesRegex(ValueError, "procurement artifact hash changed"):
                ModelingAgent.promote(proposed, root / "registry")
            self.assertFalse((root / "registry").exists())

    def test_host_procurement_binds_exact_static_hash_and_promotes_with_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            component = run_root / "inputs" / "component.json"
            component.parent.mkdir(parents=True)
            component.write_text(
                json.dumps(
                    {
                        "manufacturer": "Example",
                        "product": "EX-100",
                        "domains": ["electrical"],
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            write_host_procurement_context(
                run_root,
                component_input=component,
                source_name="component_inputs/example.json",
            )

            proposed = ModelingAgent._materialize_artifacts(
                run_root / "workspace", _canary_modeling_value()
            )
            trusted = verify_proposal_procurement_receipt(proposed)
            self.assertEqual(len(trusted), 1)
            record = ProcurementRecord.from_json(
                trusted[0].read_text(encoding="utf-8")
            )
            static_path = next(proposed.rglob("static.part"))
            static = StaticPartSpec.from_json(static_path.read_text(encoding="utf-8"))
            self.assertEqual(len(record.provides), 1)
            provision = record.provides[0]
            self.assertEqual(provision.part, static.id)
            self.assertEqual(provision.version, static.version)
            self.assertEqual(provision.static_sha256, static.sha256)
            self.assertEqual(provision.quantity, 1)
            self.assertTrue(proposal_procurement_receipt_path(proposed).is_file())

            registry = root / "registry"
            written = ModelingAgent.promote(proposed, registry)
            self.assertIn(
                registry / trusted[0].relative_to(proposed), written
            )
            self.assertTrue(any(registry.rglob("static.part")))
            promoted_record = ProcurementRecord.from_json(
                (registry / trusted[0].relative_to(proposed)).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(promoted_record.provides, record.provides)

    def test_host_procurement_receipt_rejects_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            component = run_root / "inputs" / "component.json"
            component.parent.mkdir(parents=True)
            component.write_text(
                '{"manufacturer":"Example","product":"EX-200"}',
                encoding="utf-8",
            )
            write_host_procurement_context(
                run_root,
                component_input=component,
                source_name="component_inputs/example.json",
            )
            proposed = ModelingAgent._materialize_artifacts(
                run_root / "workspace", _canary_modeling_value()
            )
            (procurement_path,) = verify_proposal_procurement_receipt(proposed)
            procurement_path.write_bytes(procurement_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "artifact hash changed"):
                ModelingAgent.promote(proposed, root / "registry")
            self.assertFalse((root / "registry").exists())

    def test_host_repairs_candidate_model_hash_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            candidate = workspace / "candidate"
            candidate.mkdir()
            value = _canary_modeling_value()
            for artifact in value["artifacts"]:
                target = candidate / artifact["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(artifact["content"], encoding="utf-8")
            model_path = next(candidate.rglob("v1.model"))
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["model"]["sha256"] = "sha256:" + "0" * 64
            model_path.write_text(json.dumps(model), encoding="utf-8")

            recovered = ModelingAgent._value_from_candidate_files(workspace)

            repaired = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertEqual(
                repaired["model"]["sha256"],
                "sha256:10a026720ba1b0eec5a5cd2ea84f7c01bbcb030ac00ddbe5c5bf4251a08849c7",
            )
            self.assertIn("Host repaired 1 model-instance PMDL hash", " ".join(recovered["evidence"]))

    def test_candidate_strips_identical_base_file_and_rejects_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            candidate = workspace / "candidate"
            base_relative = Path("electrical/resistors/resistor.pmdl")
            base = CATALOG_ROOT / base_relative
            duplicate = candidate / base_relative
            duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes(base.read_bytes())
            extra = candidate / "notes.json"
            extra.write_text('{"id":"new-evidence"}\n', encoding="utf-8")

            recovered = ModelingAgent._value_from_candidate_files(workspace)

            self.assertFalse(duplicate.exists())
            self.assertEqual(
                [item["path"] for item in recovered["artifacts"]], ["notes.json"]
            )

            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_text('{"format":"pmdl-1","tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable base catalog"):
                ModelingAgent._value_from_candidate_files(workspace)

    def test_promotion_uses_validated_atomic_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposed = ModelingAgent._materialize_artifacts(
                root / "workspace", _canary_modeling_value()
            )
            registry = root / "registry"

            written = ModelingAgent.promote(proposed, registry)

            target = (
                registry
                / "electrical"
                / "resistors"
                / "fixed_resistors"
                / "instantiations"
                / "fixture-100ohm-resistor"
                / "v1.model"
            )
            self.assertIn(target, written)
            self.assertEqual(
                target.read_bytes(),
                (proposed / target.relative_to(registry)).read_bytes(),
            )
            self.assertFalse(any(registry.rglob("*.tmp")))

    def test_promotion_revalidates_after_staged_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposed = root / "proposed"
            model = proposed / "electrical" / "resistors" / "resistor.pmdl"
            model.parent.mkdir(parents=True)
            model.write_bytes(
                (CATALOG_ROOT / "electrical" / "resistors" / "resistor.pmdl").read_bytes()
            )
            ModelingAgent.validate_artifacts(proposed)
            model.write_text('{"format":"pmdl-1","tampered":true}\n', encoding="utf-8")

            registry = root / "registry"
            with self.assertRaises(ValueError):
                ModelingAgent.promote(proposed, registry)
            self.assertFalse(
                (registry / "electrical" / "resistors" / "resistor.pmdl").exists()
            )

    def test_host_validation_accepts_normalized_deterministic_source_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            candidate.mkdir()
            for name in (
                "fixture.FCStd",
                "fixture.IGES",
                "fixture.igs",
                "fixture.WRL",
                "fixture.vrml",
            ):
                (candidate / name).write_bytes(b"deterministic-source")

            ModelingAgent.validate_artifacts(candidate)

    def test_trusted_host_gltf_external_bin_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            candidate.mkdir()
            gltf = candidate / "scene.gltf"
            linked = candidate / "scene.bin"
            gltf.write_text(
                '{"asset":{"version":"2.0"},"buffers":[{"uri":"scene.bin"}]}',
                encoding="utf-8",
            )
            linked.write_bytes(b"verified-linked-buffer")

            with self.assertRaisesRegex(ValueError, "unsupported generated artifact"):
                ModelingAgent.validate_artifacts(candidate)
            ModelingAgent.validate_artifacts(
                candidate,
                trusted_host_artifacts=(gltf, linked),
            )

    def test_materialized_shape_with_linked_gltf_buffer_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts, linked = _materialize_linked_shape(root)
            trusted = verify_proposal_shape_receipt(artifacts)
            ModelingAgent.validate_artifacts(
                artifacts,
                trusted_host_artifacts=trusted,
            )

            registry = root / "registry"
            written = ModelingAgent.promote(artifacts, registry)
            promoted = next((registry / "shape" / "source").glob("*-scene.bin"))
            self.assertIn(promoted, written)
            self.assertEqual(promoted.read_bytes(), linked.read_bytes())

    def test_forged_shape_manifest_cannot_launder_unsupported_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts, _linked = _materialize_linked_shape(
                root,
                linked_name="payload.py",
                write_receipt=False,
            )

            with self.assertRaisesRegex(ValueError, "protected host shape receipt"):
                ModelingAgent.promote(artifacts, root / "registry")
            self.assertFalse((root / "registry").exists())

    def test_shape_receipt_rejects_post_validation_manifest_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts, _linked = _materialize_linked_shape(root)
            trusted = verify_proposal_shape_receipt(artifacts)
            ModelingAgent.validate_artifacts(
                artifacts,
                trusted_host_artifacts=trusted,
            )
            manifest = artifacts / "shape" / "shape.artifact.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")

            with self.assertRaisesRegex(ValueError, "artifact hash changed"):
                ModelingAgent.promote(artifacts, root / "registry")

    def test_shape_receipt_rejects_unbound_linked_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts, _linked = _materialize_linked_shape(root)
            trusted = verify_proposal_shape_receipt(artifacts)
            ModelingAgent.validate_artifacts(
                artifacts,
                trusted_host_artifacts=trusted,
            )
            (artifacts / "shape" / "source" / "unbound.bin").write_bytes(
                b"candidate-controlled"
            )

            with self.assertRaisesRegex(ValueError, "artifact set changed"):
                ModelingAgent.promote(artifacts, root / "registry")

    def test_canary_contract_requires_static_part_and_model_instance(self):
        with self.assertRaisesRegex(ValueError, "static.part"):
            ModelingAgent._validate_canary_value(
                {
                    **_modeling_value(),
                    "artifacts": [
                        {"path": "electrical/resistors/resistor.pmdl", "content": "{}"},
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "static.part"):
            ModelingAgent._validate_canary_value(_modeling_value())

    def test_instantiation_bundle_must_use_a_declared_contract_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            wrong_prefix = Path("electrical/resistor/fixed-resistor")
            for artifact in _canary_modeling_value()["artifacts"]:
                source_path = Path(artifact["path"])
                if "instantiations" not in source_path.parts:
                    continue
                suffix = Path(
                    *source_path.parts[source_path.parts.index("instantiations") :]
                )
                target = candidate / wrong_prefix / suffix
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(artifact["content"], encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "interface.pmdl"):
                ModelingAgent.validate_artifacts(candidate)

    def test_recovers_completed_structured_event_in_existing_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            workspace = _prepared_workspace(staging)
            value = _modeling_value()
            (workspace / "codex-events.jsonl").write_text(
                _event_stream(value), encoding="utf-8"
            )
            ledger = BudgetLedger(root / "ledger.json", limit_usd=1.0)
            agent = ModelingAgent(ledger, staging)

            artifacts, recovered = agent.recover_workspace(workspace.parent)

            self.assertEqual(recovered, value)
            self.assertEqual(
                (artifacts / "components" / "target.json").read_text(encoding="utf-8"),
                '{"id":"target"}',
            )
            # Recovery is offline/idempotent and does not fabricate a new charge.
            again, again_value = agent.recover_workspace(workspace)
            self.assertEqual(again, artifacts)
            self.assertEqual(again_value, value)
            self.assertEqual(ledger.snapshot()["events"], [])

    def test_recovery_rejects_bad_schema_before_materialization_or_settlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            run_id = "modeling-malformed-schema"
            workspace = _prepared_workspace(staging, run_id)
            (workspace / "codex-events.jsonl").write_text(
                _event_stream(_modeling_value()), encoding="utf-8"
            )
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            ledger.reserve(run_id, 0.25, {"kind": "modeling", "model": "fixture"})
            agent = ModelingAgent(ledger, staging)
            malformed = json.loads(json.dumps(MODELING_SCHEMA))
            malformed["properties"]["artifacts"]["items"]["properties"][
                "content"
            ]["type"] = "bogus-provider-type"

            with mock.patch(
                "contraption.part_import.agents.MODELING_SCHEMA", malformed
            ), self.assertRaisesRegex(ValueError, "unsupported names"):
                agent.recover_workspace(workspace)

            snapshot = ledger.snapshot()
            self.assertIn(run_id, snapshot["reserved"])
            self.assertEqual(snapshot["events"], [])
            self.assertFalse((workspace / "proposed").exists())
            self.assertFalse(any(workspace.glob(".proposed-*")))

    def test_final_completed_message_is_authoritative_and_schema_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            workspace = _prepared_workspace(staging)
            (workspace / "codex-events.jsonl").write_text(
                _event_stream(_modeling_value(), {"summary": "incomplete correction"}),
                encoding="utf-8",
            )
            agent = ModelingAgent(BudgetLedger(root / "ledger.json", 1.0), staging)

            with self.assertRaisesRegex(ValueError, "missing keys"):
                agent.recover_workspace(workspace)
            self.assertFalse((workspace / "proposed").exists())

    def test_existing_workspace_settles_an_open_reservation_conservatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            run_id = "modeling-unsettled"
            workspace = _prepared_workspace(staging, run_id)
            (workspace / "codex-events.jsonl").write_text(
                _event_stream(_modeling_value()), encoding="utf-8"
            )
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            ledger.reserve(run_id, 0.25, {"kind": "modeling", "model": "fixture"})
            agent = ModelingAgent(ledger, staging)

            agent.recover_workspace(workspace)

            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["reserved"], {})
            self.assertAlmostEqual(snapshot["spent_usd"], 0.25)
            self.assertEqual(
                snapshot["events"][-1]["status"], "recovered_existing_workspace"
            )
            self.assertIsNone(snapshot["events"][-1]["usage"])

    def test_recovery_never_bypasses_artifact_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            workspace = _prepared_workspace(staging)
            (workspace / "codex-events.jsonl").write_text(
                _event_stream(_modeling_value(content="{not valid JSON")),
                encoding="utf-8",
            )
            agent = ModelingAgent(BudgetLedger(root / "ledger.json", 1.0), staging)

            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                agent.recover_workspace(workspace)
            self.assertFalse((workspace / "proposed").exists())
            self.assertFalse(any(workspace.glob(".proposed-*")))

    def test_nonzero_run_recovers_and_charges_reported_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            workspace = _prepared_workspace(staging, "modeling-run")
            value = _canary_modeling_value()
            stdout = _event_stream(value, include_usage=True)
            process = subprocess.CompletedProcess(
                args=["codex"], returncode=73, stdout=stdout, stderr="rollout exhausted"
            )
            ledger = BudgetLedger(root / "ledger.json", limit_usd=1.0)
            pricing = TokenPricing()
            agent = ModelingAgent(
                ledger,
                staging,
                codex_binary="codex-fixture",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
                pricing=pricing,
            )
            inputs = ModelingInputs(
                constraints=Path("constraints.md"),
                gold_templates=(Path("gold.pmdl"),),
                interfaces=(Path("interface.pmdl"),),
                direct_hierarchy=(),
                component_information=root / "authoritative-target.json",
            )
            inputs.component_information.write_text(
                '{"domains":["electrical"]}\n', encoding="utf-8"
            )
            expected_charge = pricing.cost(Usage(100, 25, 50))

            with mock.patch.object(
                agent, "prepare_workspace", return_value=workspace
            ), mock.patch(
                "contraption.part_import.agents.subprocess.run", return_value=process
            ) as run_mock:
                artifacts, recovered, charged = agent.run(inputs, canary=True)

            self.assertEqual(recovered, value)
            self.assertTrue(
                (
                    artifacts
                    / "electrical"
                    / "resistors"
                    / "fixed_resistors"
                    / "instantiations"
                    / "fixture-100ohm-resistor"
                    / "v1.model"
                ).is_file()
            )
            readme = (
                artifacts
                / "electrical"
                / "resistors"
                / "fixed_resistors"
                / "instantiations"
                / "fixture-100ohm-resistor"
                / "README.md"
            )
            self.assertTrue(readme.is_file())
            documentation = readme.read_text(encoding="utf-8")
            self.assertIn("## Model hypotheses at a glance", documentation)
            self.assertIn("Ideal uncertain resistor", documentation)
            self.assertAlmostEqual(charged, expected_charge)
            event = ledger.snapshot()["events"][-1]
            self.assertEqual(event["status"], "recovered_after_nonzero_exit")
            self.assertEqual(event["usage"]["input_tokens"], 100)
            self.assertEqual(event["cost_basis"], "reported_usage")
            command = run_mock.call_args.args[0]
            self.assertIn("authoritative-target.json", command[-1])
            self.assertIn("exactly one minimal complete catalog import bundle", command[-1])

    def test_nonzero_run_recovers_a_host_valid_candidate_without_final_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            workspace = _prepared_workspace(staging, "modeling-candidate-run")
            value = _canary_modeling_value()
            for artifact in value["artifacts"]:
                target = workspace / "candidate" / artifact["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(artifact["content"], encoding="utf-8")
            process = subprocess.CompletedProcess(
                args=["codex"],
                returncode=73,
                stdout=_event_stream(),
                stderr="rollout exhausted after bundle generation",
            )
            ledger = BudgetLedger(root / "ledger.json", limit_usd=1.0)
            agent = ModelingAgent(
                ledger,
                staging,
                codex_binary="codex-fixture",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
            )
            inputs = ModelingInputs(
                constraints=Path("constraints.md"),
                gold_templates=(Path("gold.pmdl"),),
                interfaces=(Path("interface.pmdl"),),
                direct_hierarchy=(),
                component_information=root / "authoritative-target.json",
            )
            inputs.component_information.write_text(
                '{"domains":["electrical"]}\n', encoding="utf-8"
            )

            with mock.patch.object(
                agent, "prepare_workspace", return_value=workspace
            ), mock.patch(
                "contraption.part_import.agents.subprocess.run", return_value=process
            ):
                artifacts, recovered, _charged = agent.run(inputs, canary=True)

            self.assertIn("host-valid candidate bundle", recovered["summary"])
            self.assertEqual(
                {item["path"] for item in recovered["artifacts"]},
                {item["path"] for item in value["artifacts"]},
            )
            self.assertTrue(
                (
                    artifacts
                    / "electrical"
                    / "resistors"
                    / "fixed_resistors"
                    / "instantiations"
                    / "fixture-100ohm-resistor"
                    / "v1.model"
                ).is_file()
            )
            self.assertEqual(
                ledger.snapshot()["events"][-1]["status"],
                "recovered_after_nonzero_exit",
            )

    def test_zero_exit_prefers_valid_candidate_over_malformed_final_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            workspace = _prepared_workspace(staging, "modeling-zero-exit-candidate")
            value = _canary_modeling_value()
            for artifact in value["artifacts"]:
                target = workspace / "candidate" / artifact["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(artifact["content"], encoding="utf-8")
            process = subprocess.CompletedProcess(
                args=["codex"],
                returncode=0,
                stdout=_event_stream({"summary": "truncated final manifest"}, include_usage=True),
                stderr="",
            )
            ledger = BudgetLedger(root / "ledger.json", limit_usd=1.0)
            agent = ModelingAgent(
                ledger,
                staging,
                codex_binary="codex-fixture",
                rollout_token_limit=1_000,
                max_input_tokens=2_000,
            )
            inputs = ModelingInputs(
                constraints=Path("constraints.md"),
                gold_templates=(Path("gold.pmdl"),),
                interfaces=(Path("interface.pmdl"),),
                direct_hierarchy=(),
                component_information=root / "authoritative-target.json",
            )
            inputs.component_information.write_text(
                '{"domains":["electrical"]}\n', encoding="utf-8"
            )

            with mock.patch.object(
                agent, "prepare_workspace", return_value=workspace
            ), mock.patch(
                "contraption.part_import.agents.subprocess.run", return_value=process
            ):
                artifacts, recovered, _charged = agent.run(inputs, canary=True)

            self.assertTrue((artifacts / value["artifacts"][1]["path"]).is_file())
            self.assertEqual(
                {item["path"] for item in recovered["artifacts"]},
                {item["path"] for item in value["artifacts"]},
            )
            self.assertEqual(
                ledger.snapshot()["events"][-1]["status"],
                "recovered_from_candidate",
            )


if __name__ == "__main__":
    unittest.main()
