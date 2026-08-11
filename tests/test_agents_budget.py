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
    ClassificationAgent,
    ModelingAgent,
    ModelingInputs,
    _validate_shape,
    run_classification_batch,
    run_modeling_proposal,
    validate_classification_proposal,
)
from contraption.part_import.budget import BudgetExceeded, BudgetLedger, TokenPricing, Usage
from contraption.catalog.interfaces import ModelInterfaceCatalog, interface_paths, load_interface_catalog
from contraption.cli import _agent_key, _default_dotenv_path, _torch_diagnostics, build_parser
from contraption.part_import.model_validation_tool import (
    assert_workspace_integrity,
    validate_candidate,
    validation_activity,
    write_validation_context,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = PROJECT_ROOT / "model_catalog"
INTERFACE_DATA = load_interface_catalog(CATALOG_ROOT).to_dict()


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
    instance = device / "instantiations/generic-100ohm-resistor"
    paths = (
        device / "resistor.pmdl",
        instance / "static.part",
        instance / "v1.model",
    )
    return {
        "summary": "modeled the authoritative target",
        "artifacts": [
            {
                "path": path.as_posix(),
                "content": (CATALOG_ROOT / path).read_text(encoding="utf-8"),
            }
            for path in paths
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
            CATALOG_ROOT / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl",
            CATALOG_ROOT / "mechanical" / "inert_objects" / "planar_rigid_bodies" / "rigid_body_planar.pmdl",
        ),
        interfaces=interface_paths(CATALOG_ROOT),
        direct_hierarchy=(CATALOG_ROOT / "electromechanical" / "motors" / "brushed_dc_motors" / "dc_motor.pmdl",),
        component_information=(
            PROJECT_ROOT
            / "examples"
            / "scanner_robot"
            / "component_inputs"
            / "romi_drive.json"
        ),
    )


class BudgetTests(unittest.TestCase):
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

    def test_actual_agent_cli_defaults_to_one_staged_romi_drive_model(self):
        args = build_parser().parse_args(["agent-run", "modeling-one"])
        self.assertEqual(args.agent_job, "modeling-one")
        self.assertEqual(args.target, "romi_drive")
        self.assertFalse(args.force)

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
                path="electrical/resistors/fixed_resistors/resistor.pmdl",
                content=(
                    CATALOG_ROOT / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl"
                ).read_text(encoding="utf-8"),
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
                second = run_modeling_proposal(
                    agent, inputs, "romi_drive", output
                )
                staged_model = (
                    Path(first["staging_artifacts"])
                    / "electrical"
                    / "resistors"
                    / "fixed_resistors"
                    / "resistor.pmdl"
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
            self.assertTrue((artifacts / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").is_file())
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


class ModelingValidationToolTests(unittest.TestCase):
    def test_drafts_validate_iteratively_with_detailed_feedback_and_call_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = ModelingAgent(BudgetLedger(root / "ledger.json", 1.0), root / "staging")
            workspace = agent.prepare_workspace(_real_modeling_inputs(), "validator-run")

            self.assertFalse((workspace / "inputs").exists())
            self.assertTrue((workspace.parent / "inputs").is_dir())
            integrity = assert_workspace_integrity(workspace)
            self.assertGreaterEqual(len(integrity["checked"]), 8)
            instructions = (workspace / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn(
                "model_catalog/electrical/resistors/fixed_resistors/resistor.pmdl",
                instructions,
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
                (CATALOG_ROOT / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").read_bytes()
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
                (CATALOG_ROOT / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").read_bytes()
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
                (CATALOG_ROOT / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").read_bytes()
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
    def test_promotion_uses_validated_atomic_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposed = root / "proposed" / "electrical" / "resistors" / "fixed_resistors"
            proposed.mkdir(parents=True)
            source = proposed / "resistor.pmdl"
            source.write_bytes(
                (CATALOG_ROOT / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").read_bytes()
            )
            registry = root / "registry"

            written = ModelingAgent.promote(root / "proposed", registry)

            target = registry / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl"
            self.assertEqual(written, [target])
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertFalse(any(registry.rglob("*.tmp")))

    def test_promotion_revalidates_after_staged_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposed = root / "proposed"
            model = proposed / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl"
            model.parent.mkdir(parents=True)
            model.write_bytes(
                (CATALOG_ROOT / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").read_bytes()
            )
            ModelingAgent.validate_artifacts(proposed)
            model.write_text('{"format":"pmdl-1","tampered":true}\n', encoding="utf-8")

            registry = root / "registry"
            with self.assertRaises(ValueError):
                ModelingAgent.promote(proposed, registry)
            self.assertFalse(
                (registry / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").exists()
            )

    def test_canary_contract_requires_static_part_and_model_instance(self):
        with self.assertRaisesRegex(ValueError, "static.part"):
            ModelingAgent._validate_canary_value(
                {
                    **_modeling_value(),
                    "artifacts": [
                        {"path": "electrical/resistors/fixed_resistors/resistor.pmdl", "content": "{}"},
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

    def test_nonzero_run_recovers_but_charges_full_reservation(self):
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
                component_information=Path("authoritative-target.json"),
            )
            expected_charge = pricing.worst_case(
                max_input_tokens=2_000, max_output_tokens=1_000
            )

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
                    / "generic-100ohm-resistor"
                    / "v1.model"
                ).is_file()
            )
            self.assertAlmostEqual(charged, expected_charge)
            event = ledger.snapshot()["events"][-1]
            self.assertEqual(event["status"], "recovered_after_nonzero_exit")
            self.assertIsNone(event["usage"])
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
                component_information=Path("authoritative-target.json"),
            )

            with mock.patch.object(
                agent, "prepare_workspace", return_value=workspace
            ), mock.patch(
                "contraption.part_import.agents.subprocess.run", return_value=process
            ):
                artifacts, recovered, _charged = agent.run(inputs, canary=True)

            self.assertIn("Recovered complete candidate bundle", recovered["summary"])
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
                    / "generic-100ohm-resistor"
                    / "v1.model"
                ).is_file()
            )
            self.assertEqual(
                ledger.snapshot()["events"][-1]["status"],
                "recovered_after_nonzero_exit",
            )


if __name__ == "__main__":
    unittest.main()
