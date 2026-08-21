from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

from contraption.catalog.interfaces import interface_paths, load_interface_catalog
from contraption.catalog.instantiations import (
    ModelInstantiationSpec,
    PROCUREMENT_METADATA_FIELDS,
)
from contraption.cli import (
    _ingestion_ledger_binding,
    _replay_direct_child,
    build_parser,
    command_agent_run,
)
from contraption.part_import.agents import (
    AgentLimits,
    ClassificationAgent,
    DirectResponsesModelingAgent,
    ModelingInputs,
    _usage_from_response,
    modeling_preflight,
)
from contraption.part_import.budget import (
    BudgetExceeded,
    BudgetLedger,
    TokenPricing,
    Usage,
)
from contraption.part_import.ingestion import (
    IngestionPolicy,
    combine_ingestion_metrics,
    combine_ingestion_metrics_with_carryovers,
    complete_batch_targets,
    failed_run_carryover,
    ingestion_metrics,
    prepare_isolated_replay,
    replay_state_fingerprint,
    run_part_ingestion,
    validate_canary_gate,
    validate_canary_report_evidence,
    validate_failed_run_carryovers,
    validate_matching_failed_run_carryovers,
    validate_replay_state,
    workflow_fingerprint,
)
from contraption.physics.dsl import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / "model_catalog"


class _InputTokenCounter:
    def __init__(self, count: int):
        self.value = count
        self.requests: list[dict] = []

    def count(self, **kwargs):
        self.requests.append(kwargs)
        return types.SimpleNamespace(input_tokens=self.value)


class _UsageDetails:
    def __init__(self, cached: int = 0, cache_write: int | None = 0):
        self.cached_tokens = cached
        self.cache_write_tokens = cache_write


class _ProviderUsage:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached: int = 0,
        cache_write: int | None = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = _UsageDetails(cached, cache_write)


class _Responses:
    def __init__(self, values: list[dict], *, counted_input: int, usages: list[object]):
        self.values = list(values)
        self.usages = list(usages)
        self.input_tokens = _InputTokenCounter(counted_input)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return types.SimpleNamespace(
            output_text=json.dumps(self.values.pop(0), sort_keys=True),
            usage=self.usages.pop(0),
        )


class _Client:
    def __init__(self, values: list[dict], *, counted_input: int, usages: list[object]):
        self.responses = _Responses(
            values, counted_input=counted_input, usages=usages
        )


class _ForbiddenProviderClient:
    @property
    def responses(self):
        raise AssertionError("deterministic recipe touched a provider client")


def _classification_value() -> dict:
    return {
        "canonical_name": "Direct fixture resistor",
        "domains": ["electrical"],
        "reuse_path": ["resistor", "fixed-resistor"],
        "new_nodes": [],
        "category": "resistor",
        "device": "fixed-resistor",
        "rationale": "The existing fixed-resistor interface fits.",
        "uncertainties": [],
    }


def _new_model_bundle(root: Path, target: str) -> dict:
    owner = Path("electrical/resistors/fixed_resistors")
    pmdl_path = owner / "direct_fixture_resistor.pmdl"
    instance = owner / "instantiations" / target

    pmdl = json.loads(
        (CATALOG / "electrical/resistors/resistor.pmdl").read_text(encoding="utf-8")
    )
    pmdl["id"] = "electrical.resistor.direct_fixture"
    pmdl["name"] = "Direct fixture resistor"
    pmdl["implements"] = "fixed-resistor"
    temporary_pmdl = root / "direct_fixture_resistor.pmdl"
    temporary_pmdl.write_text(
        json.dumps(pmdl, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    canonical = load_model(temporary_pmdl)
    model_sha = "sha256:" + hashlib.sha256(
        canonical.to_json().encode("utf-8")
    ).hexdigest()

    source_instance = (
        CATALOG
        / "electrical/resistors/fixed_resistors/instantiations"
        / "generic-100ohm-resistor"
    )
    static = json.loads((source_instance / "static.part").read_text(encoding="utf-8"))
    static["id"] = target
    static["name"] = "Direct API new-model fixture"
    dimensions = static["bodies"][0]["solids"][0]["geometry"]["dimensions_m"]
    static["bodies"][0]["solids"][0]["geometry"] = {
        "kind": "box",
        "dimensions_m": dimensions,
    }
    model = json.loads((source_instance / "v1.model").read_text(encoding="utf-8"))
    model["id"] = f"{target}.v1"
    model["part"] = target
    model["model"] = {
        "id": canonical.id,
        "version": canonical.version,
        "sha256": model_sha,
    }
    # The source fixture supplies no part-specific tolerance/uncertainty fact.
    # Keep the required instance container empty instead of inventing one.
    model["parameter_uncertainty"] = {}
    value = {
        "summary": "Created one new PMDL and its exact part/model records.",
        "artifacts": [
            {
                "path": pmdl_path.as_posix(),
                "content": temporary_pmdl.read_text(encoding="utf-8"),
            },
            {
                "path": (instance / "static.part").as_posix(),
                "content": json.dumps(static, indent=2, sort_keys=True) + "\n",
            },
            {
                "path": (instance / "v1.model").as_posix(),
                "content": json.dumps(model, indent=2, sort_keys=True) + "\n",
            },
        ],
        "assumptions": [],
        "evidence": ["Host fixture source records."],
    }
    return value


def _direct_fixture(root: Path) -> tuple[Path, Path, ModelingInputs, dict]:
    data_root = root / "data"
    shutil.copytree(CATALOG, data_root / "model_catalog")
    (data_root / "assembled_contraptions").mkdir(parents=True)
    component = root / "direct_new_model_fixture.json"
    component.write_text(
        json.dumps(
            {
                "part_kind": "fixed resistor",
                "purpose": "offline new-model envelope fixture",
                "domains": ["electrical"],
                "published_parameters": {"resistance": 100.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    copied_catalog = data_root / "model_catalog"
    generic = (
        copied_catalog
        / "electrical/resistors/fixed_resistors/instantiations"
        / "generic-100ohm-resistor"
    )
    inputs = ModelingInputs(
        constraints=PROJECT_ROOT / "prompts/model_constraints.md",
        gold_templates=(
            copied_catalog / "electrical/resistors/resistor.pmdl",
            generic / "static.part",
            generic / "v1.model",
        ),
        interfaces=interface_paths(copied_catalog),
        direct_hierarchy=(),
        component_information=component,
    )
    return data_root, copied_catalog, inputs, _new_model_bundle(root, component.stem)


def _existing_model_missing_uncertainty_fixture(
    root: Path,
) -> tuple[Path, Path, ModelingInputs, dict]:
    data_root = root / "data"
    shutil.copytree(CATALOG, data_root / "model_catalog")
    (data_root / "assembled_contraptions").mkdir(parents=True)
    copied_catalog = data_root / "model_catalog"
    target = "direct_existing_model_10r"
    component = root / f"{target}.json"
    component.write_text(
        json.dumps(
            {
                "manufacturer": "Fixture",
                "product": "10R",
                "part_kind": "fixed resistor",
                "purpose": "paid-canary omission regression",
                "domains": ["electrical"],
                "published_parameters": {
                    "resistance_ohm": 10.0,
                    "tolerance_fraction": 0.01,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generic = (
        copied_catalog
        / "electrical/resistors/fixed_resistors/instantiations"
        / "generic-100ohm-resistor"
    )
    resistor = copied_catalog / "electrical/resistors/resistor.pmdl"
    inputs = ModelingInputs(
        constraints=PROJECT_ROOT / "prompts/model_constraints.md",
        gold_templates=(resistor, generic / "static.part", generic / "v1.model"),
        interfaces=interface_paths(copied_catalog),
        direct_hierarchy=(resistor,),
        component_information=component,
    )
    static = json.loads((generic / "static.part").read_text(encoding="utf-8"))
    static["id"] = target
    static["name"] = "Direct existing-model omission fixture"
    # Exact reproduction of the second paid-canary failure: identity was
    # copied into static metadata. Include recursive/mixed-case and top-level
    # variants so the regression mirrors catalog admission semantics.
    static["manufacturer"] = "Fixture"
    static["metadata"].update(
        {
            "manufacturer": "Fixture",
            "nested": {
                "ProDuCt": "10R",
                "retained_fact": "not procurement metadata",
            },
            "list_value": [
                {"VeNdOr_SkU": "fixture-10r", "retained_number": 10}
            ],
        }
    )
    dimensions = static["bodies"][0]["solids"][0]["geometry"]["dimensions_m"]
    static["bodies"][0]["solids"][0]["geometry"] = {
        "kind": "box",
        "dimensions_m": dimensions,
    }
    model = json.loads((generic / "v1.model").read_text(encoding="utf-8"))
    model["id"] = f"{target}.v1"
    model["part"] = target
    model["parameters"] = {"resistance": 10.0}
    model["Supplier"] = "Fixture distributor"
    model["metadata"].update(
        {
            "MPN": "10R",
            "nested": {"Purchase_URL": "https://invalid.example/offer"},
            "retained_model_fact": "ideal resistor hypothesis",
        }
    )
    del model["parameter_uncertainty"]
    instance = Path(
        f"electrical/resistors/fixed_resistors/instantiations/{target}"
    )
    bundle = {
        "summary": "Reuse the exact resistor PMDL for the fixture.",
        "artifacts": [
            {
                "path": (instance / "static.part").as_posix(),
                "content": json.dumps(static, indent=2, sort_keys=True) + "\n",
            },
            {
                "path": (instance / "v1.model").as_posix(),
                "content": json.dumps(model, indent=2, sort_keys=True) + "\n",
            },
        ],
        "assumptions": [],
        "evidence": ["Offline regression matching the paid canary omission."],
    }
    return data_root, copied_catalog, inputs, bundle


class PricingAndScopeTests(unittest.TestCase):
    def test_new_zero_event_ledger_is_persisted_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent-budget.json"
            self.assertFalse(path.exists())
            ledger = BudgetLedger(path, 0.50)
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {
                    "schema_version": 1,
                    "limit_usd": 0.50,
                    "spent_usd": 0.0,
                    "reserved": {},
                    "events": [],
                },
            )
            self.assertEqual(ledger.snapshot()["events"], [])
            original_bytes = path.read_bytes()
            lowered = BudgetLedger(path, 0.25)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(lowered.snapshot()["limit_usd"], 0.25)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertFalse(path.with_suffix(".json.lock").exists())
            self.assertEqual(tuple(path.parent.glob(f".{path.name}.*.tmp")), ())

    def test_workflow_fingerprint_binds_host_implementation(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = BudgetLedger(Path(temporary) / "ledger.json", 1.0)
            classifier = ClassificationAgent(ledger)
            modeler = DirectResponsesModelingAgent(
                ledger, Path(temporary) / "staging"
            )
            interfaces = load_interface_catalog(CATALOG).to_dict()
            with mock.patch(
                "contraption.part_import.ingestion.importer_implementation_fingerprint",
                return_value="sha256:aaa",
            ):
                first = workflow_fingerprint(classifier, modeler, interfaces)
            with mock.patch(
                "contraption.part_import.ingestion.importer_implementation_fingerprint",
                return_value="sha256:bbb",
            ):
                second = workflow_fingerprint(classifier, modeler, interfaces)
            self.assertNotEqual(first, second)

    def test_nonfinite_budget_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    BudgetLedger(path, value)
            ledger = BudgetLedger(path, 1.0)
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    ledger.reserve("bad", value, {})
                with self.assertRaises(ValueError):
                    ledger.reserve(
                        "bad-scope",
                        0.01,
                        {},
                        cost_scope="part",
                        cost_scope_limit_usd=value,
                    )
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                TokenPricing(input_per_million=float("nan"))

    def test_current_luna_pricing_and_strict_short_context_threshold(self):
        pricing = TokenPricing()
        reported = Usage(100_000, 20_000, 10_000, cache_write_input_tokens=5_000)
        expected = (75_000 * 0.20 + 20_000 * 0.02 + 5_000 * 0.25 + 10_000 * 1.20) / 1_000_000
        self.assertAlmostEqual(pricing.cost(reported), expected)
        unknown = Usage(100_000, 20_000, 10_000)
        self.assertGreater(pricing.cost(unknown), pricing.cost(reported))
        self.assertAlmostEqual(
            pricing.cost(Usage(272_000, 0, 0, cache_write_input_tokens=0)),
            272_000 * 0.20 / 1_000_000,
        )
        self.assertAlmostEqual(
            pricing.cost(Usage(272_001, 0, 0, cache_write_input_tokens=0)),
            272_001 * 0.40 / 1_000_000,
        )

    def test_three_attempt_new_model_envelope_is_strictly_below_five_cents(self):
        pricing = TokenPricing()
        classification = pricing.worst_case(
            max_input_tokens=12_000, max_output_tokens=2_000
        )
        modeling = pricing.worst_case(
            max_input_tokens=20_000, max_output_tokens=8_000
        )
        self.assertAlmostEqual(classification + 3 * modeling, 0.0492)
        self.assertLess(classification + 3 * modeling, 0.05)

    def test_scope_is_atomic_and_usage_overage_is_never_undercharged(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = BudgetLedger(Path(temporary) / "ledger.json", 1.0)
            ledger.reserve(
                "a", 0.02, {"kind": "modeling"},
                cost_scope="part", cost_scope_limit_usd=0.05,
            )
            ledger.reserve(
                "b", 0.02, {"kind": "modeling"},
                cost_scope="part", cost_scope_limit_usd=0.05,
            )
            with self.assertRaises(BudgetExceeded):
                ledger.reserve(
                    "c", 0.01, {"kind": "modeling"},
                    cost_scope="part", cost_scope_limit_usd=0.05,
                )
            charged = ledger.settle(
                "a", usage=Usage(120_000, 0, 0), pricing=TokenPricing()
            )
            self.assertAlmostEqual(charged, 0.03)
            event = ledger.snapshot()["events"][-1]
            self.assertEqual(event["status"], "usage_exceeded_reservation")
            self.assertAlmostEqual(event["scope_total_usd"], 0.05)
            self.assertTrue(event["scope_limit_breached"])

    def test_missing_and_malformed_usage_are_unavailable_not_free(self):
        self.assertIsNone(_usage_from_response(types.SimpleNamespace(usage=None)))
        self.assertIsNone(
            _usage_from_response(
                types.SimpleNamespace(
                    usage=types.SimpleNamespace(input_tokens=100)
                )
            )
        )
        self.assertIsNone(
            _usage_from_response(
                types.SimpleNamespace(
                    usage=types.SimpleNamespace(
                        input_tokens=True,
                        output_tokens=3,
                        input_tokens_details=None,
                    )
                )
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            ledger = BudgetLedger(Path(temporary) / "ledger.json", 1.0)
            client = _Client(
                [_classification_value()], counted_input=500, usages=[None]
            )
            _value, usage, charged = ClassificationAgent(ledger).classify(
                {"name": "fixture"},
                load_interface_catalog(CATALOG).to_dict(),
                client=client,
            )
            self.assertIsNone(usage)
            expected = TokenPricing().worst_case(
                max_input_tokens=500, max_output_tokens=2_000
            )
            self.assertAlmostEqual(charged, expected)
            self.assertEqual(
                ledger.snapshot()["events"][-1]["cost_basis"],
                "full_reservation_conservative",
            )


class DirectIngestionTests(unittest.TestCase):
    def test_ten_yageo_0603_parts_use_validated_zero_provider_dispatch_recipe(self):
        source_jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        raw_jobs = json.loads(source_jobs.read_text(encoding="utf-8"))["jobs"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = prepare_isolated_replay(
                source_jobs, root / "isolated-data-root"
            )
            data_root = Path(manifest["data_root"])
            copied_catalog = data_root / "model_catalog"
            component_root = Path(manifest["isolated_job_file"]).parent
            resistor = copied_catalog / "electrical/resistors/resistor.pmdl"
            generic = (
                copied_catalog
                / "electrical/resistors/fixed_resistors/instantiations"
                / "generic-100ohm-resistor"
            )
            gold = (resistor, generic / "static.part", generic / "v1.model")
            interfaces = interface_paths(copied_catalog)
            eligible_jobs = []
            deferred_jobs = []
            for job in raw_jobs:
                component = component_root / job["component_information"]
                if modeling_preflight(component)["eligible"]:
                    eligible_jobs.append((job, component))
                else:
                    deferred_jobs.append((job, component))
            self.assertEqual(len(eligible_jobs), 10)
            self.assertEqual(len(deferred_jobs), 10)

            ledger = BudgetLedger(root / "ledger.json", 1.0)
            classifier = ClassificationAgent(ledger)
            modeler = DirectResponsesModelingAgent(ledger, root / "staging")
            forbidden_provider_client = _ForbiddenProviderClient()
            results = []
            with mock.patch.dict(
                os.environ, {"CONTRAPTION_DATA_ROOT": str(data_root)}, clear=False
            ):
                for job, component in eligible_jobs:
                    inputs = ModelingInputs(
                        constraints=data_root / "prompts/model_constraints.md",
                        gold_templates=gold,
                        interfaces=interfaces,
                        direct_hierarchy=(resistor,),
                        component_information=component,
                    )
                    recipe = modeler.deterministic_recipe_for(inputs)
                    self.assertIsNotNone(recipe)
                    self.assertEqual(
                        recipe["schema"],
                        "contraption.host-recipe.rectangular-chip-resistor/v1",
                    )
                    result = run_part_ingestion(
                        classifier,
                        modeler,
                        inputs,
                        job["id"],
                        copied_catalog,
                        root / "proposals",
                        ingestion_run_id="deterministic-ten-part-replay",
                        canary=job["id"] == "yageo_rc0603_10r",
                        classification_client=forbidden_provider_client,
                        modeling_client=forbidden_provider_client,
                    )
                    results.append(result)
                    self.assertTrue(result["fully_ingested"])
                    self.assertEqual(result["classification"]["charged_usd"], 0.0)
                    self.assertEqual(result["classification"]["provider_calls"], 0)
                    self.assertEqual(
                        result["classification"]["generation_mode"],
                        "host_deterministic",
                    )
                    self.assertEqual(result["modeling"]["charged_usd"], 0.0)
                    self.assertIsNone(result["modeling"]["usage"])
                    self.assertEqual(result["modeling"]["provider_calls"], 0)
                    self.assertEqual(
                        result["modeling"]["generation_mode"],
                        "host_deterministic",
                    )
                    activity = result["validation_activity"]
                    self.assertEqual(activity["logged_calls"], 0)
                    self.assertEqual(activity["failed_calls"], 0)
                    self.assertEqual(activity["provider_calls"], 0)
                    self.assertEqual(activity["deterministic_validation"], "passed")
                    self.assertEqual(
                        activity["host_generation"]["recipe_id"],
                        "rectangular-chip-resistor-0603-ideal-v1",
                    )
                    self.assertEqual(
                        result["classification"]["recipe_sha256"],
                        recipe["recipe_sha256"],
                    )
                    self.assertEqual(
                        activity["host_generation"]["recipe_sha256"],
                        recipe["recipe_sha256"],
                    )
                    self.assertEqual(
                        result["host_recipe_sha256"], recipe["recipe_sha256"]
                    )
                    self.assertRegex(
                        activity["import_plan_sha256"], r"^sha256:[0-9a-f]{64}$"
                    )
                    self.assertRegex(
                        result["classification"]["input_hash"],
                        r"^sha256:[0-9a-f]{64}$",
                    )
                    self.assertRegex(
                        result["modeling"]["input_hash"],
                        r"^sha256:[0-9a-f]{64}$",
                    )

                    instance = (
                        copied_catalog
                        / "electrical/resistors/fixed_resistors/instantiations"
                        / job["id"]
                    )
                    static = json.loads(
                        (instance / "static.part").read_text(encoding="utf-8")
                    )
                    model = json.loads(
                        (instance / "v1.model").read_text(encoding="utf-8")
                    )
                    component_value = json.loads(component.read_text(encoding="utf-8"))
                    published = component_value["published_parameters"]
                    geometry = static["bodies"][0]["solids"][0]["geometry"]
                    self.assertEqual(geometry["kind"], "box")
                    self.assertEqual(
                        geometry["dimensions_m"], published["dimensions_m"]
                    )
                    self.assertIsNone(geometry["shape_uri"])
                    self.assertIsNone(geometry["shape_sha256"])
                    self.assertIsNone(geometry["surface_id"])
                    connectors = {item["id"]: item for item in static["connectors"]}
                    self.assertEqual(set(connectors), {"p", "n"})
                    half_x = published["dimensions_m"][0] / 2.0
                    self.assertEqual(
                        connectors["p"]["local_pose"]["translation_m"],
                        [-half_x, 0.0, 0.0],
                    )
                    self.assertEqual(
                        connectors["n"]["local_pose"]["translation_m"],
                        [half_x, 0.0, 0.0],
                    )
                    for connector in connectors.values():
                        self.assertEqual(connector["provenance"]["kind"], "estimated")
                        self.assertIn(
                            "terminal geometry and land pattern are unavailable",
                            connector["provenance"]["source"],
                        )
                        self.assertEqual(
                            connector["fabrication"],
                            {
                                "kind": "electrical_termination",
                                "missing": ["conductor", "termination"],
                                "status": "missing",
                            },
                        )
                    self.assertNotIn("manufacturer", static["metadata"])
                    self.assertNotIn("product", static["metadata"])
                    self.assertNotIn("http", json.dumps(static).casefold())
                    self.assertNotIn(
                        component_value["product"].casefold(),
                        json.dumps(static).casefold(),
                    )
                    self.assertEqual(
                        model["model"],
                        {
                            "id": "electrical.resistor.ideal",
                            "version": "1.0.0",
                            "sha256": recipe["model"]["sha256"],
                        },
                    )
                    nominal = float(published["resistance_ohm"])
                    tolerance = float(published["tolerance_fraction"])
                    self.assertEqual(model["parameters"], {"resistance": nominal})
                    uncertainty = model["parameter_uncertainty"]["resistance"]
                    self.assertEqual(uncertainty["distribution"], "uniform")
                    self.assertAlmostEqual(
                        uncertainty["parameters"]["lower"],
                        nominal * (1.0 - tolerance),
                    )
                    self.assertAlmostEqual(
                        uncertainty["parameters"]["upper"],
                        nominal * (1.0 + tolerance),
                    )
                    procurement = tuple(instance.glob("*.procurement"))
                    self.assertEqual(len(procurement), 1)
                    procurement_text = procurement[0].read_text(encoding="utf-8")
                    self.assertIn(component_value["product"], procurement_text)
                    self.assertIn('"manufacturer": "Yageo"', procurement_text)

                for job, component in deferred_jobs:
                    inputs = ModelingInputs(
                        constraints=data_root / "prompts/model_constraints.md",
                        gold_templates=gold,
                        interfaces=interfaces,
                        direct_hierarchy=(),
                        component_information=component,
                    )
                    self.assertIsNone(modeler.deterministic_recipe_for(inputs))

            events = ledger.snapshot()["events"]
            self.assertEqual(events, [])
            self.assertTrue(all(result["fully_ingested"] for result in results))
            metrics = ingestion_metrics(
                results,
                ledger.snapshot(),
                "deterministic-ten-part-replay",
                expected_target_count=10,
            )
            self.assertTrue(metrics["passed"])
            self.assertEqual(metrics["provider_calls"], 0)
            self.assertEqual(
                metrics["completion_generation_modes"], {"host_deterministic": 10}
            )

            mismatch_job, mismatch_component = eligible_jobs[0]
            mismatch_inputs = ModelingInputs(
                constraints=data_root / "prompts/model_constraints.md",
                gold_templates=gold,
                interfaces=interfaces,
                direct_hierarchy=(resistor,),
                component_information=mismatch_component,
            )
            classification_recipe = "sha256:" + "a" * 64
            modeling_recipe = "sha256:" + "b" * 64
            classification_result = {
                "target": mismatch_job["id"],
                "status": "completed",
                "charged_usd": 0.0,
                "generation_mode": "host_deterministic",
                "provider_calls": 0,
                "recipe_sha256": classification_recipe,
            }
            modeling_result = {
                "target": mismatch_job["id"],
                "status": "completed",
                "charged_usd": 0.0,
                "generation_mode": "host_deterministic",
                "provider_calls": 0,
                "validation_activity": {
                    "host_generation": {"recipe_sha256": modeling_recipe}
                },
            }
            with (
                mock.patch.dict(
                    os.environ,
                    {"CONTRAPTION_DATA_ROOT": str(data_root)},
                    clear=False,
                ),
                mock.patch(
                    "contraption.part_import.ingestion._host_recipe_classification",
                    return_value=classification_result,
                ),
                mock.patch(
                    "contraption.part_import.ingestion.run_modeling_proposal",
                    return_value=modeling_result,
                ),
                mock.patch.object(modeler, "promote") as promote,
                self.assertRaisesRegex(ValueError, "recipe digest mismatch"),
            ):
                run_part_ingestion(
                    classifier,
                    modeler,
                    mismatch_inputs,
                    mismatch_job["id"],
                    copied_catalog,
                    root / "mismatch-proposals",
                    ingestion_run_id="deterministic-recipe-mismatch",
                    classification_client=forbidden_provider_client,
                    modeling_client=forbidden_provider_client,
                )
            promote.assert_not_called()

            bad_component = root / "yageo_bad_recipe.json"
            bad_value = json.loads(eligible_jobs[0][1].read_text(encoding="utf-8"))
            del bad_value["published_parameters"]["dimensions_m"]
            bad_component.write_text(json.dumps(bad_value) + "\n", encoding="utf-8")
            bad_inputs = ModelingInputs(
                constraints=data_root / "prompts/model_constraints.md",
                gold_templates=gold,
                interfaces=interfaces,
                direct_hierarchy=(resistor,),
                component_information=bad_component,
            )
            before_events = len(ledger.snapshot()["events"])
            with (
                mock.patch.dict(
                    os.environ,
                    {"CONTRAPTION_DATA_ROOT": str(data_root)},
                    clear=False,
                ),
                self.assertRaisesRegex(ValueError, "requires three dimensions_m"),
            ):
                run_part_ingestion(
                    classifier,
                    modeler,
                    bad_inputs,
                    bad_component.stem,
                    copied_catalog,
                    root / "bad-proposals",
                    ingestion_run_id="deterministic-invalid-evidence",
                    classification_client=forbidden_provider_client,
                    modeling_client=forbidden_provider_client,
                )
            self.assertEqual(len(ledger.snapshot()["events"]), before_events)

    def test_paid_canary_missing_uncertainty_is_host_normalized_on_first_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, copied_catalog, inputs, bundle = (
                _existing_model_missing_uncertainty_fixture(root)
            )
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            classifier = ClassificationAgent(ledger)
            modeler = DirectResponsesModelingAgent(ledger, root / "staging")
            classification_client = _Client(
                [_classification_value()],
                counted_input=800,
                usages=[_ProviderUsage(800, 120)],
            )
            modeling_client = _Client(
                [bundle],
                counted_input=5_000,
                usages=[_ProviderUsage(5_000, 1_100)],
            )
            with mock.patch.dict(
                os.environ, {"CONTRAPTION_DATA_ROOT": str(data_root)}, clear=False
            ):
                result = run_part_ingestion(
                    classifier,
                    modeler,
                    inputs,
                    Path(inputs.component_information).stem,
                    copied_catalog,
                    root / "proposals",
                    ingestion_run_id="missing-uncertainty-regression",
                    canary=True,
                    prior_target_charged_usd=0.00750249,
                    classification_client=classification_client,
                    modeling_client=modeling_client,
                )
            self.assertTrue(result["fully_ingested"])
            self.assertEqual(len(modeling_client.responses.requests), 1)
            activity = result["validation_activity"]
            self.assertEqual(activity["failed_calls"], 0)
            self.assertFalse(activity["raw_successful_response_complete"])
            self.assertEqual(
                result["modeling"]["host_normalizations"],
                [
                    {
                        "action": "derive_uniform_parameter_uncertainty_from_tolerance",
                        "artifact_path": (
                            "electrical/resistors/fixed_resistors/instantiations/"
                            "direct_existing_model_10r/v1.model"
                        ),
                        "attempt": 1,
                        "lower": 9.9,
                        "parameter": "resistance",
                        "raw_response_complete": False,
                        "source_field": (
                            "published_parameter_facts.tolerance_fraction"
                        ),
                        "source_value": 0.01,
                        "upper": 10.1,
                    },
                    {
                        "action": "remove_host_owned_procurement_identity_fields",
                        "artifact_path": (
                            "electrical/resistors/fixed_resistors/instantiations/"
                            "direct_existing_model_10r/static.part"
                        ),
                        "attempt": 1,
                        "destination": "adjacent_host_procurement_record",
                        "raw_response_complete": False,
                        "removed_paths": [
                            "static_part.manufacturer",
                            "static_part.metadata.list_value[0].VeNdOr_SkU",
                            "static_part.metadata.manufacturer",
                            "static_part.metadata.nested.ProDuCt",
                        ],
                    },
                    {
                        "action": "remove_host_owned_procurement_identity_fields",
                        "artifact_path": (
                            "electrical/resistors/fixed_resistors/instantiations/"
                            "direct_existing_model_10r/v1.model"
                        ),
                        "attempt": 1,
                        "destination": "adjacent_host_procurement_record",
                        "raw_response_complete": False,
                        "removed_paths": [
                            "model_instance.Supplier",
                            "model_instance.metadata.MPN",
                            "model_instance.metadata.nested.Purchase_URL",
                        ],
                    },
                ],
            )
            promoted_static = (
                copied_catalog
                / "electrical/resistors/fixed_resistors/instantiations"
                / "direct_existing_model_10r/static.part"
            )
            promoted_model = (
                copied_catalog
                / "electrical/resistors/fixed_resistors/instantiations"
                / "direct_existing_model_10r/v1.model"
            )
            normalized_static = json.loads(
                promoted_static.read_text(encoding="utf-8")
            )
            normalized = json.loads(promoted_model.read_text(encoding="utf-8"))
            self.assertNotIn("manufacturer", normalized_static)
            self.assertNotIn("Supplier", normalized)
            self.assertEqual(
                normalized_static["metadata"]["nested"]["retained_fact"],
                "not procurement metadata",
            )
            self.assertEqual(
                normalized_static["metadata"]["list_value"][0]["retained_number"],
                10,
            )
            self.assertEqual(
                normalized["metadata"]["retained_model_fact"],
                "ideal resistor hypothesis",
            )

            def reserved_keys(value):
                found = []
                if isinstance(value, dict):
                    for key, nested_value in value.items():
                        if key.casefold() in PROCUREMENT_METADATA_FIELDS:
                            found.append(key)
                        found.extend(reserved_keys(nested_value))
                elif isinstance(value, list):
                    for nested_value in value:
                        found.extend(reserved_keys(nested_value))
                return found

            self.assertEqual(reserved_keys(normalized_static["metadata"]), [])
            self.assertEqual(reserved_keys(normalized["metadata"]), [])
            procurement_records = tuple(promoted_static.parent.glob("*.procurement"))
            self.assertEqual(len(procurement_records), 1)
            procurement_text = procurement_records[0].read_text(encoding="utf-8")
            self.assertIn('"manufacturer": "Fixture"', procurement_text)
            self.assertIn('"value": "10R"', procurement_text)
            self.assertEqual(
                normalized["parameter_uncertainty"],
                {
                    "resistance": {
                        "distribution": "uniform",
                        "parameters": {"lower": 9.9, "upper": 10.1},
                    }
                },
            )
            initialized = ModelInstantiationSpec.from_dict(
                normalized
            ).initialized_parameters()["resistance"]
            self.assertEqual(initialized["uncertainty"]["distribution"], "uniform")
            self.assertEqual(
                initialized["uncertainty"]["parameters"],
                {"lower": 9.9, "upper": 10.1},
            )
            self.assertAlmostEqual(
                result["remaining_part_scope_limit_usd"], 0.04249751
            )
            self.assertLess(result["cumulative_target_charged_usd"], 0.05)
            for event in ledger.snapshot()["events"]:
                self.assertAlmostEqual(
                    event["metadata"]["cost_scope_limit_usd"], 0.04249751
                )
            plan = json.loads(
                next((root / "staging").glob("*/workspace/IMPORT_PLAN.json"))
                .read_text(encoding="utf-8")
            )
            required = plan["artifact_contracts"]["v1.model"][
                "required_top_level_fields"
            ]
            self.assertEqual(len(required), 10)
            self.assertIn("parameter_uncertainty", required)
            policy = plan["artifact_contracts"]["v1.model"][
                "uncertainty_policy"
            ]
            self.assertFalse(policy["empty_object_allowed"])
            self.assertEqual(policy["tolerance_fraction"], 0.01)

            empty_bundle = json.loads(json.dumps(bundle))
            empty_model = json.loads(empty_bundle["artifacts"][1]["content"])
            empty_model["parameter_uncertainty"] = {}
            empty_bundle["artifacts"][1]["content"] = json.dumps(empty_model)
            normalized_empty, empty_actions = (
                modeler._enforce_model_instance_uncertainty_policy(
                    empty_bundle, plan
                )
            )
            self.assertEqual(len(empty_actions), 1)
            empty_result = json.loads(
                normalized_empty["artifacts"][1]["content"]
            )
            self.assertEqual(
                empty_result["parameter_uncertainty"]["resistance"][
                    "distribution"
                ],
                "uniform",
            )

            conflicting = json.loads(json.dumps(empty_bundle))
            conflicting_model = json.loads(conflicting["artifacts"][1]["content"])
            conflicting_model["parameter_uncertainty"] = {
                "resistance": {"distribution": "normal", "std": 5.0}
            }
            conflicting["artifacts"][1]["content"] = json.dumps(
                conflicting_model
            )
            with self.assertRaisesRegex(ValueError, "conflicts"):
                modeler._enforce_model_instance_uncertainty_policy(
                    conflicting, plan
                )

            negative = json.loads(json.dumps(empty_bundle))
            negative_model = json.loads(negative["artifacts"][1]["content"])
            negative_model["parameters"] = {"resistance": -10.0}
            negative["artifacts"][1]["content"] = json.dumps(negative_model)
            normalized_negative, _negative_actions = (
                modeler._enforce_model_instance_uncertainty_policy(
                    negative, plan
                )
            )
            negative_result = json.loads(
                normalized_negative["artifacts"][1]["content"]
            )
            self.assertEqual(
                negative_result["parameter_uncertainty"]["resistance"][
                    "parameters"
                ],
                {"lower": -10.1, "upper": -9.9},
            )

            zero_plan = json.loads(json.dumps(plan))
            zero_policy = zero_plan["artifact_contracts"]["v1.model"][
                "uncertainty_policy"
            ]
            zero_policy["tolerance_fraction"] = 0.0
            zero_policy["required_distribution"] = "fixed"
            normalized_zero, zero_actions = (
                modeler._enforce_model_instance_uncertainty_policy(
                    empty_bundle, zero_plan
                )
            )
            zero_result = json.loads(normalized_zero["artifacts"][1]["content"])
            self.assertEqual(
                zero_result["parameter_uncertainty"],
                {
                    "resistance": {
                        "distribution": "fixed",
                        "parameters": {},
                    }
                },
            )
            self.assertEqual(
                zero_actions[0]["action"],
                "derive_fixed_parameter_uncertainty_from_zero_tolerance",
            )

            no_evidence_plan = json.loads(json.dumps(plan))
            no_evidence_policy = no_evidence_plan["artifact_contracts"][
                "v1.model"
            ]["uncertainty_policy"]
            no_evidence_policy.update(
                {
                    "source_field": None,
                    "tolerance_fraction": None,
                    "required_distribution": None,
                    "empty_object_allowed": True,
                }
            )
            normalized_none, no_actions = (
                modeler._enforce_model_instance_uncertainty_policy(
                    empty_bundle, no_evidence_plan
                )
            )
            self.assertEqual(no_actions, ())
            self.assertEqual(
                json.loads(normalized_none["artifacts"][1]["content"])[
                    "parameter_uncertainty"
                ],
                {},
            )
            normalized_missing_none, missing_none_actions = (
                modeler._enforce_model_instance_uncertainty_policy(
                    bundle, no_evidence_plan
                )
            )
            self.assertEqual(
                missing_none_actions[0]["action"],
                "insert_empty_required_parameter_uncertainty",
            )
            self.assertEqual(
                json.loads(
                    normalized_missing_none["artifacts"][1]["content"]
                )["parameter_uncertainty"],
                {},
            )
            invented = json.loads(json.dumps(empty_bundle))
            invented_model = json.loads(invented["artifacts"][1]["content"])
            invented_model["parameter_uncertainty"] = {
                "resistance": {"distribution": "normal", "std": 1.0}
            }
            invented["artifacts"][1]["content"] = json.dumps(invented_model)
            with self.assertRaisesRegex(ValueError, "without a source"):
                modeler._enforce_model_instance_uncertainty_policy(
                    invented, no_evidence_plan
                )
            system = modeling_client.responses.requests[0]["input"][0]["content"]
            self.assertIn("all ten required top-level keys", system)
            self.assertIn("datasheet_url, datasheet_urls", system)
            static_contract = plan["artifact_contracts"]["static.part"]
            self.assertEqual(
                static_contract["forbidden_procurement_identity_fields"],
                sorted(PROCUREMENT_METADATA_FIELDS),
            )
            self.assertEqual(
                plan["artifact_contracts"]["v1.model"][
                    "forbidden_procurement_identity_fields"
                ],
                sorted(PROCUREMENT_METADATA_FIELDS),
            )

            oversized = json.loads(json.dumps(bundle))
            oversized["summary"] = "x" * 9_000
            retry = modeler._retry_context(
                oversized,
                ValueError(
                    "PhysicalSpecError: missing model instance field(s): "
                    "parameter_uncertainty"
                ),
            )
            self.assertIn("v1_model_artifacts", retry)
            self.assertIn("model-instance-1", retry)
            self.assertIn("manufacturer, manufacturer_item_number", retry)

    def test_new_pmdl_bundle_is_host_validated_promoted_and_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, copied_catalog, inputs, bundle = _direct_fixture(root)
            component = Path(inputs.component_information)
            serialized = json.dumps(bundle, sort_keys=True)
            # This includes JSON escaping around all three complete files and
            # remains comfortably below the 8k-token response allowance for a
            # structured ASCII payload (the live call is provider-capped).
            self.assertLess(len(serialized.encode("utf-8")), 32_000)
            classification_client = _Client(
                [_classification_value()],
                counted_input=1_000,
                usages=[_ProviderUsage(1_000, 1_607)],
            )
            modeling_client = _Client(
                [bundle, bundle, bundle],
                counted_input=9_000,
                usages=[
                    _ProviderUsage(9_000, 5_900),
                    _ProviderUsage(9_000, 5_900),
                    _ProviderUsage(9_000, 5_900),
                ],
            )
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            classifier = ClassificationAgent(ledger)
            modeler = DirectResponsesModelingAgent(ledger, root / "staging")
            with mock.patch.dict(
                os.environ, {"CONTRAPTION_DATA_ROOT": str(data_root)}, clear=False
            ):
                result = run_part_ingestion(
                    classifier,
                    modeler,
                    inputs,
                    component.stem,
                    copied_catalog,
                    root / "proposals",
                    ingestion_run_id="new-model-fixture",
                    canary=True,
                    classification_client=classification_client,
                    modeling_client=modeling_client,
                )
            self.assertTrue(result["fully_ingested"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["validation_activity"]["failed_calls"], 0)
            self.assertTrue(result["promoted_paths"])
            promoted = (
                copied_catalog
                / "electrical/resistors/fixed_resistors/instantiations"
                / component.stem
                / "static.part"
            )
            self.assertTrue(promoted.is_file())
            self.assertEqual(
                modeling_client.responses.requests[0]["max_output_tokens"], 8_000
            )
            direct_system = modeling_client.responses.requests[0]["input"][0][
                "content"
            ]
            self.assertIn("complete UTF-8 file string", direct_system)
            self.assertNotIn("content set to null", direct_system)
            self.assertNotIn("--bundle candidate", direct_system)
            import_plan = json.loads(
                next((root / "staging").glob("*/workspace/IMPORT_PLAN.json"))
                .read_text(encoding="utf-8")
            )
            self.assertIsNone(import_plan["recommended_instantiation_root"])
            metrics = ingestion_metrics(
                [result], ledger.snapshot(), "new-model-fixture"
            )
            self.assertTrue(metrics["passed"])
            self.assertLess(metrics["cost_per_fully_ingested_part_usd"], 0.05)

    def test_direct_retry_quarantines_partial_publish_and_charges_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, _catalog, inputs, bundle = _direct_fixture(root)
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            modeler = DirectResponsesModelingAgent(ledger, root / "staging")
            client = _Client(
                [bundle, bundle],
                counted_input=9_000,
                usages=[
                    _ProviderUsage(9_000, 5_900),
                    _ProviderUsage(9_000, 5_900),
                ],
            )
            original = modeler._materialize_artifacts
            calls = 0

            def flaky_materialization(workspace, value, **kwargs):
                nonlocal calls
                calls += 1
                proposed = Path(workspace) / "proposed"
                if calls == 1:
                    proposed.mkdir()
                    (proposed / "stale.pmdl").write_text(
                        "stale\n", encoding="utf-8"
                    )
                    raise ValueError("simulated failure after proposal publication")
                self.assertFalse(proposed.exists())
                return original(workspace, value, **kwargs)

            with (
                mock.patch.dict(
                    os.environ,
                    {"CONTRAPTION_DATA_ROOT": str(data_root)},
                    clear=False,
                ),
                mock.patch.object(
                    modeler,
                    "_materialize_artifacts",
                    side_effect=flaky_materialization,
                ),
            ):
                artifacts, _proposal, charged = modeler.run(
                    inputs,
                    target=Path(inputs.component_information).stem,
                    ingestion_run_id="retry-run",
                    cost_scope="retry-part",
                    cost_scope_limit_usd=0.05,
                    client=client,
                )
            self.assertEqual(calls, 2)
            self.assertTrue((artifacts / "static.part").exists() or tuple(artifacts.rglob("static.part")))
            run_root = artifacts.parent.parent
            quarantine = run_root / "failed-proposal-attempt-1"
            self.assertEqual(
                (quarantine / "stale.pmdl").read_text(encoding="utf-8"),
                "stale\n",
            )
            activity = json.loads(
                (run_root / "validation-activity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(activity["failed_calls"], 1)
            events = ledger.snapshot()["events"]
            self.assertEqual([event["status"] for event in events], ["invalid_output", "completed"])
            self.assertAlmostEqual(
                charged,
                sum(float(event["charged_usd"]) for event in events),
            )
            self.assertEqual(len(client.responses.input_tokens.requests), 2)
            self.assertLess(charged, 0.05)

    def test_direct_missing_usage_charges_full_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, _catalog, inputs, bundle = _direct_fixture(root)
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            modeler = DirectResponsesModelingAgent(ledger, root / "staging")
            client = _Client([bundle], counted_input=9_000, usages=[None])
            with mock.patch.dict(
                os.environ, {"CONTRAPTION_DATA_ROOT": str(data_root)}, clear=False
            ):
                _artifacts, _proposal, charged = modeler.run(
                    inputs,
                    target=Path(inputs.component_information).stem,
                    cost_scope="missing-usage-part",
                    cost_scope_limit_usd=0.05,
                    client=client,
                )
            expected = TokenPricing().worst_case(
                max_input_tokens=9_000, max_output_tokens=8_000
            )
            self.assertAlmostEqual(charged, expected)
            self.assertEqual(
                ledger.snapshot()["events"][-1]["cost_basis"],
                "full_reservation_conservative",
            )

    def test_missing_input_token_counter_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, _catalog, inputs, bundle = _direct_fixture(root)
            ledger = BudgetLedger(root / "ledger.json", 1.0)
            modeler = DirectResponsesModelingAgent(ledger, root / "staging")
            client = _Client(
                [bundle],
                counted_input=9_000,
                usages=[_ProviderUsage(9_000, 5_900)],
            )
            client.responses.input_tokens = None
            with (
                mock.patch.dict(
                    os.environ,
                    {"CONTRAPTION_DATA_ROOT": str(data_root)},
                    clear=False,
                ),
                self.assertRaisesRegex(RuntimeError, "counter is required"),
            ):
                modeler.run(inputs, client=client)
            self.assertEqual(client.responses.requests, [])
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["events"], [])
            self.assertEqual(snapshot["reserved"], {})

    def test_new_model_plan_rejects_wrong_owner_and_extra_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _new_model_bundle(root, "direct_new_model_fixture")
            plan = {
                "target_id": "direct_new_model_fixture",
                "recommended_instantiation_root": None,
            }
            DirectResponsesModelingAgent._validate_plan_paths(bundle, plan)
            wrong_owner = json.loads(json.dumps(bundle))
            wrong_owner["artifacts"][0]["path"] = "direct_fixture_resistor.pmdl"
            with self.assertRaisesRegex(ValueError, "category/device directory"):
                DirectResponsesModelingAgent._validate_plan_paths(wrong_owner, plan)
            extra = json.loads(json.dumps(bundle))
            extra["artifacts"].append(
                {"path": "electrical/extra.part", "content": "{}\n"}
            )
            with self.assertRaisesRegex(ValueError, "another part/model"):
                DirectResponsesModelingAgent._validate_plan_paths(extra, plan)

    def test_post_publish_failure_is_quarantined_before_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "run" / "workspace"
            workspace.mkdir(parents=True)
            proposed = workspace / "proposed"
            proposed.mkdir()
            (proposed / "stale.pmdl").write_text("stale\n", encoding="utf-8")
            quarantined = DirectResponsesModelingAgent._quarantine_failed_proposal(
                workspace, workspace.parent, 1
            )
            self.assertIsNotNone(quarantined)
            self.assertFalse(proposed.exists())
            self.assertEqual(
                (quarantined / "stale.pmdl").read_text(encoding="utf-8"),
                "stale\n",
            )
            proposed.mkdir()
            (proposed / "fresh.pmdl").write_text("fresh\n", encoding="utf-8")
            self.assertFalse((proposed / "stale.pmdl").exists())

    def test_clean_twenty_part_isolation_removes_only_verified_copy_targets(self):
        source_jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        with tempfile.TemporaryDirectory() as temporary:
            source_ideal = CATALOG / "electrical/resistors/resistor.pmdl"
            before = hashlib.sha256(source_ideal.read_bytes()).hexdigest()
            manifest = prepare_isolated_replay(
                source_jobs, Path(temporary) / "isolated-data-root"
            )
            jobs = json.loads(source_jobs.read_text(encoding="utf-8"))["jobs"]
            component_root = source_jobs.parent
            eligible = sum(
                bool(
                    modeling_preflight(
                        component_root / job["component_information"]
                    )["eligible"]
                )
                for job in jobs
            )
            self.assertEqual(len(jobs), 20)
            self.assertEqual(eligible, 10)
            self.assertEqual(len(jobs) - eligible, 10)
            self.assertEqual(len(manifest["removed"]["parts"]), 10)
            self.assertEqual(
                len(manifest["removed"]["procurement_records"]), 20
            )
            self.assertEqual(len(manifest["removed"]["historical_pmdls"]), 2)
            self.assertEqual(hashlib.sha256(source_ideal.read_bytes()).hexdigest(), before)
            isolated_ideal = (
                Path(manifest["data_root"])
                / manifest["preserved"]["ideal_resistor_pmdl"]
            )
            self.assertTrue(isolated_ideal.is_file())
            for item in manifest["removed"]["parts"]:
                self.assertFalse((Path(manifest["data_root"]) / item["directory"]).exists())
            manifest_path = Path(temporary) / "isolation-manifest.json"
            run_ledger = Path(temporary) / "agent-budget.json"
            BudgetLedger(run_ledger, 0.50)
            component_assets = Path(manifest["isolated_job_file"]).parent / "component_inputs"
            extra_asset = component_assets / "fixture-datasheet.pdf"
            extra_asset.write_bytes(b"%PDF-1.4\noffline fixture\n")
            state = replay_state_fingerprint(
                manifest["data_root"],
                manifest["isolated_job_file"],
                manifest_path,
                run_ledger,
                "yageo_rc0603_10r",
            )
            validate_replay_state(
                state,
                data_root=manifest["data_root"],
                isolated_job_file=manifest["isolated_job_file"],
                isolation_manifest=manifest_path,
                run_ledger=run_ledger,
                canary_target="yageo_rc0603_10r",
            )

            component = next(
                (Path(manifest["data_root"]) / "outputs").rglob("component_inputs/*.json")
            )
            original_component = component.read_bytes()
            component.write_bytes(original_component + b" ")
            with self.assertRaisesRegex(ValueError, "changed after"):
                validate_replay_state(
                    state,
                    data_root=manifest["data_root"],
                    isolated_job_file=manifest["isolated_job_file"],
                    isolation_manifest=manifest_path,
                    run_ledger=run_ledger,
                    canary_target="yageo_rc0603_10r",
                )
            component.write_bytes(original_component)

            original_extra = extra_asset.read_bytes()
            extra_asset.write_bytes(original_extra + b"changed")
            with self.assertRaisesRegex(ValueError, "changed after"):
                validate_replay_state(
                    state,
                    data_root=manifest["data_root"],
                    isolated_job_file=manifest["isolated_job_file"],
                    isolation_manifest=manifest_path,
                    run_ledger=run_ledger,
                    canary_target="yageo_rc0603_10r",
                )
            extra_asset.write_bytes(original_extra)

            ideal = Path(manifest["data_root"]) / manifest["preserved"][
                "ideal_resistor_pmdl"
            ]
            original_ideal = ideal.read_bytes()
            ideal.write_bytes(original_ideal + b" ")
            with self.assertRaisesRegex(ValueError, "changed after"):
                validate_replay_state(
                    state,
                    data_root=manifest["data_root"],
                    isolated_job_file=manifest["isolated_job_file"],
                    isolation_manifest=manifest_path,
                    run_ledger=run_ledger,
                    canary_target="yageo_rc0603_10r",
                )
            ideal.write_bytes(original_ideal)

            original_manifest = manifest_path.read_bytes()
            manifest_path.write_bytes(original_manifest + b" ")
            with self.assertRaisesRegex(ValueError, "changed after"):
                validate_replay_state(
                    state,
                    data_root=manifest["data_root"],
                    isolated_job_file=manifest["isolated_job_file"],
                    isolation_manifest=manifest_path,
                    run_ledger=run_ledger,
                    canary_target="yageo_rc0603_10r",
                )
            manifest_path.write_bytes(original_manifest)

            original_ledger = run_ledger.read_bytes()
            run_ledger.write_bytes(original_ledger + b" ")
            with self.assertRaisesRegex(ValueError, "changed after"):
                validate_replay_state(
                    state,
                    data_root=manifest["data_root"],
                    isolated_job_file=manifest["isolated_job_file"],
                    isolation_manifest=manifest_path,
                    run_ledger=run_ledger,
                    canary_target="yageo_rc0603_10r",
                )
            run_ledger.write_bytes(original_ledger)

            unsafe_link = Path(manifest["data_root"]) / "unsafe-link"
            unsafe_link.symlink_to(component)
            with self.assertRaisesRegex(ValueError, "symlink"):
                replay_state_fingerprint(
                    manifest["data_root"],
                    manifest["isolated_job_file"],
                    manifest_path,
                    run_ledger,
                    "yageo_rc0603_10r",
                )


class IngestionKpiTests(unittest.TestCase):
    def test_batch_target_subset_cannot_pass_as_full_inventory(self):
        inventory = ["canary", "part-b", "part-c"]
        self.assertEqual(
            complete_batch_targets(inventory, "canary"), ["part-b", "part-c"]
        )
        with self.assertRaisesRegex(ValueError, "full inventory"):
            complete_batch_targets(inventory, "canary", ["part-b"])

    def test_staged_only_result_is_not_a_fully_ingested_denominator(self):
        staged = {
            "target": "staged-only",
            "status": "completed",
            "fully_ingested": False,
            "validation_activity": {"failed_calls": 0},
        }
        metrics = ingestion_metrics(
            [staged], {"events": []}, "staged-run", expected_target_count=1
        )
        self.assertEqual(metrics["fully_ingested_parts"], 0)
        self.assertFalse(metrics["passed"])

    def test_failed_retry_spend_and_expected_target_omission_are_inclusive(self):
        snapshot = {
            "events": [
                {
                    "status": "completed",
                    "charged_usd": 0.004,
                    "metadata": {
                        "ingestion_run_id": "run",
                        "kind": "classification",
                    },
                },
                {
                    "status": "invalid_output",
                    "charged_usd": 0.003,
                    "metadata": {
                        "ingestion_run_id": "run",
                        "kind": "classification",
                    },
                },
                {
                    "status": "invalid_output",
                    "charged_usd": 0.010,
                    "metadata": {
                        "ingestion_run_id": "run",
                        "kind": "modeling",
                    },
                },
                {
                    "status": "completed",
                    "charged_usd": 0.012,
                    "metadata": {
                        "ingestion_run_id": "run",
                        "kind": "modeling",
                    },
                },
            ]
        }
        result = {
            "target": "one",
            "status": "completed",
            "fully_ingested": True,
            "validation_activity": {"failed_calls": 1},
        }
        metrics = ingestion_metrics(
            [result], snapshot, "run", expected_target_count=2
        )
        self.assertAlmostEqual(metrics["total_importer_charged_usd"], 0.029)
        self.assertEqual(metrics["failed_validation_attempts"], 2)
        self.assertFalse(metrics["passed"])
        self.assertIn("omitted", " ".join(metrics["violations"]))

    def test_combined_canary_and_batch_reports_total_ten_part_kpis(self):
        canary = {
            "total_importer_charged_usd": 0.02,
            "fully_ingested_parts": 1,
            "failed_validation_attempts": 0,
            "passed": True,
        }
        batch = {
            "total_importer_charged_usd": 0.18,
            "fully_ingested_parts": 9,
            "failed_validation_attempts": 1,
            "passed": True,
        }
        combined = combine_ingestion_metrics(canary, batch)
        self.assertEqual(combined["fully_ingested_parts"], 10)
        self.assertAlmostEqual(combined["cost_per_fully_ingested_part_usd"], 0.02)
        self.assertAlmostEqual(
            combined["average_failed_validation_attempts_per_fully_ingested_part"],
            0.1,
        )
        self.assertTrue(combined["passed"])

    def test_prior_failed_canary_spend_and_attempts_remain_in_final_kpis(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "failed-v1-agent-budget.json"
            component_digest = "a" * 64
            cost_scope = (
                "part-ingestion:failed-canary:yageo_rc0603_10r:"
                + component_digest
            )
            events = [
                {
                    "status": "completed",
                    "charged_usd": 0.0016322,
                    "scope_limit_breached": False,
                    "metadata": {
                        "ingestion_run_id": "failed-canary",
                        "kind": "classification-canary",
                        "target": "yageo_rc0603_10r",
                        "cost_scope": cost_scope,
                    },
                }
            ]
            for charged in (0.0026223, 0.00161792, 0.00163007):
                events.append(
                    {
                        "status": "invalid_output",
                        "charged_usd": charged,
                        "scope_limit_breached": False,
                        "metadata": {
                            "ingestion_run_id": "failed-canary",
                            "kind": "modeling-canary",
                            "target": "yageo_rc0603_10r",
                            "cost_scope": cost_scope,
                        },
                    }
                )
            ledger.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "limit_usd": 0.5,
                        "spent_usd": 0.00750249,
                        "reserved": {},
                        "events": events,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            carryover = failed_run_carryover(
                ledger,
                expected_target="yageo_rc0603_10r",
                expected_component_sha256="sha256:" + component_digest,
            )
            self.assertAlmostEqual(
                carryover["total_importer_charged_usd"], 0.00750249
            )
            self.assertEqual(carryover["failed_validation_attempts"], 3)
            self.assertEqual(validate_failed_run_carryovers([carryover]), [carryover])
            second_ledger = Path(temporary) / "failed-v2-agent-budget.json"
            second_scope = (
                "part-ingestion:failed-canary-v2:yageo_rc0603_10r:"
                + component_digest
            )
            second_events = [
                {
                    "status": "completed",
                    "charged_usd": 0.001661,
                    "scope_limit_breached": False,
                    "metadata": {
                        "ingestion_run_id": "failed-canary-v2",
                        "kind": "classification-canary",
                        "target": "yageo_rc0603_10r",
                        "cost_scope": second_scope,
                    },
                }
            ]
            for charged in (0.0026774, 0.00167139, 0.00206139):
                second_events.append(
                    {
                        "status": "invalid_output",
                        "charged_usd": charged,
                        "scope_limit_breached": False,
                        "metadata": {
                            "ingestion_run_id": "failed-canary-v2",
                            "kind": "modeling-canary",
                            "target": "yageo_rc0603_10r",
                            "cost_scope": second_scope,
                        },
                    }
                )
            second_ledger.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "limit_usd": 0.5,
                        "spent_usd": 0.00807118,
                        "reserved": {},
                        "events": second_events,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            second_carryover = failed_run_carryover(
                second_ledger,
                expected_target="yageo_rc0603_10r",
                expected_component_sha256="sha256:" + component_digest,
            )
            third_ledger = Path(temporary) / "failed-v3-agent-budget.json"
            third_scope = (
                "part-ingestion:failed-canary-v3:yageo_rc0603_10r:"
                + component_digest
            )
            third_events = [
                {
                    "status": "completed",
                    "charged_usd": 0.00029308,
                    "scope_limit_breached": False,
                    "metadata": {
                        "ingestion_run_id": "failed-canary-v3",
                        "kind": "classification-canary",
                        "target": "yageo_rc0603_10r",
                        "cost_scope": third_scope,
                    },
                }
            ]
            for charged in (0.00257445, 0.00158627, 0.00132192):
                third_events.append(
                    {
                        "status": "invalid_output",
                        "charged_usd": charged,
                        "scope_limit_breached": False,
                        "metadata": {
                            "ingestion_run_id": "failed-canary-v3",
                            "kind": "modeling-canary",
                            "target": "yageo_rc0603_10r",
                            "cost_scope": third_scope,
                        },
                    }
                )
            third_ledger.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "limit_usd": 0.5,
                        "spent_usd": 0.00577572,
                        "reserved": {},
                        "events": third_events,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            third_carryover = failed_run_carryover(
                third_ledger,
                expected_target="yageo_rc0603_10r",
                expected_component_sha256="sha256:" + component_digest,
            )
            carryovers = validate_failed_run_carryovers(
                [carryover, second_carryover, third_carryover]
            )
            self.assertAlmostEqual(
                sum(item["total_importer_charged_usd"] for item in carryovers),
                0.02134939,
            )
            self.assertEqual(
                sum(item["failed_validation_attempts"] for item in carryovers),
                9,
            )
            self.assertEqual(
                {
                    phase: sum(
                        item["failed_validation_attempts_by_phase"].get(phase, 0)
                        for item in carryovers
                    )
                    for phase in ("classification", "modeling")
                },
                {"classification": 0, "modeling": 9},
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_failed_run_carryovers([carryover, carryover])
            canary = {
                "total_importer_charged_usd": 0.004,
                "fully_ingested_parts": 1,
                "failed_validation_attempts": 0,
                "provider_calls": 0,
                "completion_generation_modes": {"host_deterministic": 1},
                "passed": True,
            }
            batch = {
                "total_importer_charged_usd": 0.036,
                "fully_ingested_parts": 9,
                "failed_validation_attempts": 0,
                "provider_calls": 0,
                "completion_generation_modes": {"host_deterministic": 9},
                "passed": True,
            }
            combined = combine_ingestion_metrics_with_carryovers(
                canary, batch, carryovers
            )
            self.assertEqual(combined["fully_ingested_parts"], 10)
            self.assertAlmostEqual(
                combined["total_importer_charged_usd"], 0.06134939
            )
            self.assertEqual(combined["failed_validation_attempts"], 9)
            self.assertAlmostEqual(
                combined[
                    "average_failed_validation_attempts_per_fully_ingested_part"
                ],
                0.9,
            )
            self.assertEqual(combined["current_replay_provider_calls"], 0)
            self.assertEqual(combined["carryover_provider_calls"], 12)
            self.assertEqual(
                combined["completion_generation_modes"],
                {"host_deterministic": 10},
            )
            self.assertEqual(
                combined["backend_outcomes"],
                {
                    "prior_provider_failed_validation_attempts": 9,
                    "prior_provider_failed_validation_attempts_by_phase": {
                        "modeling": 9
                    },
                    "prior_provider_calls_by_phase": {
                        "classification": 3,
                        "modeling": 9,
                    },
                    "current_replay_provider_calls": 0,
                    "current_replay_completion_generation_modes": {
                        "host_deterministic": 10
                    },
                },
            )
            self.assertTrue(combined["passed"])
            with self.assertRaisesRegex(ValueError, "differ"):
                validate_matching_failed_run_carryovers(
                    carryovers, carryovers, [carryover, second_carryover]
                )

            prior_total = sum(
                item["total_importer_charged_usd"] for item in carryovers
            )
            remaining = 0.05 - prior_total
            self.assertAlmostEqual(remaining, 0.02865061)
            scoped = BudgetLedger(Path(temporary) / "retry-ledger.json", 1.0)
            scoped.reserve(
                "classification",
                0.005,
                {},
                cost_scope="retry-part",
                cost_scope_limit_usd=remaining,
            )
            scoped.reserve(
                "modeling-1",
                0.011,
                {},
                cost_scope="retry-part",
                cost_scope_limit_usd=remaining,
            )
            scoped.reserve(
                "modeling-2",
                0.011,
                {},
                cost_scope="retry-part",
                cost_scope_limit_usd=remaining,
            )
            with self.assertRaises(BudgetExceeded):
                scoped.reserve(
                    "would-cross-cumulative-cap",
                    remaining - 0.027,
                    {},
                    cost_scope="retry-part",
                    cost_scope_limit_usd=remaining,
                )
            reserved = sum(
                float(item["max_usd"])
                for item in scoped.snapshot()["reserved"].values()
            )
            self.assertLess(
                prior_total + reserved,
                0.05,
            )

            ledger.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_failed_run_carryovers([carryover])

    def test_forged_failed_canary_metrics_cannot_unlock_batch(self):
        result = {
            "target": "yageo_rc0603_10r",
            "status": "completed",
            "fully_ingested": True,
            "promoted_part_id": "yageo_rc0603_10r",
            "promoted_model_id": "yageo_rc0603_10r.v1",
        }
        events = [
            {
                "status": "invalid_output",
                "charged_usd": 0.001,
                "scope_limit_breached": False,
                "metadata": {
                    "ingestion_run_id": "forged-run",
                    "kind": "modeling-canary",
                    "target": "yageo_rc0603_10r",
                },
            }
            for _index in range(3)
        ]
        report = {
            "ingestion_run_id": "forged-run",
            "results": [result],
            "metrics": {
                "passed": True,
                "fully_ingested_parts": 1,
            },
        }
        with self.assertRaisesRegex(ValueError, "recomputation"):
            validate_canary_report_evidence(
                report,
                ledger_snapshot={"events": events},
                catalog_root=CATALOG,
                expected_target="yageo_rc0603_10r",
            )


class IngestionCliGateTests(unittest.TestCase):
    def test_batch_rejects_missing_changed_symlink_and_special_ledger_preconstruction(self):
        parser = build_parser()
        jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        outputs = PROJECT_ROOT / "outputs"
        cases = {
            "missing": "missing",
            "changed": "changed after the canary",
            "symlink": "symlink",
            "special": "regular non-symlink",
        }
        with tempfile.TemporaryDirectory(dir=outputs) as temporary:
            root = Path(temporary)
            for case, error in cases.items():
                replay = root / case
                replay.mkdir()
                ledger_path = replay / "agent-budget.json"
                BudgetLedger(ledger_path, 0.50)
                binding = _ingestion_ledger_binding(replay)
                report_path = replay / "ingestion-canary-report.json"
                report_path.write_text(
                    json.dumps(
                        {
                            "replay_run_root": str(replay.resolve()),
                            "run_ledger": binding,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if case == "missing":
                    ledger_path.unlink()
                elif case == "changed":
                    ledger_path.write_bytes(ledger_path.read_bytes() + b" ")
                elif case == "symlink":
                    external = root / f"{case}-external-ledger.json"
                    external.write_bytes(ledger_path.read_bytes())
                    ledger_path.unlink()
                    ledger_path.symlink_to(external)
                else:
                    ledger_path.unlink()
                    ledger_path.mkdir()
                args = parser.parse_args(
                    [
                        "agent-run",
                        "ingestion-batch",
                        "--job-file",
                        str(jobs),
                        "--canary-report",
                        str(report_path),
                        "--output-root",
                        str(replay),
                    ]
                )
                with (
                    mock.patch(
                        "contraption.cli._agent_key",
                        return_value=("test-key", "fixture"),
                    ),
                    mock.patch("contraption.cli._agent_ledger") as ledger_factory,
                    self.assertRaisesRegex((ValueError, FileNotFoundError), error),
                ):
                    command_agent_run(args)
                ledger_factory.assert_not_called()

    def test_failed_canary_always_writes_ledger_derived_nonpassing_report(self):
        parser = build_parser()
        jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        outputs = PROJECT_ROOT / "outputs"
        with tempfile.TemporaryDirectory(dir=outputs) as temporary:
            replay = Path(temporary)
            args = parser.parse_args(
                [
                    "agent-run",
                    "ingestion-canary",
                    "--job-file",
                    str(jobs),
                    "--target",
                    "yageo_rc0603_10r",
                    "--output-root",
                    str(replay),
                ]
            )

            def fail_after_three_invalid_calls(classifier, _modeler, *_args, **kwargs):
                run_id = kwargs["ingestion_run_id"]
                for attempt in range(1, 4):
                    call_id = f"offline-invalid-{attempt}"
                    classifier.ledger.reserve(
                        call_id,
                        0.001,
                        {
                            "ingestion_run_id": run_id,
                            "kind": "modeling-canary",
                            "target": "yageo_rc0603_10r",
                        },
                    )
                    classifier.ledger.settle(
                        call_id,
                        usage=Usage(
                            100,
                            0,
                            100,
                            cache_write_input_tokens=0,
                        ),
                        pricing=TokenPricing(),
                        status="invalid_output",
                    )
                raise ValueError(
                    "PhysicalSpecError: missing model instance field(s): "
                    "parameter_uncertainty"
                )

            with (
                mock.patch.dict(os.environ, {}, clear=False),
                mock.patch(
                    "contraption.cli._agent_key",
                    return_value=("test-key", "fixture"),
                ),
                mock.patch(
                    "contraption.cli.run_part_ingestion",
                    side_effect=fail_after_three_invalid_calls,
                ),
                mock.patch("builtins.print"),
            ):
                exit_code = command_agent_run(args)
            self.assertEqual(exit_code, 2)
            report_path = replay / "ingestion-canary-report.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["metrics"]["passed"])
            self.assertEqual(report["metrics"]["failed_validation_attempts"], 3)
            self.assertGreater(report["metrics"]["total_importer_charged_usd"], 0)
            self.assertEqual(report["results"][0]["status"], "failed")
            self.assertIn("parameter_uncertainty", report["failure_reason"])
            self.assertIn("replay_state", report)
            self.assertNotIn("replay_state_error", report)
            self.assertEqual(
                report["run_ledger"], _ingestion_ledger_binding(replay)
            )
            self.assertEqual(
                report["replay_state"]["run_ledger"],
                {
                    "path": report["run_ledger"]["path"],
                    "sha256": report["run_ledger"]["sha256"],
                },
            )

    def test_canary_replay_state_failure_is_reported_and_nonpassing(self):
        parser = build_parser()
        jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        outputs = PROJECT_ROOT / "outputs"
        with tempfile.TemporaryDirectory(dir=outputs) as temporary:
            replay = Path(temporary)
            args = parser.parse_args(
                [
                    "agent-run",
                    "ingestion-canary",
                    "--job-file",
                    str(jobs),
                    "--target",
                    "yageo_rc0603_10r",
                    "--output-root",
                    str(replay),
                ]
            )
            completed = {
                "target": "yageo_rc0603_10r",
                "status": "completed",
                "fully_ingested": True,
                "charged_usd": 0.0,
                "classification": {
                    "generation_mode": "host_deterministic",
                    "provider_calls": 0,
                    "charged_usd": 0.0,
                },
                "modeling": {
                    "generation_mode": "host_deterministic",
                    "provider_calls": 0,
                    "charged_usd": 0.0,
                },
                "validation_activity": {"failed_calls": 0},
            }
            with (
                mock.patch(
                    "contraption.cli._agent_key",
                    return_value=("test-key", "fixture"),
                ),
                mock.patch(
                    "contraption.cli.run_part_ingestion",
                    return_value=completed,
                ),
                mock.patch(
                    "contraption.cli.replay_state_fingerprint",
                    side_effect=ValueError("offline replay fingerprint failure"),
                ),
                mock.patch("builtins.print"),
            ):
                exit_code = command_agent_run(args)
            self.assertEqual(exit_code, 2)
            report = json.loads(
                (replay / "ingestion-canary-report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(report["metrics"]["passed"])
            self.assertEqual(report["metrics"]["fully_ingested_parts"], 0)
            self.assertEqual(report["results"][0]["status"], "failed")
            self.assertFalse(report["results"][0]["fully_ingested"])
            self.assertIn("fingerprint failure", report["replay_state_error"])
            self.assertEqual(report["failure_reason"], report["replay_state_error"])

    def test_canary_and_batch_parser_contract_and_canary_gate(self):
        parser = build_parser()
        canary = parser.parse_args(
            [
                "agent-run",
                "ingestion-canary",
                "--job-file",
                "jobs.json",
                "--target",
                "part-a",
                "--output-root",
                "replay-run",
                "--prior-failed-ledger",
                "prior-v1-ledger.json",
                "--prior-failed-ledger",
                "prior-v2-ledger.json",
            ]
        )
        self.assertEqual(canary.agent_job, "ingestion-canary")
        self.assertEqual(canary.target, "part-a")
        self.assertEqual(canary.output_root, "replay-run")
        self.assertEqual(canary.ledger_limit_usd, 0.50)
        self.assertEqual(
            canary.prior_failed_ledger,
            ["prior-v1-ledger.json", "prior-v2-ledger.json"],
        )
        batch = parser.parse_args(
            [
                "agent-run",
                "ingestion-batch",
                "--job-file",
                "jobs.json",
                "--canary-report",
                "canary.json",
                "--output-root",
                "replay-run",
                "--target",
                "part-b",
                "--prior-failed-ledger",
                "prior-v1-ledger.json",
                "--prior-failed-ledger",
                "prior-v2-ledger.json",
            ]
        )
        self.assertEqual(batch.agent_job, "ingestion-batch")
        self.assertEqual(batch.target, ["part-b"])
        self.assertEqual(
            batch.prior_failed_ledger,
            ["prior-v1-ledger.json", "prior-v2-ledger.json"],
        )
        report = {
            "schema": "contraption.part-ingestion-report/v1",
            "mode": "canary",
            "workflow_sha256": "sha256:workflow",
            "job_file_sha256": "sha256:jobs",
            "metrics": {"passed": True, "fully_ingested_parts": 1},
        }
        validate_canary_gate(
            report,
            workflow_sha256="sha256:workflow",
            job_file_sha256="sha256:jobs",
        )
        failed = json.loads(json.dumps(report))
        failed["metrics"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "did not pass"):
            validate_canary_gate(
                failed,
                workflow_sha256="sha256:workflow",
                job_file_sha256="sha256:jobs",
            )

    def test_ingestion_paths_and_ledger_are_replay_local(self):
        parser = build_parser()
        jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        replay = PROJECT_ROOT / "outputs/offline-ingestion-path-test"
        base = [
            "agent-run",
            "ingestion-canary",
            "--job-file",
            str(jobs),
            "--target",
            "yageo_rc0603_10r",
        ]
        with mock.patch(
            "contraption.cli._agent_key", return_value=("test-key", "fixture")
        ):
            source_args = parser.parse_args(
                [*base, "--output-root", str(jobs.parent)]
            )
            with self.assertRaisesRegex(ValueError, "source job run"):
                command_agent_run(source_args)

            outside_args = parser.parse_args(
                [*base, "--output-root", str(PROJECT_ROOT / "assembled_contraptions/replay")]
            )
            with self.assertRaisesRegex(ValueError, "descendant of outputs"):
                command_agent_run(outside_args)

            staging_args = parser.parse_args(
                [
                    *base,
                    "--output-root",
                    str(replay),
                    "--staging-root",
                    str(PROJECT_ROOT / "assembled_contraptions/staging"),
                ]
            )
            with self.assertRaisesRegex(ValueError, "staging must be"):
                command_agent_run(staging_args)

            for value in ("nan", "inf", "-inf"):
                limit_args = parser.parse_args(
                    [
                        *base,
                        "--output-root",
                        str(replay),
                        f"--ledger-limit-usd={value}",
                    ]
                )
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    command_agent_run(limit_args)

        args = parser.parse_args([*base, "--output-root", str(replay)])
        with (
            mock.patch(
                "contraption.cli._agent_key", return_value=("test-key", "fixture")
            ),
            mock.patch("contraption.cli._agent_ledger") as ledger_factory,
            mock.patch(
                "contraption.cli.prepare_isolated_replay",
                side_effect=RuntimeError("offline stop"),
            ),
            self.assertRaisesRegex(RuntimeError, "offline stop"),
        ):
            command_agent_run(args)
        ledger_factory.assert_called_once_with(
            str((replay / "agent-budget.json").resolve()), limit_usd=0.50
        )

    def test_replay_root_must_be_fresh_and_direct_children_cannot_be_symlinks(self):
        parser = build_parser()
        jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        outputs = PROJECT_ROOT / "outputs"
        with tempfile.TemporaryDirectory(dir=outputs) as temporary:
            replay = Path(temporary)
            (replay / "stale.json").write_text("{}\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "agent-run",
                    "ingestion-canary",
                    "--job-file",
                    str(jobs),
                    "--target",
                    "yageo_rc0603_10r",
                    "--output-root",
                    str(replay),
                ]
            )
            with (
                mock.patch(
                    "contraption.cli._agent_key",
                    return_value=("test-key", "fixture"),
                ),
                self.assertRaisesRegex(ValueError, "absent or an empty"),
            ):
                command_agent_run(args)

        with tempfile.TemporaryDirectory(dir=outputs) as temporary:
            root = Path(temporary)
            external = Path(tempfile.mkdtemp())
            try:
                (root / "agent-staging").symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    _replay_direct_child(root.resolve(), "agent-staging")
            finally:
                shutil.rmtree(external)

    def test_batch_output_root_must_match_canary_run(self):
        parser = build_parser()
        jobs = PROJECT_ROOT / "outputs/part-import-2026-08-18/agent_jobs.json"
        canary_root = PROJECT_ROOT / "outputs/offline-canary-root"
        batch_root = PROJECT_ROOT / "outputs/offline-batch-root"
        args = parser.parse_args(
            [
                "agent-run",
                "ingestion-batch",
                "--job-file",
                str(jobs),
                "--canary-report",
                str(canary_root / "ingestion-canary-report.json"),
                "--output-root",
                str(batch_root),
            ]
        )
        report = {"replay_run_root": str(canary_root)}
        with (
            mock.patch(
                "contraption.cli._agent_key", return_value=("test-key", "fixture")
            ),
            mock.patch("contraption.cli._load_json", return_value=report),
            self.assertRaisesRegex(ValueError, "differs from the canary"),
        ):
            command_agent_run(args)


if __name__ == "__main__":
    unittest.main()
