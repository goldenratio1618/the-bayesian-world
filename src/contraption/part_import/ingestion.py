"""Cost-gated, end-to-end part-ingestion orchestration and KPIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from .agents import (
    CLASSIFICATION_SCHEMA,
    MAX_FULLY_INGESTED_PART_COST_USD,
    RESPONSES_MODELING_SCHEMA,
    ClassificationAgent,
    DirectResponsesModelingAgent,
    ModelingInputs,
    modeling_preflight,
    run_classification_batch,
    run_modeling_proposal,
    validate_classification_proposal,
    write_json_atomic,
)
from .budget import BudgetLedger


INGESTION_REPORT_SCHEMA = "contraption.part-ingestion-report/v1"
FAILED_RUN_CARRYOVER_SCHEMA = "contraption.part-ingestion-failed-run/v1"


@dataclass(frozen=True)
class IngestionPolicy:
    """Strict goals for newly completed parts in one measured run."""

    max_cost_per_fully_ingested_part_usd: float = (
        MAX_FULLY_INGESTED_PART_COST_USD
    )
    max_average_failed_validation_attempts: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.max_cost_per_fully_ingested_part_usd <= 0.05:
            raise ValueError("per-part cost limit must be in (0, 0.05]")
        if not 0 < self.max_average_failed_validation_attempts <= 1.0:
            raise ValueError("failed-validation average limit must be in (0, 1]")


def part_cost_scope(run_id: str, target: str, component: str | Path) -> str:
    if not run_id or not target:
        raise ValueError("run_id and target are required for a part cost scope")
    digest = hashlib.sha256(Path(component).read_bytes()).hexdigest()
    return f"part-ingestion:{run_id}:{target}:{digest}"


def _regular_file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"replay state path is not a regular file: {path}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_tree_digest(root: Path) -> dict[str, Any]:
    raw_root = root.expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"replay state root is a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError(f"replay state root is not a regular directory: {root}")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"replay state contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"replay state contains a special file: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
        total_bytes += len(payload)
    return {
        "sha256": "sha256:" + digest.hexdigest(),
        "regular_file_count": count,
        "total_file_bytes": total_bytes,
    }


def replay_state_fingerprint(
    data_root: str | Path,
    isolated_job_file: str | Path,
    isolation_manifest: str | Path,
    run_ledger: str | Path,
    canary_target: str,
) -> dict[str, Any]:
    """Bind the exact post-canary replay bytes required by the batch."""

    raw_root = Path(data_root).expanduser()
    raw_job = Path(isolated_job_file).expanduser()
    raw_manifest = Path(isolation_manifest).expanduser()
    raw_ledger = Path(run_ledger).expanduser()
    if (
        raw_root.is_symlink()
        or raw_job.is_symlink()
        or raw_manifest.is_symlink()
        or raw_ledger.is_symlink()
    ):
        raise ValueError("replay state control paths cannot be symlinks")
    root = raw_root.resolve()
    job_path = raw_job.resolve()
    manifest_path = raw_manifest.resolve()
    ledger_path = raw_ledger.resolve()
    if root not in job_path.parents:
        raise ValueError("isolated job inventory is outside its replay data root")
    if manifest_path != root.parent / "isolation-manifest.json":
        raise ValueError("isolation manifest is outside the replay run root")
    if ledger_path != root.parent / "agent-budget.json":
        raise ValueError("run ledger is outside the replay run root")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if not isinstance(job, dict) or job.get("format") != "agent-jobs-1":
        raise ValueError("isolated replay job inventory has the wrong format")
    target_ids = {
        item.get("id")
        for item in job.get("jobs", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if canary_target not in target_ids:
        raise ValueError("canary target is absent from the isolated job inventory")
    component_root = job_path.parent / "component_inputs"
    component_files: list[dict[str, Any]] = []
    if component_root.is_symlink() or not component_root.is_dir():
        raise ValueError("isolated component-input directory is missing or unsafe")
    expected_components: set[Path] = set()
    for item in job.get("jobs", []):
        if not isinstance(item, Mapping) or not isinstance(
            item.get("component_information"), str
        ):
            raise ValueError("isolated replay job has an invalid component path")
        component = (job_path.parent / item["component_information"]).resolve()
        if component_root.resolve() not in component.parents:
            raise ValueError("isolated replay component escaped component_inputs")
        expected_components.add(component)
    for path in sorted(component_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"isolated component inputs contain a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"isolated component inputs contain a special file: {path}")
        component_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _regular_file_sha256(path),
            }
        )
    discovered_components = {
        (root / item["path"]).resolve() for item in component_files
    }
    if not expected_components.issubset(discovered_components):
        raise ValueError(
            "isolated job-declared component inputs are missing from their asset closure"
        )
    catalog = root / "model_catalog"
    return {
        "schema": "contraption.part-import-replay-state/v1",
        "canary_target": canary_target,
        "data_root": str(root),
        "isolation_manifest": {
            "path": str(manifest_path),
            "sha256": _regular_file_sha256(manifest_path),
        },
        "isolated_job": {
            "path": str(job_path),
            "sha256": _regular_file_sha256(job_path),
        },
        "run_ledger": {
            "path": str(ledger_path),
            "sha256": _regular_file_sha256(ledger_path),
        },
        "component_inputs": component_files,
        "catalog_tree": _regular_tree_digest(catalog),
        "data_root_tree": _regular_tree_digest(root),
    }


def validate_replay_state(
    expected: Mapping[str, Any],
    *,
    data_root: str | Path,
    isolated_job_file: str | Path,
    isolation_manifest: str | Path,
    run_ledger: str | Path,
    canary_target: str,
) -> None:
    actual = replay_state_fingerprint(
        data_root,
        isolated_job_file,
        isolation_manifest,
        run_ledger,
        canary_target,
    )
    if dict(expected) != actual:
        raise ValueError(
            "isolated replay state changed after the passing canary; batch refused"
        )


def complete_batch_targets(
    inventory_targets: Iterable[str],
    canary_target: str,
    requested_targets: Iterable[str] = (),
) -> list[str]:
    """Return the complete non-canary inventory or reject a partial batch."""

    inventory = list(inventory_targets)
    if (
        not inventory
        or any(not isinstance(item, str) or not item for item in inventory)
        or len(inventory) != len(set(inventory))
    ):
        raise ValueError("ingestion inventory targets must be unique nonempty strings")
    if inventory.count(canary_target) != 1:
        raise ValueError("canary target must occur exactly once in the inventory")
    expected = [target for target in inventory if target != canary_target]
    requested = list(requested_targets)
    selected = requested or expected
    if len(selected) != len(set(selected)):
        raise ValueError("ingestion-batch targets must be unique")
    if len(selected) != len(expected) or set(selected) != set(expected):
        raise ValueError(
            "ingestion-batch targets must equal the full inventory minus canary"
        )
    return selected


def prepare_isolated_replay(
    source_job_file: str | Path,
    destination_data_root: str | Path,
) -> dict[str, Any]:
    """Create a clean, identity-checked catalog copy for a paid replay.

    Existing target parts and procurement records are removed only from the new
    copy and only after their exact product/id/model bytes are recorded. Shared
    parent interfaces and the ideal resistor PMDL remain intact.
    """

    source_job_file = Path(source_job_file).expanduser().resolve()
    destination = Path(destination_data_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"isolated replay data root must be new: {destination}"
        )
    raw_jobs = json.loads(source_job_file.read_text(encoding="utf-8"))
    if not isinstance(raw_jobs, dict) or raw_jobs.get("format") != "agent-jobs-1":
        raise ValueError("isolated replay requires an agent-jobs-1 inventory")
    session_root = source_job_file.parent
    catalog_value = raw_jobs.get("catalog")
    constraints_value = raw_jobs.get("constraints")
    if not isinstance(catalog_value, str) or not isinstance(constraints_value, str):
        raise ValueError("agent job catalog and constraints paths must be strings")
    source_catalog = (session_root / catalog_value).resolve()
    source_constraints = (session_root / constraints_value).resolve()
    if not source_catalog.is_dir() or not source_constraints.is_file():
        raise FileNotFoundError("agent job catalog or constraints source is missing")

    destination.mkdir(parents=True)
    isolated_catalog = destination / "model_catalog"
    shutil.copytree(source_catalog, isolated_catalog)
    isolated_constraints = destination / "prompts" / source_constraints.name
    isolated_constraints.parent.mkdir(parents=True)
    shutil.copy2(source_constraints, isolated_constraints)
    # asset_root() requires both top-level asset directories. The replay never
    # reads or writes a production contraption under this empty directory.
    (destination / "assembled_contraptions").mkdir()
    isolated_session = destination / "outputs" / session_root.name
    isolated_session.mkdir(parents=True)
    shutil.copy2(source_job_file, isolated_session / source_job_file.name)
    source_components = session_root / "component_inputs"
    shutil.copytree(source_components, isolated_session / "component_inputs")

    static_by_product: dict[str, list[Path]] = {}
    for static_path in isolated_catalog.rglob("static.part"):
        text = static_path.read_text(encoding="utf-8")
        for raw_job in raw_jobs.get("jobs", []):
            if not isinstance(raw_job, Mapping):
                continue
            component_path = session_root / str(raw_job.get("component_information", ""))
            component = json.loads(component_path.read_text(encoding="utf-8"))
            product = component.get("product")
            if isinstance(product, str) and product.casefold() in text.casefold():
                static_by_product.setdefault(product.casefold(), []).append(static_path)

    procurement_by_product: dict[str, list[Path]] = {}
    for procurement_path in isolated_catalog.rglob("*.procurement"):
        text = procurement_path.read_text(encoding="utf-8")
        for raw_job in raw_jobs.get("jobs", []):
            if not isinstance(raw_job, Mapping):
                continue
            component_path = session_root / str(raw_job.get("component_information", ""))
            component = json.loads(component_path.read_text(encoding="utf-8"))
            product = component.get("product")
            if isinstance(product, str) and product.casefold() in text.casefold():
                procurement_by_product.setdefault(product.casefold(), []).append(
                    procurement_path
                )

    removed_parts: list[dict[str, Any]] = []
    removed_procurement: list[dict[str, Any]] = []
    removed_directories: set[Path] = set()
    for raw_job in raw_jobs.get("jobs", []):
        if not isinstance(raw_job, Mapping):
            raise ValueError("agent job entry must be an object")
        target = raw_job.get("id")
        relative_component = raw_job.get("component_information")
        if not isinstance(target, str) or not isinstance(relative_component, str):
            raise ValueError("agent job target/component path is invalid")
        source_component = (session_root / relative_component).resolve()
        component = json.loads(source_component.read_text(encoding="utf-8"))
        product = component.get("product")
        if not isinstance(product, str) or not product:
            raise ValueError(f"component {target!r} has no exact product identity")
        eligible = modeling_preflight(source_component)["eligible"]
        static_matches = list(dict.fromkeys(static_by_product.get(product.casefold(), [])))
        procurement_matches = list(
            dict.fromkeys(procurement_by_product.get(product.casefold(), []))
        )
        if eligible:
            if len(static_matches) != 1:
                raise ValueError(
                    f"eligible replay target {target!r} matched {len(static_matches)} static parts"
                )
            static_path = static_matches[0]
            part = json.loads(static_path.read_text(encoding="utf-8"))
            model_path = static_path.parent / "v1.model"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"eligible replay target {target!r} has no v1.model"
                )
            model = json.loads(model_path.read_text(encoding="utf-8"))
            part_id = part.get("id")
            if not isinstance(part_id, str) or model.get("part") != part_id:
                raise ValueError(
                    f"eligible replay target {target!r} has inconsistent part/model identity"
                )
            directory = static_path.parent.resolve()
            if isolated_catalog.resolve() not in directory.parents:
                raise ValueError("target removal escaped the isolated catalog")
            removed_parts.append(
                {
                    "target": target,
                    "product": product,
                    "part_id": part_id,
                    "directory": directory.relative_to(destination).as_posix(),
                    "static_sha256": "sha256:"
                    + hashlib.sha256(static_path.read_bytes()).hexdigest(),
                    "model_sha256": "sha256:"
                    + hashlib.sha256(model_path.read_bytes()).hexdigest(),
                }
            )
            removed_directories.add(directory)
        if len(procurement_matches) > 1:
            raise ValueError(
                f"replay target {target!r} matched multiple procurement records"
            )
        if procurement_matches:
            procurement_path = procurement_matches[0]
            removed_procurement.append(
                {
                    "target": target,
                    "product": product,
                    "path": procurement_path.relative_to(destination).as_posix(),
                    "sha256": "sha256:"
                    + hashlib.sha256(procurement_path.read_bytes()).hexdigest(),
                }
            )
            if not eligible:
                directory = procurement_path.parent.resolve()
                if isolated_catalog.resolve() not in directory.parents:
                    raise ValueError("procurement removal escaped the isolated catalog")
                removed_directories.add(directory)

    for directory in sorted(removed_directories, key=lambda path: len(path.parts), reverse=True):
        if directory.exists():
            shutil.rmtree(directory)

    removed_pmdls: list[dict[str, Any]] = []
    for relative in (
        "electrical/resistors/fixed_resistors/resistor_with_parasitics.pmdl",
        "electrical/resistors/fixed_resistors/resistor_thick_film_parasitic.pmdl",
    ):
        path = isolated_catalog / relative
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            removed_pmdls.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "id": payload.get("id"),
                    "version": payload.get("version"),
                    "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            path.unlink()

    ideal = isolated_catalog / "electrical" / "resistors" / "resistor.pmdl"
    if not ideal.is_file():
        raise FileNotFoundError("isolated replay lost the shared ideal resistor PMDL")
    for item in removed_parts:
        if (destination / item["directory"]).exists():
            raise RuntimeError("isolated target part still exists after clean-room removal")
    manifest = {
        "schema": "contraption.part-import-replay-isolation/v2",
        "source_job_file": str(source_job_file),
        "source_job_sha256": "sha256:"
        + hashlib.sha256(source_job_file.read_bytes()).hexdigest(),
        "data_root": str(destination),
        "isolated_job_file": str(isolated_session / source_job_file.name),
        "catalog": str(isolated_catalog),
        "preserved": {
            "ideal_resistor_pmdl": ideal.relative_to(destination).as_posix(),
            "ideal_resistor_sha256": "sha256:"
            + hashlib.sha256(ideal.read_bytes()).hexdigest(),
        },
        "removed": {
            "parts": removed_parts,
            "procurement_records": removed_procurement,
            "historical_pmdls": removed_pmdls,
        },
        "preconditions": {
            "source_catalog_unchanged": True,
            "target_directories_absent_in_isolated_catalog": True,
            "shared_ideal_resistor_present": True,
        },
    }
    manifest_path = destination.parent / "isolation-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _host_recipe_classification(
    inputs: ModelingInputs,
    target: str,
    catalog_root: str | Path,
    output_directory: str | Path,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the exact physical hierarchy implied by a matched host recipe."""

    from ..catalog.interfaces import load_interface_catalog

    proposal = {
        "canonical_name": "0603 fixed thick-film resistor",
        "domains": ["electrical"],
        "reuse_path": ["resistor", "fixed-resistor"],
        "new_nodes": [],
        "category": "resistor",
        "device": "fixed-resistor",
        "rationale": (
            "The versioned host rectangular-chip recipe binds the existing "
            "fixed-resistor hierarchy and exact ideal-resistor PMDL."
        ),
        "uncertainties": [
            "Terminal geometry and land pattern are unavailable; connector frames "
            "are explicit host envelope-face estimates."
        ],
    }
    interface_data = load_interface_catalog(catalog_root).to_dict()
    validate_classification_proposal(proposal, interface_data)
    component_path = Path(inputs.component_information)
    recipe_sha256 = recipe.get("recipe_sha256")
    component_evidence = recipe.get("component_evidence")
    if (
        not isinstance(recipe_sha256, str)
        or not isinstance(component_evidence, Mapping)
        or recipe.get("target_id") != target
        or component_evidence.get("sha256")
        != "sha256:" + hashlib.sha256(component_path.read_bytes()).hexdigest()
    ):
        raise ValueError("host recipe classification/component binding is invalid")
    hash_payload = {
        "workflow": "contraption.host-recipe-classification/v1",
        "target": target,
        "recipe_sha256": recipe_sha256,
        "component_sha256": component_evidence["sha256"],
        "proposal": proposal,
    }
    input_hash = "sha256:" + hashlib.sha256(
        json.dumps(
            hash_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    receipt_path = Path(output_directory) / f"{target}.json"
    receipt = {
        "schema": "contraption.classification-proposal/v1",
        "status": "completed",
        "target": target,
        "source_file": component_path.name,
        "input_hash": input_hash,
        "model": "host-deterministic",
        "reasoning_effort": None,
        "proposal": proposal,
        "usage": None,
        "charged_usd": 0.0,
        "generation_mode": "host_deterministic",
        "provider_calls": 0,
        "recipe_sha256": recipe_sha256,
    }
    write_json_atomic(receipt_path, receipt)
    return {
        "target": target,
        "status": "completed",
        "input_hash": input_hash,
        "proposal_path": str(receipt_path.resolve()),
        "usage": None,
        "charged_usd": 0.0,
        "generation_mode": "host_deterministic",
        "provider_calls": 0,
        "recipe_sha256": recipe_sha256,
    }


def run_part_ingestion(
    classification_agent: ClassificationAgent,
    modeling_agent: DirectResponsesModelingAgent,
    inputs: ModelingInputs,
    target: str,
    catalog_root: str | Path,
    output_directory: str | Path,
    *,
    ingestion_run_id: str,
    policy: IngestionPolicy = IngestionPolicy(),
    canary: bool = False,
    force: bool = False,
    prior_target_charged_usd: float = 0.0,
    classification_client: Any | None = None,
    modeling_client: Any | None = None,
) -> dict[str, Any]:
    """Classify and model one part under one atomic cost scope.

    Unsupported-physics targets are preflighted before classification, so they
    remain zero-dispatch and are excluded from the fully-ingested denominator.
    """

    preflight = modeling_preflight(inputs.component_information)
    root = Path(output_directory)
    if (
        isinstance(prior_target_charged_usd, bool)
        or not isinstance(prior_target_charged_usd, (int, float))
        or not math.isfinite(float(prior_target_charged_usd))
        or not 0 <= float(prior_target_charged_usd)
        < policy.max_cost_per_fully_ingested_part_usd
    ):
        raise ValueError("prior target charge must be finite and below the part cap")
    remaining_scope_limit = (
        policy.max_cost_per_fully_ingested_part_usd
        - float(prior_target_charged_usd)
    )
    if not preflight["eligible"]:
        modeling = run_modeling_proposal(
            modeling_agent,
            inputs,
            target,
            root / "modeling",
            force=force,
            canary=canary,
        )
        promoted: list[str] = []
        staged = modeling.get("staging_artifacts")
        if isinstance(staged, str):
            promoted = [
                str(path)
                for path in modeling_agent.promote(staged, catalog_root)
            ]
            from ..catalog.procurement import ProcurementRegistry

            ProcurementRegistry.load_catalog(catalog_root)
        return {
            "target": target,
            "status": "deferred_unsupported_physics",
            "preflight": preflight,
            "classification": {
                "status": "not_dispatched_unsupported_physics",
                "charged_usd": 0.0,
            },
            "modeling": modeling,
            "promoted_paths": promoted,
            "charged_usd": 0.0,
            "fully_ingested": False,
        }

    scope = part_cost_scope(
        ingestion_run_id, target, inputs.component_information
    )
    deterministic_recipe = modeling_agent.deterministic_recipe_for(inputs)
    if deterministic_recipe is not None:
        classification = _host_recipe_classification(
            inputs,
            target,
            catalog_root,
            root / "classification",
            deterministic_recipe,
        )
    else:
        classification = run_classification_batch(
            classification_agent,
            (inputs.component_information,),
            catalog_root,
            root / "classification",
            force=force,
            client=classification_client,
            canary=canary,
            ingestion_run_id=ingestion_run_id,
            cost_scope=scope,
            cost_scope_limit_usd=remaining_scope_limit,
        )[0]
    modeling = run_modeling_proposal(
        modeling_agent,
        inputs,
        target,
        root / "modeling",
        force=force,
        canary=canary,
        ingestion_run_id=ingestion_run_id,
        cost_scope=scope,
        cost_scope_limit_usd=remaining_scope_limit,
        client=modeling_client,
    )
    bound_recipe_sha256: str | None = None
    classification_is_host = (
        classification.get("generation_mode") == "host_deterministic"
    )
    modeling_is_host = modeling.get("generation_mode") == "host_deterministic"
    if classification_is_host or modeling_is_host:
        classification_recipe = classification.get("recipe_sha256")
        activity = modeling.get("validation_activity")
        host_generation = (
            activity.get("host_generation") if isinstance(activity, Mapping) else None
        )
        modeling_recipe = (
            host_generation.get("recipe_sha256")
            if isinstance(host_generation, Mapping)
            else None
        )
        if (
            not classification_is_host
            or not modeling_is_host
            or not isinstance(classification_recipe, str)
            or not isinstance(modeling_recipe, str)
            or classification_recipe != modeling_recipe
        ):
            raise ValueError(
                "host deterministic classification/modeling recipe digest mismatch"
            )
        bound_recipe_sha256 = classification_recipe
    fresh = (
        classification["status"] == "completed"
        and modeling["status"] == "completed"
    )
    promoted_paths: list[str] = []
    promoted_part_id: str | None = None
    if fresh:
        staged_root = Path(modeling["staging_artifacts"])
        staged_static = tuple(staged_root.rglob("static.part"))
        if len(staged_static) != 1:
            raise ValueError(
                "completed modeling proposal must contain exactly one static.part"
            )
        staged_part = json.loads(staged_static[0].read_text(encoding="utf-8"))
        promoted_part_id = staged_part.get("id")
        if not isinstance(promoted_part_id, str) or not promoted_part_id:
            raise ValueError("completed modeling proposal has no part id")
        staged_models = tuple(staged_root.rglob("v1.model"))
        if len(staged_models) != 1:
            raise ValueError(
                "completed modeling proposal must contain exactly one v1.model"
            )
        staged_model = json.loads(staged_models[0].read_text(encoding="utf-8"))
        promoted_model_id = staged_model.get("id")
        if not isinstance(promoted_model_id, str) or not promoted_model_id:
            raise ValueError("completed modeling proposal has no model-instance id")
        promoted_paths = [
            str(path)
            for path in modeling_agent.promote(staged_root, catalog_root)
        ]
        from ..catalog.instantiations import PartInstantiationRegistry
        from ..catalog.interfaces import load_interface_catalog
        from ..physics.dsl import ModelRegistry

        interfaces = load_interface_catalog(catalog_root)
        models = ModelRegistry()
        models.load_directory(catalog_root, interfaces=interfaces)
        registry = PartInstantiationRegistry.load_catalog(
            catalog_root, models=models
        )
        if (
            promoted_model_id not in registry
            or registry[promoted_model_id].static.id != promoted_part_id
        ):
            raise RuntimeError(
                "promoted part/model identity is absent from the independently reloaded "
                "isolated catalog"
            )
    return {
        "target": target,
        "status": "completed" if fresh else "resumed_exact_input",
        "preflight": preflight,
        "cost_scope": scope,
        "classification": classification,
        "modeling": modeling,
        "promoted_paths": promoted_paths,
        "promoted_part_id": promoted_part_id,
        "promoted_model_id": promoted_model_id if fresh else None,
        "charged_usd": float(classification["charged_usd"])
        + float(modeling["charged_usd"]),
        "prior_target_charged_usd": float(prior_target_charged_usd),
        "cumulative_target_charged_usd": float(prior_target_charged_usd)
        + float(classification["charged_usd"])
        + float(modeling["charged_usd"]),
        "remaining_part_scope_limit_usd": remaining_scope_limit,
        "validation_activity": modeling.get("validation_activity", {}),
        "host_recipe_sha256": bound_recipe_sha256,
        "fully_ingested": fresh,
    }


def ingestion_metrics(
    results: Iterable[Mapping[str, Any]],
    ledger_snapshot: Mapping[str, Any],
    ingestion_run_id: str,
    *,
    policy: IngestionPolicy = IngestionPolicy(),
    expected_target_count: int | None = None,
) -> dict[str, Any]:
    """Compute inclusive run KPIs from authoritative ledger events.

    Every classification/modeling event in the run contributes to the
    numerator, including invalid outputs, retries, and failed calls. Only fresh
    classification+modeling completions contribute to the denominator.
    """

    records = [dict(item) for item in results]
    events = [
        event
        for event in ledger_snapshot.get("events", [])
        if isinstance(event, Mapping)
        and isinstance(event.get("metadata"), Mapping)
        and event["metadata"].get("ingestion_run_id") == ingestion_run_id
    ]
    total = sum(float(event.get("charged_usd", 0.0)) for event in events)
    completed = [
        item
        for item in records
        if item.get("status") == "completed"
        and item.get("fully_ingested") is True
    ]
    deferred = [
        item
        for item in records
        if item.get("status") == "deferred_unsupported_physics"
    ]
    failed = [
        item
        for item in records
        if item.get("status")
        not in {"completed", "deferred_unsupported_physics"}
    ]
    telemetry_failed_validation_attempts = 0
    for item in records:
        activity = item.get("validation_activity")
        if not isinstance(activity, Mapping):
            modeling = item.get("modeling")
            activity = (
                modeling.get("validation_activity")
                if isinstance(modeling, Mapping)
                else None
            )
        if isinstance(activity, Mapping):
            raw = activity.get("failed_calls", 0)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                telemetry_failed_validation_attempts += raw
    event_failed_validation_attempts = sum(
        1
        for event in events
        if event.get("status") == "invalid_output"
        and str(event["metadata"].get("kind", "")).startswith(
            ("classification", "modeling")
        )
    )
    # Direct Responses has one invalid_output event per rejected classification
    # or failed host modeling validation. CLI workspaces expose richer internal
    # validator telemetry instead.
    failed_validation_attempts = max(
        telemetry_failed_validation_attempts,
        event_failed_validation_attempts,
    )

    count = len(completed)
    cost_per = total / count if count else None
    failed_average = failed_validation_attempts / count if count else None
    scope_breaches = [
        event
        for event in events
        if event.get("scope_limit_breached") is True
        or event.get("status") == "usage_exceeded_reservation"
    ]
    violations: list[str] = []
    if not count:
        violations.append("no freshly fully ingested part")
    if failed:
        violations.append("one or more requested targets did not complete freshly")
    if expected_target_count is not None and len(records) != expected_target_count:
        violations.append("the report omitted one or more requested targets")
    if cost_per is not None and not (
        cost_per < policy.max_cost_per_fully_ingested_part_usd
    ):
        violations.append("cost per fully ingested part is not strictly below the limit")
    if failed_average is not None and not (
        failed_average < policy.max_average_failed_validation_attempts
    ):
        violations.append(
            "average failed validation attempts is not strictly below the limit"
        )
    if scope_breaches:
        violations.append("a provider call exceeded its atomic part reservation")

    phase_costs: dict[str, float] = {}
    for event in events:
        kind = str(event["metadata"].get("kind", "unknown"))
        phase = "classification" if kind.startswith("classification") else "modeling"
        phase_costs[phase] = phase_costs.get(phase, 0.0) + float(
            event.get("charged_usd", 0.0)
        )
    completion_generation_modes: dict[str, int] = {}
    for item in completed:
        modeling = item.get("modeling")
        mode = (
            modeling.get("generation_mode", "provider_response")
            if isinstance(modeling, Mapping)
            else "unknown"
        )
        mode_name = str(mode)
        completion_generation_modes[mode_name] = (
            completion_generation_modes.get(mode_name, 0) + 1
        )
    return {
        "ingestion_run_id": ingestion_run_id,
        "total_importer_charged_usd": total,
        "phase_charged_usd": phase_costs,
        "fully_ingested_parts": count,
        "deferred_unsupported_physics_parts": len(deferred),
        "failed_or_unmeasured_parts": len(failed),
        "reported_targets": len(records),
        "expected_targets": expected_target_count,
        "cost_per_fully_ingested_part_usd": cost_per,
        "failed_validation_attempts": failed_validation_attempts,
        "average_failed_validation_attempts_per_fully_ingested_part": failed_average,
        "provider_calls": len(events),
        "completion_generation_modes": completion_generation_modes,
        "thresholds": asdict(policy),
        "passed": not violations,
        "violations": violations,
    }


def workflow_fingerprint(
    classification_agent: ClassificationAgent,
    modeling_agent: DirectResponsesModelingAgent,
    interface_catalog: Mapping[str, Any],
    *,
    policy: IngestionPolicy = IngestionPolicy(),
) -> str:
    payload = {
        "classification": {
            "model": classification_agent.model,
            "reasoning_effort": classification_agent.reasoning_effort,
            "limits": asdict(classification_agent.limits),
            "schema": CLASSIFICATION_SCHEMA,
            "prompt": classification_agent.system_prompt(dict(interface_catalog)),
        },
        "modeling": {
            "backend": modeling_agent.backend_id,
            "model": modeling_agent.model,
            "reasoning_effort": modeling_agent.reasoning_effort,
            "max_input_tokens": modeling_agent.max_input_tokens,
            "max_output_tokens": modeling_agent.rollout_token_limit,
            "max_validation_attempts": modeling_agent.max_validation_attempts,
            "schema": RESPONSES_MODELING_SCHEMA,
            "prompt": modeling_agent.prompt("<component>.json"),
        },
        "pricing": {
            "classification": asdict(classification_agent.pricing),
            "modeling": asdict(modeling_agent.pricing),
        },
        "host_implementation_sha256": importer_implementation_fingerprint(),
        "policy": asdict(policy),
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def importer_implementation_fingerprint() -> str:
    """Hash host importer, validator, overlay, and catalog-admission source."""

    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parents[1]
    project_metadata = project_root / "pyproject.toml"
    files = {*package_root.rglob("*.py")}
    if project_metadata.is_file() and not project_metadata.is_symlink():
        files.add(project_metadata)
    digest = hashlib.sha256()
    try:
        openai_version = importlib_metadata.version("openai")
    except importlib_metadata.PackageNotFoundError:
        openai_version = "not-installed"
    version_label = f"openai-sdk:{openai_version}".encode("utf-8")
    digest.update(len(version_label).to_bytes(8, "big"))
    digest.update(version_label)
    for path in sorted(files, key=lambda item: item.relative_to(project_root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"importer implementation source is missing or unsafe: {path}")
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def combine_ingestion_metrics(
    *metrics: Mapping[str, Any],
    policy: IngestionPolicy = IngestionPolicy(),
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("at least one ingestion metric set is required")
    total = sum(float(item["total_importer_charged_usd"]) for item in metrics)
    completed = sum(int(item["fully_ingested_parts"]) for item in metrics)
    failed_validations = sum(int(item["failed_validation_attempts"]) for item in metrics)
    provider_calls = sum(int(item.get("provider_calls", 0)) for item in metrics)
    completion_generation_modes: dict[str, int] = {}
    for item in metrics:
        raw_modes = item.get("completion_generation_modes", {})
        if not isinstance(raw_modes, Mapping):
            raise ValueError("completion generation modes must be an object")
        for mode, raw_count in raw_modes.items():
            if (
                not isinstance(mode, str)
                or not mode
                or isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
            ):
                raise ValueError("completion generation modes contain an invalid count")
            completion_generation_modes[mode] = (
                completion_generation_modes.get(mode, 0) + raw_count
            )
    cost_per = total / completed if completed else None
    failed_average = failed_validations / completed if completed else None
    passed = bool(
        completed
        and all(item.get("passed") is True for item in metrics)
        and cost_per is not None
        and cost_per < policy.max_cost_per_fully_ingested_part_usd
        and failed_average is not None
        and failed_average < policy.max_average_failed_validation_attempts
    )
    return {
        "total_importer_charged_usd": total,
        "fully_ingested_parts": completed,
        "failed_validation_attempts": failed_validations,
        "provider_calls": provider_calls,
        "completion_generation_modes": completion_generation_modes,
        "cost_per_fully_ingested_part_usd": cost_per,
        "average_failed_validation_attempts_per_fully_ingested_part": failed_average,
        "thresholds": asdict(policy),
        "passed": passed,
    }


def failed_run_carryover(
    ledger_path: str | Path,
    *,
    expected_target: str,
    expected_component_sha256: str,
) -> dict[str, Any]:
    """Summarize one dedicated, failed ingestion ledger without dispatching.

    The raw ledger remains the evidence. Its exact digest and run/target binding
    are carried into the next clean replay so failed spend and validation calls
    cannot disappear from final KPIs.
    """

    raw_path = Path(ledger_path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("failed-run ledger cannot be a symlink")
    path = raw_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    snapshot = json.loads(payload)
    if not isinstance(snapshot, Mapping):
        raise ValueError("failed-run ledger root must be an object")
    events = snapshot.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("failed-run ledger has no settled events")
    if snapshot.get("reserved") not in ({}, None):
        raise ValueError("failed-run ledger still has active reservations")

    run_ids: set[str] = set()
    component_digests: set[str] = set()
    total = 0.0
    failed_validations = 0
    provider_calls_by_phase: dict[str, int] = {}
    failed_validations_by_phase: dict[str, int] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("failed-run ledger contains a malformed event")
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("failed-run ledger event has no metadata")
        run_id = metadata.get("ingestion_run_id")
        target = metadata.get("target")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("failed-run ledger event has no ingestion run id")
        if target != expected_target:
            raise ValueError("failed-run ledger belongs to another target")
        cost_scope = metadata.get("cost_scope")
        if not isinstance(cost_scope, str) or ":" not in cost_scope:
            raise ValueError("failed-run ledger event has no component-bound cost scope")
        raw_component_digest = cost_scope.rsplit(":", 1)[-1]
        if (
            len(raw_component_digest) != 64
            or any(character not in "0123456789abcdef" for character in raw_component_digest)
        ):
            raise ValueError("failed-run ledger has an invalid component digest")
        component_digests.add("sha256:" + raw_component_digest)
        run_ids.add(run_id)
        charge = event.get("charged_usd")
        if (
            isinstance(charge, bool)
            or not isinstance(charge, (int, float))
            or not math.isfinite(float(charge))
            or float(charge) < 0
        ):
            raise ValueError("failed-run ledger has an invalid charge")
        total += float(charge)
        if event.get("scope_limit_breached") is True:
            raise ValueError("failed-run ledger contains a part-scope breach")
        kind = str(metadata.get("kind", ""))
        phase = (
            "classification"
            if kind.startswith("classification")
            else "modeling" if kind.startswith("modeling") else "unknown"
        )
        provider_calls_by_phase[phase] = provider_calls_by_phase.get(phase, 0) + 1
        if event.get("status") == "invalid_output" and kind.startswith(
            ("classification", "modeling")
        ):
            failed_validations += 1
            failed_validations_by_phase[phase] = (
                failed_validations_by_phase.get(phase, 0) + 1
            )
        if kind.startswith("modeling") and event.get("status") == "completed":
            raise ValueError("carryover ledger contains a completed modeling call")
    if (
        len(run_ids) != 1
        or component_digests != {expected_component_sha256}
        or failed_validations < 1
    ):
        raise ValueError("ledger is not one failed ingestion run")
    spent = snapshot.get("spent_usd")
    if (
        isinstance(spent, bool)
        or not isinstance(spent, (int, float))
        or not math.isfinite(float(spent))
        or abs(float(spent) - total) > 1e-9
    ):
        raise ValueError("failed-run ledger spend does not equal its event charges")
    return {
        "schema": FAILED_RUN_CARRYOVER_SCHEMA,
        "source_ledger": str(path),
        "source_ledger_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "ingestion_run_id": next(iter(run_ids)),
        "target": expected_target,
        "component_sha256": expected_component_sha256,
        "total_importer_charged_usd": total,
        "failed_validation_attempts": failed_validations,
        "provider_calls": len(events),
        "provider_calls_by_phase": provider_calls_by_phase,
        "failed_validation_attempts_by_phase": failed_validations_by_phase,
        "scope_limit_breached": False,
    }


def validate_failed_run_carryovers(
    carryovers: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Re-read and exactly verify every bound failed-run ledger."""

    verified: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for expected in carryovers:
        if expected.get("schema") != FAILED_RUN_CARRYOVER_SCHEMA:
            raise ValueError("failed-run carryover has the wrong schema")
        path = expected.get("source_ledger")
        target = expected.get("target")
        component_sha256 = expected.get("component_sha256")
        if (
            not isinstance(path, str)
            or not isinstance(target, str)
            or not isinstance(component_sha256, str)
        ):
            raise ValueError("failed-run carryover has no ledger/target binding")
        actual = failed_run_carryover(
            path,
            expected_target=target,
            expected_component_sha256=component_sha256,
        )
        if dict(expected) != actual:
            raise ValueError("failed-run ledger changed after the canary")
        identity = (actual["source_ledger_sha256"], actual["ingestion_run_id"])
        if identity in seen:
            raise ValueError("duplicate failed-run carryover")
        seen.add(identity)
        verified.append(actual)
    return verified


def validate_matching_failed_run_carryovers(
    report_values: Iterable[Mapping[str, Any]],
    bound_values: Iterable[Mapping[str, Any]],
    supplied_values: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require report, isolated binding, and batch CLI evidence to match exactly."""

    report = validate_failed_run_carryovers(report_values)
    bound = validate_failed_run_carryovers(bound_values)
    supplied = validate_failed_run_carryovers(supplied_values)
    if report != bound or report != supplied:
        raise ValueError(
            "prior failed-run report, isolated binding, and batch arguments differ"
        )
    return report


def combine_ingestion_metrics_with_carryovers(
    canary_metrics: Mapping[str, Any],
    batch_metrics: Mapping[str, Any],
    carryovers: Iterable[Mapping[str, Any]],
    *,
    policy: IngestionPolicy = IngestionPolicy(),
) -> dict[str, Any]:
    """Compute final KPIs including failed attempts from earlier clean replays."""

    base = combine_ingestion_metrics(canary_metrics, batch_metrics, policy=policy)
    prior = [dict(item) for item in carryovers]
    if any(item.get("scope_limit_breached") is True for item in prior):
        raise ValueError("failed-run carryover contains a part-scope breach")
    prior_cost = sum(float(item["total_importer_charged_usd"]) for item in prior)
    prior_failures = sum(int(item["failed_validation_attempts"]) for item in prior)
    prior_provider_calls = sum(int(item.get("provider_calls", 0)) for item in prior)
    prior_failed_by_phase: dict[str, int] = {}
    prior_calls_by_phase: dict[str, int] = {}
    for item in prior:
        for source_key, destination in (
            ("failed_validation_attempts_by_phase", prior_failed_by_phase),
            ("provider_calls_by_phase", prior_calls_by_phase),
        ):
            raw_counts = item.get(source_key, {})
            if not isinstance(raw_counts, Mapping):
                raise ValueError(f"carryover {source_key} must be an object")
            for phase, raw_count in raw_counts.items():
                if (
                    not isinstance(phase, str)
                    or isinstance(raw_count, bool)
                    or not isinstance(raw_count, int)
                    or raw_count < 0
                ):
                    raise ValueError(f"carryover {source_key} is invalid")
                destination[phase] = destination.get(phase, 0) + raw_count
    completed = int(base["fully_ingested_parts"])
    total = float(base["total_importer_charged_usd"]) + prior_cost
    failed = int(base["failed_validation_attempts"]) + prior_failures
    cost_per = total / completed if completed else None
    failed_average = failed / completed if completed else None
    passed = bool(
        base["passed"] is True
        and cost_per is not None
        and cost_per < policy.max_cost_per_fully_ingested_part_usd
        and failed_average is not None
        and failed_average < policy.max_average_failed_validation_attempts
    )
    return {
        "total_importer_charged_usd": total,
        "carryover_charged_usd": prior_cost,
        "fully_ingested_parts": completed,
        "failed_validation_attempts": failed,
        "carryover_failed_validation_attempts": prior_failures,
        "provider_calls": int(base.get("provider_calls", 0)) + prior_provider_calls,
        "current_replay_provider_calls": int(base.get("provider_calls", 0)),
        "carryover_provider_calls": prior_provider_calls,
        "completion_generation_modes": dict(
            base.get("completion_generation_modes", {})
        ),
        "backend_outcomes": {
            "prior_provider_failed_validation_attempts": prior_failures,
            "prior_provider_failed_validation_attempts_by_phase": prior_failed_by_phase,
            "prior_provider_calls_by_phase": prior_calls_by_phase,
            "current_replay_provider_calls": int(base.get("provider_calls", 0)),
            "current_replay_completion_generation_modes": dict(
                base.get("completion_generation_modes", {})
            ),
        },
        "cost_per_fully_ingested_part_usd": cost_per,
        "average_failed_validation_attempts_per_fully_ingested_part": failed_average,
        "prior_failed_runs": len(prior),
        "thresholds": asdict(policy),
        "passed": passed,
    }


def build_ingestion_report(
    *,
    mode: str,
    results: Iterable[Mapping[str, Any]],
    ledger: BudgetLedger,
    ingestion_run_id: str,
    workflow_sha256: str,
    job_file_sha256: str,
    policy: IngestionPolicy = IngestionPolicy(),
    expected_target_count: int | None = None,
) -> dict[str, Any]:
    records = [dict(item) for item in results]
    return {
        "schema": INGESTION_REPORT_SCHEMA,
        "mode": mode,
        "ingestion_run_id": ingestion_run_id,
        "workflow_sha256": workflow_sha256,
        "job_file_sha256": job_file_sha256,
        "results": records,
        "metrics": ingestion_metrics(
            records,
            ledger.snapshot(),
            ingestion_run_id,
            policy=policy,
            expected_target_count=expected_target_count,
        ),
    }


def validate_canary_report_evidence(
    report: Mapping[str, Any],
    *,
    ledger_snapshot: Mapping[str, Any],
    catalog_root: str | Path,
    expected_target: str,
) -> None:
    """Recompute a canary gate from its ledger and promoted catalog state."""

    run_id = report.get("ingestion_run_id")
    results = report.get("results")
    recorded_metrics = report.get("metrics")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(results, list)
        or len(results) != 1
        or not isinstance(recorded_metrics, Mapping)
    ):
        raise ValueError("canary report lacks authoritative run evidence")
    result = results[0]
    if (
        not isinstance(result, Mapping)
        or result.get("target") != expected_target
        or result.get("status") != "completed"
        or result.get("fully_ingested") is not True
        or result.get("promoted_part_id") != expected_target
        or not isinstance(result.get("promoted_model_id"), str)
    ):
        raise ValueError("canary report does not record one fresh promoted target")
    for event in ledger_snapshot.get("events", []):
        if not isinstance(event, Mapping):
            continue
        metadata = event.get("metadata")
        if (
            isinstance(metadata, Mapping)
            and metadata.get("ingestion_run_id") == run_id
            and metadata.get("target") != expected_target
        ):
            raise ValueError("canary ledger contains an event for another target")
    recomputed = ingestion_metrics(
        results,
        ledger_snapshot,
        run_id,
        expected_target_count=1,
    )
    if dict(recorded_metrics) != recomputed:
        raise ValueError("canary metrics differ from replay-ledger recomputation")

    from ..catalog.instantiations import PartInstantiationRegistry
    from ..catalog.interfaces import load_interface_catalog
    from ..physics.dsl import ModelRegistry

    catalog = Path(catalog_root)
    interfaces = load_interface_catalog(catalog)
    models = ModelRegistry()
    models.load_directory(catalog, interfaces=interfaces)
    registry = PartInstantiationRegistry.load_catalog(catalog, models=models)
    model_id = result["promoted_model_id"]
    if model_id not in registry or registry[model_id].static.id != expected_target:
        raise ValueError("canary promoted target is absent from the isolated catalog")


def validate_canary_gate(
    report: Mapping[str, Any],
    *,
    workflow_sha256: str,
    job_file_sha256: str,
) -> None:
    if report.get("schema") != INGESTION_REPORT_SCHEMA or report.get("mode") != "canary":
        raise ValueError("canary report has the wrong schema or mode")
    if report.get("workflow_sha256") != workflow_sha256:
        raise ValueError("canary report was produced by a different importer workflow")
    if report.get("job_file_sha256") != job_file_sha256:
        raise ValueError("canary report was produced from a different job inventory")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("passed") is not True:
        raise ValueError("canary report did not pass the inclusive ingestion KPIs")
    if metrics.get("fully_ingested_parts") != 1:
        raise ValueError("canary report must contain exactly one fresh completed part")
