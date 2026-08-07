"""Command-line entry points for validation, demos, compilation, and agents."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

from .agents import (
    ClassificationAgent,
    ModelingAgent,
    ModelingInputs,
    load_dotenv_key,
    run_classification_batch,
    run_modeling_proposal,
    write_json_atomic,
)
from .backend import infer_backend
from .budget import BudgetLedger
from .build import generate_build_instructions
from .compiler import OnlineModelIR, compile_contraption, syntax_check
from .controls import ControlProgram
from .dsl import ModelRegistry
from .paths import asset_root, source_root
from .scanner import (
    load_scanner_mission,
    scanner_metrics,
    simulate_scanner_robot,
    validate_scanner_simulation_coverage,
)
from .specs import ContraptionSpec
from .taxonomy import load_default_taxonomy
from .validation import validate_contraption_structure
from .visualization import generate_viewer


PROJECT_ROOT = asset_root()
SOURCE_ROOT = source_root()
WORK_ROOT = SOURCE_ROOT if PROJECT_ROOT == SOURCE_ROOT else Path.cwd().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("CONTRAPTION_OUTPUT_ROOT", str(WORK_ROOT / "outputs"))
).expanduser().resolve()
SCANNER_ROOT = PROJECT_ROOT / "examples" / "scanner_robot"
DEFAULT_OUTPUT = OUTPUT_ROOT / "scanner_demo"
AGENT_PROPOSALS = OUTPUT_ROOT / "agent-proposals"
AGENT_STAGING = OUTPUT_ROOT / "agent-staging"
SCANNER_AGENT_TARGETS = (
    "camera_compute",
    "power",
    "romi_arm",
    "romi_control",
    "romi_drive",
    "romi_encoders",
)

_MODELING_CONTEXT: dict[str, tuple[str, ...]] = {
    "camera_compute": ("mechanical/camera_mass.pmdl",),
    "power": ("electrical/voltage_source.pmdl", "electrical/capacitor.pmdl"),
    "romi_arm": ("electrical/dc_motor.pmdl", "mechanical/revolute_joint.pmdl"),
    "romi_control": ("electrical/voltage_source.pmdl",),
    "romi_drive": ("electrical/dc_motor.pmdl", "mechanical/wheel_contact.pmdl"),
    "romi_encoders": ("mechanical/revolute_joint.pmdl", "mechanical/wheel_contact.pmdl"),
}


def _default_dotenv_path() -> Path:
    """Resolve the canonical local dotenv without silently choosing ambiguity."""

    candidates = (WORK_ROOT / ".env", WORK_ROOT.parent / ".env")
    existing = tuple(path for path in candidates if path.is_file())
    if len(existing) > 1:
        joined = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"multiple dotenv files contain potentially conflicting configuration: {joined}; "
            "pass --env-file explicitly"
        )
    return existing[0] if existing else candidates[0]


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: Any) -> Path:
    return write_json_atomic(path, value)


def _model_report() -> tuple[ModelRegistry, list[dict[str, Any]]]:
    registry = ModelRegistry()
    reports: list[dict[str, Any]] = []
    for path in sorted((PROJECT_ROOT / "models").rglob("*.pmdl")):
        try:
            model = registry.load(path)
            reports.append({"path": str(path), "id": model.id, "valid": True})
        except Exception as exc:
            reports.append(
                {"path": str(path), "valid": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    return registry, reports


def _torch_diagnostics() -> dict[str, Any]:
    result: dict[str, Any] = {
        "installed": False,
        "version": None,
        "compiled_cuda_runtime": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_selected_device_index": None,
        "cuda_selected_device_name": None,
    }
    try:
        result["installed"] = importlib.util.find_spec("torch") is not None
    except Exception as exc:
        result["discovery_error"] = f"{type(exc).__name__}: {exc}"
        return result
    if not result["installed"]:
        return result
    try:
        import torch  # type: ignore
    except Exception as exc:
        result["import_error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        result["version"] = str(torch.__version__)
        result["compiled_cuda_runtime"] = getattr(torch.version, "cuda", None)
    except Exception as exc:
        result["metadata_error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        if result["cuda_available"] and result["cuda_device_count"] > 0:
            selected = int(torch.cuda.current_device())
            result["cuda_selected_device_index"] = selected
            result["cuda_selected_device_name"] = str(
                torch.cuda.get_device_name(selected)
            )
    except Exception as exc:
        result["cuda_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return result


def command_doctor(_args: argparse.Namespace) -> int:
    dotenv_path = _default_dotenv_path()
    torch = _torch_diagnostics()
    result = {
        "asset_root": str(PROJECT_ROOT),
        "work_root": str(WORK_ROOT),
        "python": sys.version.split()[0],
        "numpy": importlib.util.find_spec("numpy") is not None,
        "torch_optional": torch["installed"],
        "torch": torch,
        "openai_agents_optional": importlib.util.find_spec("openai") is not None,
        "openai_key_available": bool(os.environ.get("OPENAI_API_KEY") or load_dotenv_key(dotenv_path)),
        "dotenv_path": str(dotenv_path),
        "dotenv_exists": dotenv_path.is_file(),
        "codex_cli": shutil.which(os.environ.get("CODEX_BIN", "codex")),
        "c99_compiler": next(
            (shutil.which(name) for name in ("cc", "gcc", "clang") if shutil.which(name)),
            None,
        ),
        "budget_limit_usd": 100.0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_validate(_args: argparse.Namespace) -> int:
    taxonomy = load_default_taxonomy()
    registry, models = _model_report()
    raw_spec = _load_json(SCANNER_ROOT / "contraption.json")
    spec = ContraptionSpec.from_dict(raw_spec)
    structural = validate_contraption_structure(spec)
    coverage_digest = validate_scanner_simulation_coverage(
        spec, _load_json(SCANNER_ROOT / "simulation_coverage.json")
    )
    program = ControlProgram.from_dict(_load_json(SCANNER_ROOT / "controls" / "scanner.control.json"))
    ir = OnlineModelIR.from_dict(_load_json(SCANNER_ROOT / "online_model.json"))
    report = {
        "taxonomy": {
            "domains": len(taxonomy.domains),
            "categories": len(taxonomy.categories),
            "subcategories": len(taxonomy.subcategories),
            "instantiations": len(taxonomy.instantiations),
        },
        "models": models,
        "model_count": len(registry),
        "contraption": {
            "scope": "structural_and_component_references_only",
            **structural.to_dict(),
        },
        "controller": {"name": program.name, "states": [item.name for item in program.states]},
        "online_ir": {
            "states": len(ir.state_names),
            "inputs": len(ir.input_names),
            "measurements": len(ir.measurement_names),
        },
        "simulation_coverage": {
            "valid": True,
            "topology_sha256": coverage_digest,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if structural.valid and all(item["valid"] for item in models) else 1


def _trajectory_payload(result: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    numerical = infer_backend(result.samples)
    return {
        "schema": "contraption.trajectory/v1",
        "time": numerical.to_numpy(result.time).tolist(),
        "state_names": list(result.state_names),
        "output_names": list(result.output_names),
        "mean": numerical.to_numpy(result.mean).tolist(),
        "state_distribution": result.summary.to_dict(),
        "output_distribution": result.output_summary.to_dict(),
        "metadata": dict(result.metadata),
        "metrics": dict(metrics),
    }


def command_demo(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_spec = _load_json(SCANNER_ROOT / "contraption.json")
    spec = ContraptionSpec.from_dict(raw_spec)
    validation = validate_contraption_structure(spec)
    validation.require_valid()
    program = ControlProgram.from_dict(_load_json(SCANNER_ROOT / "controls" / "scanner.control.json"))
    mission = load_scanner_mission(SCANNER_ROOT / "scanner_parameters.json")
    simulation_coverage = _load_json(SCANNER_ROOT / "simulation_coverage.json")
    duration = float(args.duration) if args.duration is not None else (10.0 if args.quick else None)
    # With zero observed collisions, roughly 2,703 samples are needed before a
    # one-sided 95% Wilson upper bound can demonstrate probability below 0.001.
    # 128 keeps the fixed-seed quick motion metrics stable on both NumPy and
    # Torch while remaining explicitly underpowered for the collision bound.
    samples = int(args.samples) if args.samples is not None else (128 if args.quick else 4096)
    result = simulate_scanner_robot(
        program,
        mission,
        physical_spec=spec,
        simulation_coverage=simulation_coverage,
        duration=duration,
        dt=float(args.dt),
        num_samples=samples,
        seed=int(args.seed),
        backend=args.backend,
        device=args.device,
    )
    metrics = scanner_metrics(result, mission)
    trajectory = _trajectory_payload(result, metrics)
    trajectory_path = _write_json(output / "trajectory.json", trajectory)

    online_data = _load_json(SCANNER_ROOT / "online_model.json")
    online_ir = OnlineModelIR.from_dict(online_data)
    artifact = compile_contraption(
        raw_spec,
        assembled_system=online_data,
        model_name="scanner_online",
    )
    compiled_paths = artifact.write(output / "online")
    syntax = syntax_check(artifact)

    build_plan = generate_build_instructions(raw_spec, output / "build")
    viewer = generate_viewer(
        raw_spec,
        trajectory,
        output / "viewer",
        title="Apartment scanner robot — probabilistic mission",
        runtime_model=online_ir,
    )
    motion_smoke_passed = bool(
        metrics["acceptance"]["orbit_radius_rmse"]
        and metrics["acceptance"]["camera_pointing_p95"]
        and metrics["collision_probability"] <= 0.001
    )
    report = {
        "schema": "contraption.demo-report/v1",
        "accepted": bool(metrics["accepted"]),
        "acceptance_scope": "in_silico_motion_with_statistical_collision_bound",
        "quick_smoke": bool(args.quick),
        "quick_smoke_passed": bool(args.quick and motion_smoke_passed),
        "physical_build_ready": len(build_plan.unresolved) == 0,
        "deployment_ready": False,
        "metrics": metrics,
        "validation": {
            "scope": "structural_and_component_references_only",
            **validation.to_dict(),
        },
        "build": {
            "bom_items": len(build_plan.bill_of_materials),
            "steps": len(build_plan.steps),
            "unresolved_safety_gate_items": len(build_plan.unresolved),
        },
        "online_compiler": {
            "model": artifact.model_name,
            "files": {name: str(path) for name, path in compiled_paths.items()},
            "c99_syntax_checked": syntax.ok,
            "compiler": syntax.compiler,
            "stderr": syntax.stderr,
        },
        "artifacts": {
            "trajectory": str(trajectory_path),
            "viewer": str(output / "viewer" / "index.html"),
            "build_plan": str(output / "build" / "BUILD_INSTRUCTIONS.md"),
            "online_manifest": str(output / "online" / "scanner_online.manifest.json"),
        },
    }
    report_path = _write_json(output / "report.json", report)
    print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
    return 0 if report["accepted"] or report["quick_smoke_passed"] else 1


def _agent_ledger(path: str | None) -> BudgetLedger:
    return BudgetLedger(
        Path(path).resolve() if path else OUTPUT_ROOT / "agent-budget.json",
        limit_usd=100.0,
    )


def command_budget(args: argparse.Namespace) -> int:
    print(json.dumps(_agent_ledger(args.ledger).snapshot(), indent=2, sort_keys=True))
    return 0


def command_agent_canary(args: argparse.Namespace) -> int:
    ledger = _agent_ledger(args.ledger)
    key, dotenv = _agent_key(args.env_file)
    results: dict[str, Any] = {}
    successes = 0
    if args.kind in {"both", "classification"}:
        if not key:
            results["classification"] = {
                "status": "blocked",
                "reason": f"OPENAI_API_KEY is absent and {dotenv} has no key",
            }
        else:
            try:
                component = _load_json(SCANNER_ROOT / "component_inputs" / "romi_drive.json")
                taxonomy = _load_json(PROJECT_ROOT / "data" / "taxonomy.json")
                value, usage, charged = ClassificationAgent(ledger, api_key=key).classify(
                    component, taxonomy, canary=True
                )
                results["classification"] = {
                    "status": "completed",
                    "result": value,
                    "usage": usage.__dict__,
                    "charged_usd": charged,
                }
                successes += 1
            except Exception as exc:
                results["classification"] = {
                    "status": "failed",
                    "reason": _agent_failure_reason(exc, key),
                }
    if args.kind in {"both", "modeling"}:
        try:
            inputs = ModelingInputs(
                constraints=PROJECT_ROOT / "prompts" / "model_constraints.md",
                gold_templates=(PROJECT_ROOT / "models" / "electrical" / "resistor.pmdl",),
                taxonomy=PROJECT_ROOT / "data" / "taxonomy.json",
                direct_hierarchy=(
                    PROJECT_ROOT / "models" / "mechanical" / "camera_mass.pmdl",
                ),
                component_information=(
                    SCANNER_ROOT / "component_inputs" / "camera_compute.json"
                ),
            )
            agent = ModelingAgent(
                ledger,
                AGENT_STAGING,
                codex_binary=os.environ.get("CODEX_BIN"),
                api_key=key,
            )
            artifacts, value, charged = agent.run(inputs, canary=True)
            results["modeling"] = {
                "status": "completed",
                "artifacts": str(artifacts),
                "proposal": value,
                "charged_usd": charged,
            }
            successes += 1
        except Exception as exc:
            results["modeling"] = {
                "status": "failed",
                "reason": _agent_failure_reason(exc, key),
            }
    results["budget"] = ledger.snapshot()
    destination = OUTPUT_ROOT / "agent-canary-report.json"
    _write_json(destination, results)
    print(json.dumps({"report": str(destination), **results}, indent=2, sort_keys=True))
    return 0 if successes else 2


def _agent_key(env_file: str | None) -> tuple[str | None, Path]:
    dotenv = Path(env_file).expanduser().resolve() if env_file else _default_dotenv_path()
    return os.environ.get("OPENAI_API_KEY") or load_dotenv_key(dotenv), dotenv


def _agent_failure_reason(exc: Exception, key: str | None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message.replace(key, "[REDACTED_OPENAI_API_KEY]") if key else message


def _full_modeling_inputs(target: str) -> ModelingInputs:
    if target not in SCANNER_AGENT_TARGETS:
        raise ValueError(f"unknown scanner agent target {target!r}")
    model_root = PROJECT_ROOT / "models"
    return ModelingInputs(
        constraints=PROJECT_ROOT / "prompts" / "model_constraints.md",
        gold_templates=(
            model_root / "electrical" / "resistor.pmdl",
            model_root / "mechanical" / "rigid_body_planar.pmdl",
        ),
        taxonomy=PROJECT_ROOT / "data" / "taxonomy.json",
        direct_hierarchy=tuple(model_root / relative for relative in _MODELING_CONTEXT[target]),
        component_information=SCANNER_ROOT / "component_inputs" / f"{target}.json",
    )


def command_agent_run(args: argparse.Namespace) -> int:
    """Run paid, non-canary jobs with resumable validated proposal receipts."""

    ledger = _agent_ledger(args.ledger)
    key, dotenv = _agent_key(args.env_file)
    proposal_root = Path(args.output_root).resolve()
    try:
        if args.agent_job == "classification-all":
            if not key:
                raise RuntimeError(
                    f"OPENAI_API_KEY is absent and {dotenv} has no key; "
                    "classification-all was not dispatched"
                )
            component_paths = tuple(
                SCANNER_ROOT / "component_inputs" / f"{target}.json"
                for target in SCANNER_AGENT_TARGETS
            )
            results = run_classification_batch(
                ClassificationAgent(ledger, api_key=key),
                component_paths,
                PROJECT_ROOT / "data" / "taxonomy.json",
                proposal_root / "classification",
                force=args.force,
            )
            payload = {
                "job": "classification-all",
                "results": results,
                "budget": ledger.snapshot(),
            }
        elif args.agent_job == "modeling-one":
            agent = ModelingAgent(
                ledger,
                Path(args.staging_root).resolve(),
                codex_binary=os.environ.get("CODEX_BIN"),
                api_key=key,
            )
            result = run_modeling_proposal(
                agent,
                _full_modeling_inputs(args.target),
                args.target,
                proposal_root / "modeling",
                force=args.force,
            )
            payload = {
                "job": "modeling-one",
                "result": result,
                "budget": ledger.snapshot(),
            }
        else:  # pragma: no cover - argparse enforces this boundary.
            raise ValueError(f"unsupported agent job {args.agent_job!r}")
    except Exception as exc:
        raise RuntimeError(_agent_failure_reason(exc, key)) from None
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contraption")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="report optional runtime/tool availability").set_defaults(
        handler=command_doctor
    )
    commands.add_parser(
        "validate",
        help="validate PMDLs plus scanner manifest structure/component references",
    ).set_defaults(
        handler=command_validate
    )
    demo = commands.add_parser("demo", help="run the scanner mission and generate artifacts")
    demo.add_argument("--output", default=str(DEFAULT_OUTPUT))
    demo.add_argument("--quick", action="store_true")
    demo.add_argument("--duration", type=float)
    demo.add_argument("--samples", type=int)
    demo.add_argument("--dt", type=float, default=0.05)
    demo.add_argument("--seed", type=int, default=20260806)
    demo.add_argument("--backend", choices=("numpy", "torch", "auto"), default="numpy")
    demo.add_argument("--device")
    demo.set_defaults(handler=command_demo)
    budget = commands.add_parser("budget", help="show the hard agent-dollar ledger")
    budget.add_argument("--ledger")
    budget.set_defaults(handler=command_budget)
    canary = commands.add_parser("agent-canary", help="run guarded classification/modeling canaries")
    canary.add_argument("--kind", choices=("both", "classification", "modeling"), default="both")
    canary.add_argument("--ledger")
    canary.add_argument("--env-file")
    canary.set_defaults(handler=command_agent_canary)
    actual = commands.add_parser(
        "agent-run",
        help="run paid, resumable classification or modeling jobs without promotion",
    )
    actual_jobs = actual.add_subparsers(dest="agent_job", required=True)

    def actual_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--ledger")
        command.add_argument("--env-file")
        command.add_argument("--output-root", default=str(AGENT_PROPOSALS))
        command.add_argument(
            "--force",
            action="store_true",
            help="dispatch again even when the exact input hash already completed",
        )
        command.set_defaults(handler=command_agent_run)

    classify_all = actual_jobs.add_parser(
        "classification-all", help="classify all six scanner component input records"
    )
    actual_common(classify_all)
    modeling_one = actual_jobs.add_parser(
        "modeling-one", help="stage one full modeling proposal without promoting it"
    )
    actual_common(modeling_one)
    modeling_one.add_argument("--target", choices=SCANNER_AGENT_TARGETS, default="romi_drive")
    modeling_one.add_argument("--staging-root", default=str(AGENT_STAGING))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
