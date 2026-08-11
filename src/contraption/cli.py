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

import numpy as np

from .applications.scanner import (
    ScannerMission,
    scanner_metrics,
    simulate_scanner_robot,
)
from .catalog.instantiations import PartInstantiationRegistry
from .catalog.interfaces import interface_paths, load_interface_catalog
from .manufacturing.build import generate_build_instructions
from .part_import.agents import (
    ClassificationAgent,
    ModelingAgent,
    ModelingInputs,
    load_dotenv_key,
    run_classification_batch,
    run_modeling_proposal,
    write_json_atomic,
)
from .part_import.budget import BudgetLedger
from .paths import asset_root, source_root
from .physics.backend import infer_backend
from .physics.compiler import compile_resolved_assembly, syntax_check
from .physics.controls import ControlProgram
from .physics.dsl import ModelRegistry
from .physics.resolved import ResolvedAssembly, resolve_assembly
from .physics.simulator import SimulationResult
from .physics.uq import summarize_samples
from .visualization.scanner_scene import scanner_physical_scene
from .visualization.server import LiveScannerApplication, serve_live_scanner
from .visualization.viewer import generate_viewer


PROJECT_ROOT = asset_root()
SOURCE_ROOT = source_root()
WORK_ROOT = SOURCE_ROOT if PROJECT_ROOT == SOURCE_ROOT else Path.cwd().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("CONTRAPTION_OUTPUT_ROOT", str(WORK_ROOT / "outputs"))
).expanduser().resolve()
SCANNER_ROOT = PROJECT_ROOT / "examples" / "scanner_robot"
PART_IMPORT_CANARY_ROOT = PROJECT_ROOT / "examples" / "part_import_canary"
DEFAULT_SPEC = SCANNER_ROOT / "contraption.json"
DEFAULT_CATALOG = PROJECT_ROOT / "model_catalog"
DEFAULT_CONTROLLER_ROOT = SCANNER_ROOT / "controls"
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
    "camera_compute": ("mechanical/inert_objects/camera_masses/camera_mass.pmdl",),
    "power": ("electrical/voltage_sources/voltage_source.pmdl", "electrical/capacitors/ceramic_capacitors/capacitor.pmdl"),
    "romi_arm": ("electromechanical/motors/brushed_dc_motors/dc_motor.pmdl", "mechanical/joints/revolute_joints/revolute_joint.pmdl"),
    "romi_control": ("electrical/voltage_sources/voltage_source.pmdl",),
    "romi_drive": ("electromechanical/motors/brushed_dc_motors/dc_motor.pmdl", "mechanical/wheels/driven_wheels/wheel_contact.pmdl"),
    "romi_encoders": ("mechanical/joints/revolute_joints/revolute_joint.pmdl", "mechanical/wheels/driven_wheels/wheel_contact.pmdl"),
}


def _controller_identity(assembly: ResolvedAssembly) -> dict[str, str] | None:
    reference = assembly.specification.controller
    if reference is None:
        return None
    return {
        name: str(reference[name]) for name in ("id", "version", "sha256")
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


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: Any) -> Path:
    return write_json_atomic(path, value)


def _load_resolved_assembly(
    specification_path: str | Path = DEFAULT_SPEC,
    catalog_path: str | Path = DEFAULT_CATALOG,
    controller_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> ResolvedAssembly:
    """Load and verify one canonical contraption/catalog/PMDL closure."""

    catalog_root = Path(catalog_path).expanduser().resolve()
    interfaces = load_interface_catalog(catalog_root)
    registry = ModelRegistry()
    registry.load_directory(catalog_root, interfaces=interfaces)
    instantiations = PartInstantiationRegistry.load_catalog(catalog_root, models=registry)
    controller_roots = tuple(
        Path(path).expanduser().resolve()
        for path in (controller_paths or (DEFAULT_CONTROLLER_ROOT,))
    )
    controllers: dict[str, ControlProgram] = {}
    for root in controller_roots:
        if root.is_dir():
            paths = sorted(root.rglob("*.json"))
        elif root.is_file() and root.suffix == ".json":
            paths = [root]
        else:
            raise FileNotFoundError(f"controller path does not exist: {root}")
        if not paths:
            raise FileNotFoundError(f"controller path contains no JSON files: {root}")
        for path in paths:
            program = ControlProgram.from_dict(_load_json(path))
            if program.name in controllers:
                raise ValueError(
                    f"controller id {program.name!r} is duplicated in the controller registry"
                )
            controllers[program.name] = program
    specification = _load_json(Path(specification_path).expanduser().resolve())
    return resolve_assembly(
        specification,
        instantiations,
        registry,
        control_programs=controllers,
    )


def _assembly_from_args(args: argparse.Namespace) -> ResolvedAssembly:
    return _load_resolved_assembly(
        args.spec,
        args.catalog,
        args.controller_root,
    )


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


def command_validate(args: argparse.Namespace) -> int:
    interfaces = load_interface_catalog(args.catalog)
    assembly = _assembly_from_args(args)
    dynamics_completeness = assembly.dynamics_completeness
    report = {
        "valid": True,
        "scope": "canonical_catalog_instantiation_physical_and_pmdl_closure",
        "interfaces": {
            "domains": len(interfaces.domains),
            "categories": len(interfaces.categories),
            "devices": len(interfaces.devices),
        },
        "assembly": assembly.diagnostics(),
        "controller": _controller_identity(assembly),
        "dynamics_completeness": dynamics_completeness.to_dict(),
        "parts": {
            "registered": len(assembly.parts),
            "used": len({item.part for item in assembly.specification.components}),
        },
        "artifact_closure": {
            "simulation": assembly.assembly_sha256,
            "visualization": assembly.assembly_sha256,
            "build": assembly.assembly_sha256,
            "compiler": assembly.assembly_sha256,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _trajectory_payload(result: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, SimulationResult):
        raise TypeError("trajectory serialization requires an actual SimulationResult")
    numerical = infer_backend(result.samples)
    return {
        "schema": "contraption.trajectory/v2",
        "time": numerical.to_numpy(result.time).tolist(),
        "state_names": list(result.state_names),
        "samples": numerical.to_numpy(result.samples).tolist(),
        "output_names": list(result.output_names),
        "output_samples": numerical.to_numpy(result.output_samples).tolist(),
        "metadata": dict(result.metadata),
        "metrics": dict(metrics),
    }


def _trajectory_result(
    assembly: ResolvedAssembly, trajectory_path: str | Path
) -> SimulationResult:
    """Reconstruct an exact ensemble artifact for canonical pose resolution."""

    trajectory = _load_json(trajectory_path)
    required = {
        "schema",
        "time",
        "state_names",
        "samples",
        "output_names",
        "output_samples",
        "metadata",
        "metrics",
    }
    if set(trajectory) != required:
        raise ValueError(
            "trajectory must contain exactly the v2 provenance fields; "
            f"missing={sorted(required - set(trajectory))}, "
            f"unknown={sorted(set(trajectory) - required)}"
        )
    if trajectory["schema"] != "contraption.trajectory/v2":
        raise ValueError(
            "viewer requires contraption.trajectory/v2 with exact per-sample states; "
            f"got {trajectory['schema']!r}"
        )
    metadata = trajectory["metadata"]
    metrics = trajectory["metrics"]
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise ValueError("trajectory.metadata must be an object with string keys")
    if "body_pose_frames" in metadata:
        raise ValueError(
            "trajectory.metadata.body_pose_frames is a forbidden redundant physical "
            "representation; poses are reconstructed from exact samples"
        )
    if not isinstance(metrics, Mapping) or any(
        not isinstance(key, str) for key in metrics
    ):
        raise ValueError("trajectory.metrics must be an object with string keys")
    state_names_value = trajectory["state_names"]
    output_names_value = trajectory["output_names"]
    if not isinstance(state_names_value, list) or any(
        not isinstance(name, str) or not name for name in state_names_value
    ):
        raise ValueError("trajectory.state_names must be an array of non-empty strings")
    if not isinstance(output_names_value, list) or any(
        not isinstance(name, str) or not name for name in output_names_value
    ):
        raise ValueError("trajectory.output_names must be an array of non-empty strings")
    state_names = tuple(state_names_value)
    output_names = tuple(output_names_value)
    if len(set(state_names)) != len(state_names):
        raise ValueError("trajectory.state_names must be unique")
    if len(set(output_names)) != len(output_names):
        raise ValueError("trajectory.output_names must be unique")
    if state_names != tuple(assembly.system.state_names):
        raise ValueError("trajectory.state_names do not exactly match the resolved assembly")
    try:
        time = np.asarray(trajectory["time"], dtype=np.float64)
        samples = np.asarray(trajectory["samples"], dtype=np.float64)
        output_samples = np.asarray(trajectory["output_samples"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trajectory arrays must be numeric: {exc}") from exc
    if time.ndim != 1 or len(time) < 1:
        raise ValueError("trajectory.time must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(time)) or abs(float(time[0])) > 1e-12 or np.any(
        np.diff(time) <= 0.0
    ):
        raise ValueError(
            "trajectory.time must be finite, start at zero, and increase strictly"
        )
    if (
        samples.ndim != 3
        or samples.shape[0] < 1
        or samples.shape[1:] != (len(time), len(state_names))
    ):
        raise ValueError(
            "trajectory.samples must have shape [sample,time,state] matching its declarations"
        )
    expected_output_shape = (samples.shape[0], len(time), len(output_names))
    if output_samples.ndim != 3 or output_samples.shape != expected_output_shape:
        raise ValueError(
            "trajectory.output_samples must have shape [sample,time,output] "
            "matching its declarations"
        )
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(output_samples)):
        raise ValueError("trajectory samples must contain only finite values")
    if metadata.get("assembly_sha256") != assembly.assembly_sha256:
        raise ValueError("trajectory assembly hash does not match the resolved assembly")
    if metadata.get("pmdl_sha256") != assembly.system.pmdl_sha256:
        raise ValueError("trajectory PMDL hash does not match the resolved assembly")
    declared_samples = metadata.get("sample_count")
    if (
        isinstance(declared_samples, bool)
        or not isinstance(declared_samples, int)
        or declared_samples != samples.shape[0]
    ):
        raise ValueError("trajectory metadata.sample_count does not match its sample axis")
    return SimulationResult(
        time=time,
        state_names=state_names,
        samples=samples,
        output_names=output_names,
        output_samples=output_samples,
        summary=summarize_samples(samples, backend="numpy"),
        output_summary=summarize_samples(output_samples, backend="numpy"),
        metadata=dict(metadata),
    )


def _simulate_and_write(
    assembly: ResolvedAssembly,
    args: argparse.Namespace,
    output: Path,
) -> tuple[Any, ScannerMission, Mapping[str, Any], Path, Path, Mapping[str, Any]]:
    mission = ScannerMission.from_assembly(assembly)
    result = simulate_scanner_robot(
        assembly,
        mission,
        duration=args.duration,
        dt=float(args.dt),
        num_samples=int(args.samples),
        seed=int(args.seed),
        backend=args.backend,
        device=args.device,
        use_model_uncertainty=bool(args.model_uncertainty),
        process_noise=bool(args.process_noise),
    )
    metrics = scanner_metrics(assembly, result)
    trajectory_path = _write_json(
        output / "trajectory.json", _trajectory_payload(result, metrics)
    )
    scene = scanner_physical_scene(assembly, result)
    if scene.get("assembly_sha256") != assembly.assembly_sha256:
        raise RuntimeError("scanner scene is not bound to the resolved assembly hash")
    scene_path = _write_json(output / "physical-scene.json", scene)
    return result, mission, metrics, trajectory_path, scene_path, scene


def command_simulate(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assembly = _assembly_from_args(args)
    _result, _mission, metrics, trajectory_path, scene_path, _scene = _simulate_and_write(
        assembly, args, output
    )
    report = {
        "schema": "contraption.simulation-report/v2",
        "assembly_sha256": assembly.assembly_sha256,
        "pmdl_sha256": assembly.system.pmdl_sha256,
        "controller": _controller_identity(assembly),
        "accepted": bool(metrics.get("accepted", False)),
        "acceptance_scope": "in_silico_component_assembly_only",
        "dynamics_completeness": metrics.get("dynamics_completeness"),
        "metrics": metrics,
        "artifacts": {
            "trajectory": str(trajectory_path),
            "physical_scene": str(scene_path),
        },
    }
    report_path = _write_json(output / "report.json", report)
    print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
    return 0


def command_view(args: argparse.Namespace) -> int:
    assembly = _assembly_from_args(args)
    result = (
        None
        if args.trajectory is None
        else _trajectory_result(assembly, args.trajectory)
    )
    sample_index = (
        0
        if result is None
        else result.metadata.get("pose_frame_sample_index", 0)
    )
    output = Path(args.output).resolve()
    artifact = generate_viewer(
        assembly,
        result,
        sample_index=sample_index,
        title=args.title,
    )
    paths = artifact.write(output)
    print(
        json.dumps(
            {
                "assembly_sha256": assembly.assembly_sha256,
                "controller": _controller_identity(assembly),
                "viewer": str(paths["index.html"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_compile(args: argparse.Namespace) -> int:
    assembly = _assembly_from_args(args)
    output = Path(args.output).resolve()
    artifact = compile_resolved_assembly(
        assembly,
        model_name=args.model_name,
        nominal_dt=float(args.nominal_dt),
        expected_assembly_sha256=assembly.assembly_sha256,
        expected_pmdl_sha256=assembly.system.pmdl_sha256,
        check_syntax=not args.skip_syntax_check,
        compiler=args.compiler,
    )
    paths = artifact.write(output)
    check = (
        syntax_check(artifact, args.compiler)
        if not args.skip_syntax_check
        else None
    )
    print(
        json.dumps(
            {
                "assembly_sha256": assembly.assembly_sha256,
                "pmdl_sha256": assembly.system.pmdl_sha256,
                "controller": _controller_identity(assembly),
                "dynamics_completeness": artifact.manifest[
                    "dynamics_completeness"
                ],
                "files": {name: str(path) for name, path in paths.items()},
                "syntax": {
                    "checked": check is not None and check.compiler is not None,
                    "ok": None if check is None else check.ok,
                    "compiler": None if check is None else check.compiler,
                    "stderr": "" if check is None else check.stderr,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_build(args: argparse.Namespace) -> int:
    assembly = _assembly_from_args(args)
    output = Path(args.output).resolve()
    plan = generate_build_instructions(assembly, output)
    print(
        json.dumps(
            {
                "assembly_sha256": plan.assembly_sha256,
                "pmdl_sha256": plan.pmdl_sha256,
                "controller": None if plan.controller is None else dict(plan.controller),
                "build_ready": plan.build_ready,
                "unresolved_count": len(plan.unresolved),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_serve(args: argparse.Namespace) -> int:
    assembly = _assembly_from_args(args)
    application = LiveScannerApplication(
        assembly,
        duration=args.duration,
        dt=float(args.dt),
        backend=args.backend,
        device=args.device,
        seed=int(args.seed),
    )
    address = f"http://{args.host}:{args.port}"
    print(
        json.dumps(
            {
                "assembly_sha256": assembly.assembly_sha256,
                "controller": _controller_identity(assembly),
                "viewer": address,
                "simulation_endpoint": address + "/api/simulate",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    serve_live_scanner(application, host=args.host, port=args.port)
    return 0


def command_demo(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assembly = _assembly_from_args(args)
    result, _mission, metrics, trajectory_path, scene_path, _scene = _simulate_and_write(
        assembly, args, output
    )
    generate_viewer(
        assembly,
        result,
        sample_index=result.metadata.get("pose_frame_sample_index", 0),
        output=output / "viewer",
        title="Apartment scanner robot — component-assembly simulation",
    )
    build_plan = generate_build_instructions(assembly, output / "build")
    compiled = compile_resolved_assembly(
        assembly,
        output / "online",
        model_name="scanner_online",
        nominal_dt=float(args.dt),
        expected_assembly_sha256=assembly.assembly_sha256,
        expected_pmdl_sha256=assembly.system.pmdl_sha256,
        check_syntax=not args.skip_syntax_check,
        compiler=args.compiler,
    )
    syntax = (
        syntax_check(compiled, args.compiler)
        if not args.skip_syntax_check
        else None
    )
    report = {
        "schema": "contraption.demo-report/v2",
        "assembly_sha256": assembly.assembly_sha256,
        "pmdl_sha256": assembly.system.pmdl_sha256,
        "controller": _controller_identity(assembly),
        "accepted": bool(metrics.get("accepted", False)),
        "physical_build_ready": build_plan.build_ready,
        "deployment_ready": False,
        "metrics": metrics,
        "online_compiler": {
            "model": compiled.model_name,
            "c99_syntax_checked": syntax is not None and syntax.compiler is not None,
            "c99_syntax_ok": None if syntax is None else syntax.ok,
            "compiler": None if syntax is None else syntax.compiler,
        },
        "artifacts": {
            "trajectory": str(trajectory_path),
            "physical_scene": str(scene_path),
            "viewer": str(output / "viewer" / "index.html"),
            "build_plan": str(output / "build" / "BUILD_INSTRUCTIONS.md"),
            "online_manifest": str(
                output / "online" / f"{compiled.model_name}.manifest.json"
            ),
        },
    }
    report_path = _write_json(output / "report.json", report)
    print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
    return 0


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
                component = _load_json(PART_IMPORT_CANARY_ROOT / "fixed_resistor.json")
                interface_data = load_interface_catalog(DEFAULT_CATALOG).to_dict()
                value, usage, charged = ClassificationAgent(ledger, api_key=key).classify(
                    component, interface_data, canary=True
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
                gold_templates=(
                    DEFAULT_CATALOG / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl",
                    DEFAULT_CATALOG / "electrical" / "resistors" / "fixed_resistors" / "instantiations" / "generic-100ohm-resistor" / "static.part",
                    DEFAULT_CATALOG / "electrical" / "resistors" / "fixed_resistors" / "instantiations" / "generic-100ohm-resistor" / "v1.model",
                ),
                interfaces=(
                    DEFAULT_CATALOG / "electrical" / "interface.pmdl",
                    DEFAULT_CATALOG / "electrical" / "resistors" / "interface.pmdl",
                    DEFAULT_CATALOG / "electrical" / "resistors" / "fixed_resistors" / "interface.pmdl",
                ),
                direct_hierarchy=(),
                component_information=PART_IMPORT_CANARY_ROOT / "fixed_resistor.json",
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
    model_root = DEFAULT_CATALOG
    return ModelingInputs(
        constraints=PROJECT_ROOT / "prompts" / "model_constraints.md",
        gold_templates=(
            model_root / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl",
            model_root / "electrical" / "resistors" / "fixed_resistors" / "instantiations" / "generic-100ohm-resistor" / "static.part",
            model_root / "electrical" / "resistors" / "fixed_resistors" / "instantiations" / "generic-100ohm-resistor" / "v1.model",
            model_root / "mechanical" / "inert_objects" / "planar_rigid_bodies" / "rigid_body_planar.pmdl",
        ),
        interfaces=interface_paths(DEFAULT_CATALOG),
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
                DEFAULT_CATALOG,
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

    def assembly_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--spec", default=str(DEFAULT_SPEC))
        command.add_argument(
            "--catalog",
            default=str(DEFAULT_CATALOG),
            help="model_catalog root containing interfaces, PMDLs, and instantiations",
        )
        command.add_argument(
            "--controller-root",
            action="append",
            help="controller JSON/directory (repeatable; defaults to scanner controls)",
        )

    def simulation_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--duration", type=float)
        command.add_argument("--samples", type=int, default=1)
        command.add_argument("--dt", type=float, default=0.05)
        command.add_argument("--seed", type=int, default=20260806)
        command.add_argument(
            "--backend", choices=("numpy", "torch", "auto"), default="numpy"
        )
        command.add_argument("--device")
        command.add_argument("--model-uncertainty", action="store_true")
        command.add_argument("--process-noise", action="store_true")

    def compiler_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--compiler")
        command.add_argument(
            "--skip-syntax-check",
            action="store_true",
            help="emit C99 without requiring a local host compiler check",
        )

    commands.add_parser("doctor", help="report optional runtime/tool availability").set_defaults(
        handler=command_doctor
    )
    validate = commands.add_parser(
        "validate",
        help="resolve and validate one canonical catalog/physical/PMDL closure",
    )
    assembly_inputs(validate)
    validate.set_defaults(handler=command_validate)

    simulate_command = commands.add_parser(
        "simulate", help="simulate the canonical scanner component assembly"
    )
    assembly_inputs(simulate_command)
    simulation_options(simulate_command)
    simulate_command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    simulate_command.set_defaults(handler=command_simulate)

    view = commands.add_parser(
        "view", help="generate a display-only viewer from a canonical resolved assembly"
    )
    assembly_inputs(view)
    view.add_argument(
        "--trajectory",
        help="v2 trajectory containing exact hash-bound simulation samples",
    )
    view.add_argument("--output", default=str(DEFAULT_OUTPUT / "viewer"))
    view.add_argument("--title")
    view.set_defaults(handler=command_view)

    compile_command = commands.add_parser(
        "compile",
        help="derive and syntax-check dynamics/estimator C99 from the resolved assembly",
    )
    assembly_inputs(compile_command)
    compiler_options(compile_command)
    compile_command.add_argument("--output", default=str(DEFAULT_OUTPUT / "online"))
    compile_command.add_argument("--model-name", default="scanner_online")
    compile_command.add_argument("--nominal-dt", type=float, default=0.05)
    compile_command.set_defaults(handler=command_compile)

    build_command = commands.add_parser(
        "build", help="derive a hash-bound build plan from the resolved assembly"
    )
    assembly_inputs(build_command)
    build_command.add_argument("--output", default=str(DEFAULT_OUTPUT / "build"))
    build_command.set_defaults(handler=command_build)

    serve = commands.add_parser(
        "serve",
        help="serve live controls that rerun the canonical Python simulation",
    )
    assembly_inputs(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--duration", type=float, default=2.0)
    serve.add_argument("--dt", type=float, default=0.05)
    serve.add_argument("--seed", type=int, default=20260806)
    serve.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    serve.add_argument("--device")
    serve.set_defaults(handler=command_serve)

    demo = commands.add_parser(
        "demo", help="simulate, view, compile, and plan the canonical scanner assembly"
    )
    assembly_inputs(demo)
    simulation_options(demo)
    compiler_options(demo)
    demo.add_argument("--output", default=str(DEFAULT_OUTPUT))
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
