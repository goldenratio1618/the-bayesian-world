"""Command-line entry points for validation, demos, compilation, and agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping
import uuid

import numpy as np

from .catalog.interfaces import interface_paths, load_interface_catalog
from .control import compile_resolved_controller, control_digest
from .live import LiveApplication, scene_from_result
from .loading import load_contraption
from .manufacturing.build import generate_build_instructions
from .part_import.agents import (
    ClassificationAgent,
    DirectResponsesModelingAgent,
    ModelingAgent,
    ModelingInputs,
    load_dotenv_key,
    run_classification_batch,
    run_modeling_proposal,
    write_json_atomic,
)
from .part_import.ingestion import (
    build_ingestion_report,
    combine_ingestion_metrics,
    combine_ingestion_metrics_with_carryovers,
    complete_batch_targets,
    failed_run_carryover,
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
from .part_import.part_markdown import write_part_markdown
from .part_import.budget import BudgetLedger
from .paths import asset_root, source_root
from .physics.backend import infer_backend
from .physics.resolved import ResolvedAssembly
from .physics.simulator import SimulationResult, controller_time_step, simulate
from .physics.uq import summarize_samples
from .visualization.server import serve_live
from .visualization.viewer import generate_viewer


PROJECT_ROOT = asset_root()
SOURCE_ROOT = source_root()
WORK_ROOT = SOURCE_ROOT if PROJECT_ROOT == SOURCE_ROOT else Path.cwd().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("CONTRAPTION_OUTPUT_ROOT", str(WORK_ROOT / "outputs"))
).expanduser().resolve()
PART_IMPORT_CANARY_ROOT = OUTPUT_ROOT / "part-import-canary"
PRIOR_FAILED_RUNS_BINDING = Path("outputs/part-ingestion-prior-failed-runs.json")
INGESTION_LEDGER_BINDING_SCHEMA = "contraption.part-ingestion-ledger-binding/v1"
DEFAULT_CATALOG = PROJECT_ROOT / "model_catalog"
DEFAULT_OUTPUT = OUTPUT_ROOT / "contraption_run"


def _controller_identities(assembly: ResolvedAssembly) -> list[dict[str, str]]:
    return [
        {
            "id": controller.id,
            "program_id": controller.spec.id,
            "version": controller.spec.version,
            "sha256": control_digest(controller.spec),
        }
        for _name, controller in sorted(assembly.controllers.items())
    ]


def _compilation_report(bundle: Any) -> dict[str, Any]:
    target_suffixes = {"c99": {".c", ".h"}, "verilog": {".v"}}
    target_digests: dict[str, dict[str, str]] = {}
    for target in bundle.targets:
        suffixes = target_suffixes[target]
        artifacts = {
            artifact.path: artifact.sha256
            for artifact in bundle.artifacts
            if Path(artifact.path).suffix in suffixes
        }
        if not artifacts:
            raise RuntimeError(
                f"controller compiler produced no {target!r} artifacts"
            )
        target_digests[target] = artifacts
    return {
        "source_digest": bundle.source_digest,
        "targets": list(bundle.targets),
        "target_digests": target_digests,
        # CompilationBundle.manifest is the canonical recursively-plain
        # projection of its deeply immutable closure.  A shallow dict() here
        # leaves nested FrozenDict values that json.dumps cannot serialize.
        "closure": bundle.manifest["closure"],
    }


def _validate_controller_compilation(
    assembly: ResolvedAssembly,
) -> dict[str, dict[str, Any]]:
    """Lower every resolved controller in memory as an admission requirement."""

    return {
        controller_id: _compilation_report(
            compile_resolved_controller(
                assembly,
                controller_id,
                identifier=controller_id,
                targets=("c99", "verilog"),
            )
        )
        for controller_id in sorted(assembly.controllers)
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


def _controller_input_scalar(value: Any, context: str) -> bool | float:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if np.isfinite(result):
            return result
    raise ValueError(f"{context} must be a finite JSON number or boolean")


def _controller_inputs_from_args(args: argparse.Namespace) -> dict[str, bool | float]:
    """Decode unambiguous external controller-pin values from CLI arguments."""

    result: dict[str, bool | float] = {}
    input_file = args.controller_input_file
    if input_file is not None:
        for name, value in _load_json(Path(input_file).expanduser().resolve()).items():
            result[name] = _controller_input_scalar(
                value, f"controller input file value {name!r}"
            )
    for assignment in args.controller_input:
        if "=" not in assignment:
            raise ValueError(
                f"controller input {assignment!r} must use NAME=JSON syntax"
            )
        name, encoded = assignment.split("=", 1)
        if not name or name.strip() != name:
            raise ValueError(
                f"controller input name {name!r} must be non-empty and unpadded"
            )
        if name in result:
            raise ValueError(f"controller input {name!r} was supplied more than once")
        try:
            value = json.loads(encoded, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"controller input {name!r} must contain one valid JSON scalar: {exc}"
            ) from exc
        result[name] = _controller_input_scalar(
            value, f"controller input {name!r}"
        )
    return result


def _write_json(path: str | Path, value: Any) -> Path:
    return write_json_atomic(path, value)


def _load_resolved_assembly(
    specification_path: str | Path,
) -> ResolvedAssembly:
    """Load exactly one canonical ``contraption-4`` filesystem closure."""

    return load_contraption(Path(specification_path).expanduser().resolve())


def _assembly_from_args(args: argparse.Namespace) -> ResolvedAssembly:
    return _load_resolved_assembly(args.spec)


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
        "iverilog": shutil.which("iverilog"),
        "verilator": shutil.which("verilator"),
        "yosys": shutil.which("yosys"),
        "budget_limit_usd": 100.0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    assembly = _assembly_from_args(args)
    controller_compilation = _validate_controller_compilation(assembly)
    dynamics_completeness = assembly.dynamics_completeness
    report = {
        "valid": True,
        "scope": "contraption-4_filesystem_closure",
        "assembly": assembly.diagnostics(),
        "controllers": _controller_identities(assembly),
        "controller_compilation": controller_compilation,
        "dynamics_completeness": (
            None
            if dynamics_completeness is None
            else dynamics_completeness.to_dict()
        ),
        "parts": {
            "registered": len(assembly.parts),
            "used": len(
                {item.part for item in assembly.specification.components}
                | {item.part for item in assembly.physical.world_objects}
            ),
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
) -> tuple[SimulationResult, Mapping[str, Any], Path, Path, Mapping[str, Any]]:
    result = simulate(
        assembly,
        duration=_simulation_duration(assembly, args.duration),
        dt=_simulation_dt(assembly, args.dt),
        num_samples=int(args.samples),
        seed=int(args.seed),
        backend=args.backend,
        device=args.device,
        use_model_uncertainty=bool(args.model_uncertainty),
        process_noise=bool(args.process_noise),
        controller_inputs=_controller_inputs_from_args(args),
    )
    metrics = {
        "duration_s": float(result.time[-1] - result.time[0]),
        "sample_count": int(result.samples.shape[0]),
        "state_count": len(result.state_names),
        "output_count": len(result.output_names),
        "controllers": list(result.controller_traces),
        "verifications": {
            name: report.to_dict()
            for name, report in result.verification_reports.items()
        },
        "dynamics_completeness": (
            None
            if assembly.dynamics_completeness is None
            else assembly.dynamics_completeness.to_dict()
        ),
    }
    trajectory_path = _write_json(
        output / "trajectory.json", _trajectory_payload(result, metrics)
    )
    scene = scene_from_result(assembly, result)
    scene_path = _write_json(output / "physical-scene.json", scene)
    return result, metrics, trajectory_path, scene_path, scene


def _simulation_dt(
    assembly: ResolvedAssembly, requested: float | None
) -> float:
    return controller_time_step(
        (item.spec.period_s for item in assembly.controllers.values()),
        requested,
    )


def _mission_setting(
    assembly: ResolvedAssembly,
    name: str,
    default: float | int,
) -> float | int:
    environment = assembly.specification.environment
    mission = environment.get("mission", {})
    if not isinstance(mission, Mapping):
        raise ValueError("contraption.environment.mission must be an object")
    return mission.get(name, default)


def _simulation_duration(
    assembly: ResolvedAssembly,
    requested: float | None,
) -> float:
    value = (
        _mission_setting(assembly, "duration_s", 1.0)
        if requested is None
        else requested
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("simulation duration must be a finite positive number")
    return float(value)


def _optical_frame_count(
    assembly: ResolvedAssembly,
    requested: int | None,
) -> int:
    value = (
        _mission_setting(assembly, "camera_frame_count", 1)
        if requested is None
        else requested
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("optical frame count must be a positive integer")
    return value


def command_simulate(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assembly = _assembly_from_args(args)
    result, metrics, trajectory_path, scene_path, _scene = _simulate_and_write(
        assembly, args, output
    )
    optical_captures = []
    optical_observations = []
    if args.optical_capture:
        from .optics import capture_result

        frame_count = _optical_frame_count(assembly, args.optical_frame_count)
        if args.optical_frame_count is None and args.duration is not None:
            # An explicit short simulation is an ad-hoc capture rather than the
            # authored mission.  Preserve the historical single final-frame
            # behavior unless the caller also explicitly requests a sequence.
            frame_count = 1
        if args.optical_time_index is not None and frame_count != 1:
            raise ValueError(
                "--optical-time-index requires --optical-frame-count 1"
            )
        if frame_count > len(result.time):
            raise ValueError(
                "optical frame count exceeds the number of simulation time points"
            )
        time_indices = (
            [args.optical_time_index if args.optical_time_index is not None else -1]
            if frame_count == 1
            else np.rint(
                np.linspace(0, len(result.time) - 1, frame_count)
            ).astype(int).tolist()
        )
        base_seed = args.seed if args.optical_seed is None else args.optical_seed
        for capture_index, time_index in enumerate(time_indices):
            resolved_index = time_index if time_index >= 0 else len(result.time) + time_index
            optical_capture = capture_result(
                assembly,
                result,
                (
                    output / "optical"
                    if frame_count == 1
                    else output / "optical" / f"frame-{resolved_index:08d}"
                ),
                sample_index=args.optical_sample_index,
                time_index=time_index,
                sensor_ids=tuple(args.optical_sensor),
                sensor_resolution_px=(args.optical_width, args.optical_height),
                backend=args.optical_backend,
                device=args.optical_device,
                seed=base_seed + capture_index * 1000,
                external_scene=args.optical_scene,
            )
            optical_captures.append(optical_capture)
            optical_observations.extend(optical_capture.observations)
        metrics["optical_capture"] = {
            "backend": optical_captures[0].backend,
            "device": optical_captures[0].device,
            "frame_count": len(optical_captures),
            "sensor_count": len(optical_captures[0].frame.sensors),
            "observation_count": len(optical_observations),
            "frames": [
                {
                    "time_index": item.frame.time_index,
                    "time_s": item.frame.time_s,
                    "runtime_scene_sha256": item.frame.scene.artifact_sha256,
                    "external_scene_sha256": item.frame.external_scene_sha256,
                }
                for item in optical_captures
            ],
        }
    artifacts = {
        "trajectory": str(trajectory_path),
        "physical_scene": str(scene_path),
    }
    if optical_captures:
        if len(optical_captures) == 1:
            artifacts["optical_capture"] = str(optical_captures[0].report_path)
        else:
            artifacts["optical_captures"] = [
                str(item.report_path) for item in optical_captures
            ]
        viewer = generate_viewer(
            assembly,
            result,
            sample_index=args.optical_sample_index,
            title="Full scanner orbit and camera views",
            optical_sensors=tuple(optical_captures[0].frame.sensors),
            optical_observations=tuple(optical_observations),
        )
        viewer_paths = viewer.write(output / "viewer")
        artifacts["viewer"] = str(viewer_paths["index.html"])
    report = {
        "schema": "contraption.simulation-report/v2",
        "assembly_sha256": assembly.assembly_sha256,
        "pmdl_sha256": assembly.system.pmdl_sha256,
        "controllers": _controller_identities(assembly),
        "dynamics_completeness": metrics.get("dynamics_completeness"),
        "metrics": metrics,
        "artifacts": artifacts,
    }
    report_path = _write_json(output / "report.json", report)
    print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
    return 0


def command_optical_reconstruct(args: argparse.Namespace) -> int:
    from .optics import reconstruct_observations

    result = reconstruct_observations(
        args.sensor,
        args.observation,
        args.output,
        id=args.id,
        voxel_size_m=args.voxel_size,
        block_size=args.block_size,
        origin_world_m=tuple(args.origin),
        truncation_distance_m=args.truncation_distance,
        pixel_stride=args.pixel_stride,
        surface_occupancy_threshold=args.surface_occupancy_threshold,
        surface_maximum_abs_tsdf=args.surface_maximum_abs_tsdf,
        surface_maximum_occupied_voxels=args.surface_maximum_occupied_voxels,
        surface_maximum_triangles=args.surface_maximum_triangles,
    )
    report_path = Path(args.output).resolve() / "report.json"
    print(
        json.dumps(
            {"report": str(report_path), **result.to_dict()},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
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
                "controllers": _controller_identities(assembly),
                "viewer": str(paths["index.html"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _compile_controllers(
    assembly: ResolvedAssembly, output: Path
) -> dict[str, Any]:
    """Compile every resolved controller into its own deterministic directory."""

    output.mkdir(parents=True, exist_ok=True)
    compiled: dict[str, Any] = {}
    for controller_id in sorted(assembly.controllers):
        bundle = compile_resolved_controller(
            assembly,
            controller_id,
            identifier=controller_id,
            targets=("c99", "verilog"),
        )
        paths = bundle.write(output / controller_id)
        compiled[controller_id] = {
            **_compilation_report(bundle),
            "files": [str(path) for path in paths],
        }
    return compiled


def command_compile(args: argparse.Namespace) -> int:
    assembly = _assembly_from_args(args)
    output = Path(args.output).resolve()
    compiled = _compile_controllers(assembly, output)
    print(
        json.dumps(
            {
                "assembly_sha256": assembly.assembly_sha256,
                "pmdl_sha256": assembly.system.pmdl_sha256,
                "controllers": _controller_identities(assembly),
                "compiled": compiled,
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
                "controllers": [dict(item) for item in plan.controllers],
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
    application = LiveApplication(
        assembly,
        duration=args.duration,
        dt=args.dt,
        backend=args.backend,
        device=args.device,
        seed=int(args.seed),
        initial_inputs=_controller_inputs_from_args(args),
    )
    address = f"http://{args.host}:{args.port}"
    print(
        json.dumps(
            {
                "assembly_sha256": assembly.assembly_sha256,
                "controllers": _controller_identities(assembly),
                "viewer": address,
                "simulation_endpoint": address + "/api/simulate",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    serve_live(application, host=args.host, port=args.port)
    return 0


def command_demo(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assembly = _assembly_from_args(args)
    result, metrics, trajectory_path, scene_path, _scene = _simulate_and_write(
        assembly, args, output
    )
    generate_viewer(
        assembly,
        result,
        sample_index=result.metadata.get("pose_frame_sample_index", 0),
        output=output / "viewer",
        title=assembly.specification.name,
    )
    build_plan = generate_build_instructions(assembly, output / "build")
    compiled = _compile_controllers(assembly, output / "controllers")
    report = {
        "schema": "contraption.demo-report/v2",
        "assembly_sha256": assembly.assembly_sha256,
        "pmdl_sha256": assembly.system.pmdl_sha256,
        "controllers": _controller_identities(assembly),
        "physical_build_ready": build_plan.build_ready,
        "deployment_ready": False,
        "metrics": metrics,
        "compiled_controllers": compiled,
        "artifacts": {
            "trajectory": str(trajectory_path),
            "physical_scene": str(scene_path),
            "viewer": str(output / "viewer" / "index.html"),
            "build_plan": str(output / "build" / "BUILD_INSTRUCTIONS.md"),
            "compiled_controllers": str(output / "controllers"),
        },
    }
    report_path = _write_json(output / "report.json", report)
    print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
    return 0


def _agent_ledger(path: str | None, *, limit_usd: float = 100.0) -> BudgetLedger:
    return BudgetLedger(
        Path(path).resolve() if path else OUTPUT_ROOT / "agent-budget.json",
        limit_usd=limit_usd,
    )


def command_budget(args: argparse.Namespace) -> int:
    print(json.dumps(_agent_ledger(args.ledger).snapshot(), indent=2, sort_keys=True))
    return 0


def command_part_markdown(args: argparse.Namespace) -> int:
    part_directory = Path(args.part_directory).expanduser().resolve()
    catalog_root = (
        Path(args.catalog).expanduser().resolve() if args.catalog else None
    )
    output = Path(args.output).expanduser().resolve() if args.output else None
    written = write_part_markdown(
        part_directory,
        catalog_root=catalog_root,
        output=output,
    )
    print(json.dumps({"part_directory": str(part_directory), "markdown": str(written)}, indent=2, sort_keys=True))
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
                    DEFAULT_CATALOG / "electrical" / "resistors" / "resistor.pmdl",
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
                PART_IMPORT_CANARY_ROOT / "agent-staging",
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
    destination = PART_IMPORT_CANARY_ROOT / "agent-canary-report.json"
    _write_json(destination, results)
    print(json.dumps({"report": str(destination), **results}, indent=2, sort_keys=True))
    return 0 if successes else 2


def _agent_key(env_file: str | None) -> tuple[str | None, Path]:
    dotenv = Path(env_file).expanduser().resolve() if env_file else _default_dotenv_path()
    return os.environ.get("OPENAI_API_KEY") or load_dotenv_key(dotenv), dotenv


def _agent_failure_reason(exc: Exception, key: str | None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message.replace(key, "[REDACTED_OPENAI_API_KEY]") if key else message


@dataclass(frozen=True, slots=True)
class _AgentJob:
    id: str
    component_information: Path
    direct_hierarchy: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _AgentJobBundle:
    catalog: Path
    constraints: Path
    gold_templates: tuple[Path, ...]
    jobs: tuple[_AgentJob, ...]

    def job(self, identifier: str) -> _AgentJob:
        for job in self.jobs:
            if job.id == identifier:
                return job
        raise ValueError(
            f"unknown agent target {identifier!r}; declared targets are "
            f"{[job.id for job in self.jobs]}"
        )


def _job_path(root: Path, value: Any, context: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{context} must be a non-empty relative path")
    result = (root / value).resolve()
    exists = result.is_dir() if directory else result.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{context} does not resolve to an existing {kind}: {result}")
    return result


def _load_agent_job_bundle(path: str | Path) -> _AgentJobBundle:
    source = Path(path).expanduser().resolve()
    data = _load_json(source)
    expected = {"format", "catalog", "constraints", "gold_templates", "jobs"}
    if set(data) != expected or data.get("format") != "agent-jobs-1":
        raise ValueError(
            "agent job file must be an exact agent-jobs-1 object with catalog, "
            "constraints, gold_templates, and jobs"
        )
    root = source.parent
    catalog = _job_path(root, data["catalog"], "agent jobs catalog", directory=True)
    constraints = _job_path(root, data["constraints"], "agent jobs constraints")
    raw_templates = data["gold_templates"]
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("agent jobs gold_templates must be a non-empty list")
    templates = tuple(
        _job_path(root, value, f"agent jobs gold_templates[{index}]")
        for index, value in enumerate(raw_templates)
    )
    raw_jobs = data["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("agent jobs jobs must be a non-empty list")
    jobs: list[_AgentJob] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "component_information",
            "direct_hierarchy",
        }:
            raise ValueError(
                f"agent jobs jobs[{index}] must contain exactly id, "
                "component_information, and direct_hierarchy"
            )
        identifier = raw["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"agent jobs jobs[{index}].id must be a non-empty string")
        if identifier in identifiers:
            raise ValueError(f"duplicate agent job id {identifier!r}")
        identifiers.add(identifier)
        hierarchy = raw["direct_hierarchy"]
        if not isinstance(hierarchy, list):
            raise ValueError(
                f"agent jobs jobs[{index}].direct_hierarchy must be a list"
            )
        jobs.append(
            _AgentJob(
                identifier,
                _job_path(
                    root,
                    raw["component_information"],
                    f"agent jobs jobs[{index}].component_information",
                ),
                tuple(
                    _job_path(
                        root,
                        value,
                        f"agent jobs jobs[{index}].direct_hierarchy[{item_index}]",
                    )
                    for item_index, value in enumerate(hierarchy)
                ),
            )
        )
    return _AgentJobBundle(catalog, constraints, templates, tuple(jobs))


def _full_modeling_inputs(bundle: _AgentJobBundle, target: str) -> ModelingInputs:
    job = bundle.job(target)
    return ModelingInputs(
        constraints=bundle.constraints,
        gold_templates=bundle.gold_templates,
        interfaces=interface_paths(bundle.catalog),
        direct_hierarchy=job.direct_hierarchy,
        component_information=job.component_information,
    )


def _replay_direct_child(root: Path, name: str) -> Path:
    child = root / name
    if child.is_symlink():
        raise ValueError(f"replay path cannot be a symlink: {child}")
    resolved = child.resolve()
    if resolved.parent != root:
        raise ValueError(f"replay path escaped its run root: {child}")
    return resolved


def _ingestion_ledger_binding(replay_root: str | Path) -> dict[str, Any]:
    """Return the exact durable replay ledger identity without creating it."""

    root = Path(replay_root).resolve()
    path = _replay_direct_child(root, "agent-budget.json")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"ingestion replay ledger is missing: {path}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"ingestion replay ledger must be a regular non-symlink file: {path}"
        )
    payload = path.read_bytes()
    return {
        "schema": INGESTION_LEDGER_BINDING_SCHEMA,
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def _validate_ingestion_ledger_binding(
    value: Mapping[str, Any], replay_root: str | Path
) -> dict[str, Any]:
    """Re-read and exactly compare the canary's run-local ledger binding."""

    if not isinstance(value, Mapping):
        raise ValueError("canary report has no run-ledger binding")
    actual = _ingestion_ledger_binding(replay_root)
    if dict(value) != actual:
        raise ValueError("ingestion replay ledger changed after the canary")
    return actual


def _prior_failed_run_bindings(
    raw_ledgers: list[str],
    *,
    target: str,
    component_sha256: str,
    current_ledger: Path,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for raw_prior in raw_ledgers:
        prior_path = Path(raw_prior).expanduser()
        if prior_path.is_symlink():
            raise ValueError("prior failed ledger cannot be a symlink")
        prior_path = prior_path.resolve()
        if prior_path == current_ledger:
            raise ValueError("prior failed ledger cannot be the new replay ledger")
        if prior_path == OUTPUT_ROOT or OUTPUT_ROOT not in prior_path.parents:
            raise ValueError(
                "prior failed ledger must be a strict descendant of outputs/"
            )
        bindings.append(
            failed_run_carryover(
                prior_path,
                expected_target=target,
                expected_component_sha256=component_sha256,
            )
        )
    return validate_failed_run_carryovers(bindings)


def _prior_failed_binding_path(data_root: str | Path) -> Path:
    root = Path(data_root).resolve()
    directory = root / PRIOR_FAILED_RUNS_BINDING.parent
    path = root / PRIOR_FAILED_RUNS_BINDING
    if directory.is_symlink() or path.is_symlink():
        raise ValueError("prior failed-run binding cannot be a symlink")
    if path.resolve().parents[1] != root:
        raise ValueError("prior failed-run binding escaped the isolated data root")
    return path


def command_agent_run(args: argparse.Namespace) -> int:
    """Run paid importer jobs with resumable receipts and guarded promotion."""

    key, dotenv = _agent_key(args.env_file)
    job_file = Path(args.job_file).expanduser().resolve()
    if OUTPUT_ROOT == job_file.parent or OUTPUT_ROOT in job_file.parent.parents:
        run_root = job_file.parent
    else:
        run_root = OUTPUT_ROOT / "part-import" / job_file.stem
    ingestion_job = args.agent_job in {"ingestion-canary", "ingestion-batch"}
    raw_proposal_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else run_root / "agent-proposals"
    )
    if ingestion_job and raw_proposal_root.is_symlink():
        raise ValueError("ingestion --output-root cannot be a symlink")
    proposal_root = raw_proposal_root.resolve()
    ingestion_staging_root: Path | None = None
    if ingestion_job:
        if proposal_root == OUTPUT_ROOT or OUTPUT_ROOT not in proposal_root.parents:
            raise ValueError(
                "ingestion --output-root must be a strict descendant of outputs/"
            )
        if (
            proposal_root == job_file.parent
            or job_file.parent in proposal_root.parents
            or proposal_root in job_file.parent.parents
        ):
            raise ValueError(
                "ingestion --output-root must be a fresh replay run, not the source job run"
            )
        if args.agent_job == "ingestion-canary" and proposal_root.exists():
            if not proposal_root.is_dir() or any(proposal_root.iterdir()):
                raise ValueError(
                    "ingestion canary --output-root must be absent or an empty directory"
                )
        ingestion_staging_root = _replay_direct_child(
            proposal_root, "agent-staging"
        )
        if ingestion_staging_root.exists() and not ingestion_staging_root.is_dir():
            raise ValueError("ingestion staging path must be a directory")
        raw_supplied_staging = (
            Path(args.staging_root).expanduser()
            if args.staging_root
            else ingestion_staging_root
        )
        if raw_supplied_staging.is_symlink():
            raise ValueError("ingestion staging cannot be a symlink")
        supplied_staging = raw_supplied_staging.resolve()
        if supplied_staging != ingestion_staging_root:
            raise ValueError(
                "ingestion staging must be <output-root>/agent-staging"
            )
    ledger_path: str | Path | None = args.ledger
    preloaded_canary_report_path: Path | None = None
    preloaded_canary_report: dict[str, Any] | None = None
    if ingestion_job:
        expected_ledger = _replay_direct_child(proposal_root, "agent-budget.json")
        raw_supplied_ledger = (
            Path(args.ledger).expanduser() if args.ledger else expected_ledger
        )
        if raw_supplied_ledger.is_symlink():
            raise ValueError("ingestion ledger cannot be a symlink")
        supplied_ledger = raw_supplied_ledger.resolve()
        if supplied_ledger != expected_ledger:
            raise ValueError(
                "ingestion ledger must be <output-root>/agent-budget.json"
            )
        ledger_path = supplied_ledger
        if args.agent_job == "ingestion-batch":
            raw_canary_report = Path(args.canary_report).expanduser()
            if raw_canary_report.is_symlink():
                raise ValueError("canary report cannot be a symlink")
            preloaded_canary_report_path = raw_canary_report.resolve()
            preloaded_canary_report = _load_json(preloaded_canary_report_path)
            raw_replay_root = preloaded_canary_report.get("replay_run_root")
            if not isinstance(raw_replay_root, str) or not raw_replay_root:
                raise ValueError("canary report has no replay run root")
            recorded_replay_root = Path(raw_replay_root).expanduser()
            if recorded_replay_root.is_symlink():
                raise ValueError("recorded replay run root cannot be a symlink")
            if recorded_replay_root.resolve() != proposal_root:
                raise ValueError(
                    "batch --output-root differs from the canary replay run root"
                )
            expected_canary_report = _replay_direct_child(
                proposal_root, "ingestion-canary-report.json"
            )
            if preloaded_canary_report_path != expected_canary_report:
                raise ValueError("canary report is outside its replay run root")
            _validate_ingestion_ledger_binding(
                preloaded_canary_report.get("run_ledger"), proposal_root
            )
    ledger = _agent_ledger(
        str(ledger_path) if ledger_path is not None else None,
        limit_usd=float(getattr(args, "ledger_limit_usd", 100.0)),
    )
    if preloaded_canary_report is not None:
        # Construction validates the existing state but must not recreate or
        # change the exact canary ledger before the batch gate.
        _validate_ingestion_ledger_binding(
            preloaded_canary_report.get("run_ledger"), proposal_root
        )
    exit_code = 0
    try:
        if args.agent_job == "classification-all":
            jobs = _load_agent_job_bundle(job_file)
            if not key:
                raise RuntimeError(
                    f"OPENAI_API_KEY is absent and {dotenv} has no key; "
                    "classification-all was not dispatched"
                )
            component_paths = tuple(job.component_information for job in jobs.jobs)
            results = run_classification_batch(
                ClassificationAgent(ledger, api_key=key),
                component_paths,
                jobs.catalog,
                proposal_root / "classification",
                force=args.force,
            )
            payload = {
                "job": "classification-all",
                "results": results,
                "budget": ledger.snapshot(),
            }
        elif args.agent_job == "modeling-one":
            jobs = _load_agent_job_bundle(job_file)
            agent = ModelingAgent(
                ledger,
                (
                    Path(args.staging_root).expanduser().resolve()
                    if args.staging_root
                    else run_root / "agent-staging"
                ),
                codex_binary=os.environ.get("CODEX_BIN"),
                api_key=key,
            )
            result = run_modeling_proposal(
                agent,
                _full_modeling_inputs(jobs, args.target),
                args.target,
                proposal_root / "modeling",
                force=args.force,
            )
            payload = {
                "job": "modeling-one",
                "result": result,
                "budget": ledger.snapshot(),
            }
        elif args.agent_job in {"ingestion-canary", "ingestion-batch"}:
            if not key:
                raise RuntimeError(
                    f"OPENAI_API_KEY is absent and {dotenv} has no key; "
                    f"{args.agent_job} was not dispatched"
                )
            job_sha256 = "sha256:" + hashlib.sha256(job_file.read_bytes()).hexdigest()
            prior_failed_runs: list[dict[str, Any]] = []
            if args.agent_job == "ingestion-canary":
                source_inputs = _full_modeling_inputs(
                    _load_agent_job_bundle(job_file), args.target
                )
                component_sha256 = "sha256:" + hashlib.sha256(
                    Path(source_inputs.component_information).read_bytes()
                ).hexdigest()
                prior_failed_runs = _prior_failed_run_bindings(
                    args.prior_failed_ledger,
                    target=args.target,
                    component_sha256=component_sha256,
                    current_ledger=(proposal_root / "agent-budget.json").resolve(),
                )
                isolation = prepare_isolated_replay(
                    job_file, proposal_root / "isolated-data-root"
                )
                isolation_manifest = _replay_direct_child(
                    proposal_root, "isolation-manifest.json"
                )
                prior_failed_binding = _prior_failed_binding_path(
                    isolation["data_root"]
                )
                write_json_atomic(
                    prior_failed_binding,
                    {
                        "schema": "contraption.part-ingestion-prior-failed-runs/v1",
                        "runs": prior_failed_runs,
                    },
                )
                os.environ["CONTRAPTION_DATA_ROOT"] = isolation["data_root"]
                jobs = _load_agent_job_bundle(isolation["isolated_job_file"])
            else:
                if (
                    preloaded_canary_report_path is None
                    or preloaded_canary_report is None
                ):  # pragma: no cover - ingestion-batch invariant.
                    raise RuntimeError("batch canary report was not preloaded")
                canary_report_path = preloaded_canary_report_path
                canary_report = preloaded_canary_report
                raw_prior_failed_runs = canary_report.get("prior_failed_runs", [])
                if not isinstance(raw_prior_failed_runs, list):
                    raise ValueError("canary report has invalid prior failed runs")
                prior_failed_runs = validate_failed_run_carryovers(
                    raw_prior_failed_runs
                )
                replay_root = proposal_root
                expected_canary_report = _replay_direct_child(
                    replay_root, "ingestion-canary-report.json"
                )
                if canary_report_path != expected_canary_report:
                    raise ValueError("canary report is outside its replay run root")
                raw_manifest = canary_report.get("isolation_manifest")
                if not isinstance(raw_manifest, str):
                    raise ValueError("canary report has no isolation manifest")
                recorded_manifest = Path(raw_manifest).expanduser()
                if recorded_manifest.is_symlink():
                    raise ValueError("isolation manifest cannot be a symlink")
                isolation_manifest = recorded_manifest.resolve()
                expected_manifest = _replay_direct_child(
                    replay_root, "isolation-manifest.json"
                )
                if isolation_manifest != expected_manifest:
                    raise ValueError("canary isolation manifest is outside its replay root")
                expected_state = canary_report.get("replay_state")
                if not isinstance(expected_state, Mapping):
                    raise ValueError("canary report has no bound replay state")
                excluded = canary_report.get("canary_target")
                if not isinstance(excluded, str) or not excluded:
                    raise ValueError("canary report has no target binding")
                data_root = _replay_direct_child(replay_root, "isolated-data-root")
                if not data_root.is_dir():
                    raise FileNotFoundError("isolated replay data root is missing")
                state_job = expected_state.get("isolated_job")
                if not isinstance(state_job, Mapping) or not isinstance(
                    state_job.get("path"), str
                ):
                    raise ValueError("canary replay state has no isolated job binding")
                recorded_job = Path(state_job["path"]).expanduser()
                if recorded_job.is_symlink():
                    raise ValueError("isolated job inventory cannot be a symlink")
                isolated_job_file = recorded_job.resolve()
                if data_root not in isolated_job_file.parents:
                    raise ValueError("isolated job inventory escaped its data root")
                validate_replay_state(
                    expected_state,
                    data_root=data_root,
                    isolated_job_file=isolated_job_file,
                    isolation_manifest=isolation_manifest,
                    run_ledger=expected_ledger,
                    canary_target=excluded,
                )
                isolation = _load_json(isolation_manifest)
                if isolation.get("schema") != "contraption.part-import-replay-isolation/v2":
                    raise ValueError("unsupported replay isolation manifest")
                if isolation.get("source_job_sha256") != job_sha256:
                    raise ValueError("isolation manifest belongs to another job inventory")
                manifest_data_root = Path(str(isolation.get("data_root", ""))).expanduser()
                if manifest_data_root.is_symlink() or manifest_data_root != data_root:
                    raise ValueError("isolation manifest data root changed after canary")
                manifest_job = Path(
                    str(isolation.get("isolated_job_file", ""))
                ).expanduser()
                if manifest_job.is_symlink() or manifest_job != isolated_job_file:
                    raise ValueError("isolation manifest job path changed after canary")
                os.environ["CONTRAPTION_DATA_ROOT"] = str(data_root)
                jobs = _load_agent_job_bundle(isolated_job_file)
                prior_failed_binding = _prior_failed_binding_path(data_root)
                binding_value = _load_json(prior_failed_binding)
                if (
                    binding_value.get("schema")
                    != "contraption.part-ingestion-prior-failed-runs/v1"
                    or not isinstance(binding_value.get("runs"), list)
                ):
                    raise ValueError("isolated prior failed-run binding is malformed")
                bound_prior_failed_runs = validate_failed_run_carryovers(
                    binding_value["runs"]
                )
                batch_canary_inputs = _full_modeling_inputs(jobs, excluded)
                batch_component_sha256 = "sha256:" + hashlib.sha256(
                    Path(batch_canary_inputs.component_information).read_bytes()
                ).hexdigest()
                supplied_prior_failed_runs = _prior_failed_run_bindings(
                    args.prior_failed_ledger,
                    target=excluded,
                    component_sha256=batch_component_sha256,
                    current_ledger=(proposal_root / "agent-budget.json").resolve(),
                )
                prior_failed_runs = validate_matching_failed_run_carryovers(
                    prior_failed_runs,
                    bound_prior_failed_runs,
                    supplied_prior_failed_runs,
                )

            if ingestion_staging_root is None:  # pragma: no cover - branch invariant.
                raise RuntimeError("ingestion staging root was not initialized")
            staging_root = ingestion_staging_root
            classifier = ClassificationAgent(ledger, api_key=key)
            modeler = DirectResponsesModelingAgent(
                ledger, staging_root, api_key=key
            )
            interface_data = load_interface_catalog(jobs.catalog).to_dict()
            workflow_sha256 = workflow_fingerprint(
                classifier, modeler, interface_data
            )
            if args.agent_job == "ingestion-canary":
                ingestion_run_id = f"ingestion-canary-{uuid.uuid4()}"
                canary_failure: str | None = None
                prior_target_charged_usd = sum(
                    float(item["total_importer_charged_usd"])
                    for item in prior_failed_runs
                    if item.get("target") == args.target
                )
                try:
                    result = run_part_ingestion(
                        classifier,
                        modeler,
                        _full_modeling_inputs(jobs, args.target),
                        args.target,
                        jobs.catalog,
                        proposal_root / "proposals",
                        ingestion_run_id=ingestion_run_id,
                        canary=True,
                        force=args.force,
                        prior_target_charged_usd=prior_target_charged_usd,
                    )
                except Exception as exc:
                    canary_failure = _agent_failure_reason(exc, key)
                    result = {
                        "target": args.target,
                        "status": "failed",
                        "fully_ingested": False,
                        "reason": canary_failure,
                    }
                report = build_ingestion_report(
                    mode="canary",
                    results=(result,),
                    ledger=ledger,
                    ingestion_run_id=ingestion_run_id,
                    workflow_sha256=workflow_sha256,
                    job_file_sha256=job_sha256,
                    expected_target_count=1,
                )
                report.update(
                    {
                        "canary_target": args.target,
                        "inventory_targets": [job.id for job in jobs.jobs],
                        "replay_run_root": str(proposal_root),
                        "isolation_manifest": str(isolation_manifest),
                        "isolation_data_root": isolation["data_root"],
                        "prior_failed_runs": prior_failed_runs,
                        "prior_failed_runs_binding": str(prior_failed_binding),
                        "prior_target_charged_usd": prior_target_charged_usd,
                        "remaining_part_scope_limit_usd": (
                            0.05 - prior_target_charged_usd
                        ),
                        "run_ledger": _ingestion_ledger_binding(proposal_root),
                    }
                )
                try:
                    replay_state = replay_state_fingerprint(
                        isolation["data_root"],
                        isolation["isolated_job_file"],
                        isolation_manifest,
                        expected_ledger,
                        args.target,
                    )
                    expected_state_ledger = {
                        "path": report["run_ledger"]["path"],
                        "sha256": report["run_ledger"]["sha256"],
                    }
                    if replay_state.get("run_ledger") != expected_state_ledger:
                        raise ValueError(
                            "run ledger changed while binding canary replay state"
                        )
                    report["replay_state"] = replay_state
                except Exception as exc:
                    replay_state_error = _agent_failure_reason(exc, key)
                    report["replay_state_error"] = replay_state_error
                    result = {
                        **result,
                        "status": "failed",
                        "fully_ingested": False,
                        "reason": replay_state_error,
                    }
                    report["results"] = [result]
                    report["metrics"] = build_ingestion_report(
                        mode="canary",
                        results=(result,),
                        ledger=ledger,
                        ingestion_run_id=ingestion_run_id,
                        workflow_sha256=workflow_sha256,
                        job_file_sha256=job_sha256,
                        expected_target_count=1,
                    )["metrics"]
                    if canary_failure is None:
                        canary_failure = replay_state_error
                if canary_failure is not None:
                    report["failure_reason"] = canary_failure
                report_path = write_json_atomic(
                    proposal_root / "ingestion-canary-report.json", report
                )
                if canary_failure is not None:
                    exit_code = 2
            else:
                validate_canary_report_evidence(
                    canary_report,
                    ledger_snapshot=ledger.snapshot(),
                    catalog_root=jobs.catalog,
                    expected_target=excluded,
                )
                validate_canary_gate(
                    canary_report,
                    workflow_sha256=workflow_sha256,
                    job_file_sha256=job_sha256,
                )
                bound_inventory = canary_report.get("inventory_targets")
                actual_inventory = [job.id for job in jobs.jobs]
                if not isinstance(bound_inventory, list) or bound_inventory != actual_inventory:
                    raise ValueError(
                        "isolated job inventory differs from the passing canary"
                    )
                expected_targets = complete_batch_targets(
                    actual_inventory, excluded
                )
                targets = complete_batch_targets(
                    actual_inventory, excluded, args.target or ()
                )
                ingestion_run_id = f"ingestion-batch-{uuid.uuid4()}"
                results = []
                for target in targets:
                    try:
                        results.append(
                            run_part_ingestion(
                                classifier,
                                modeler,
                                _full_modeling_inputs(jobs, target),
                                target,
                                jobs.catalog,
                                proposal_root / "proposals",
                                ingestion_run_id=ingestion_run_id,
                                force=args.force,
                            )
                        )
                    except Exception as exc:
                        results.append(
                            {
                                "target": target,
                                "status": "failed",
                                "fully_ingested": False,
                                "reason": _agent_failure_reason(exc, key),
                            }
                        )
                        break
                report = build_ingestion_report(
                    mode="batch",
                    results=results,
                    ledger=ledger,
                    ingestion_run_id=ingestion_run_id,
                    workflow_sha256=workflow_sha256,
                    job_file_sha256=job_sha256,
                    expected_target_count=len(expected_targets),
                )
                report.update(
                    {
                        "canary_report": str(canary_report_path),
                        "isolation_manifest": str(isolation_manifest),
                        "isolation_data_root": isolation["data_root"],
                        "requested_targets": targets,
                        "combined_canary_and_batch": combine_ingestion_metrics(
                            canary_report["metrics"], report["metrics"]
                        ),
                        "prior_failed_runs": prior_failed_runs,
                        "combined_canary_batch_and_prior_failures": (
                            combine_ingestion_metrics_with_carryovers(
                                canary_report["metrics"],
                                report["metrics"],
                                prior_failed_runs,
                            )
                        ),
                        "cumulative_canary_target_charged_usd": (
                            sum(
                                float(item["total_importer_charged_usd"])
                                for item in prior_failed_runs
                                if item.get("target") == excluded
                            )
                            + sum(
                                float(item.get("charged_usd", 0.0))
                                for item in canary_report.get("results", [])
                                if isinstance(item, Mapping)
                                and item.get("target") == excluded
                            )
                        ),
                        "run_ledger": _ingestion_ledger_binding(proposal_root),
                    }
                )
                report_path = write_json_atomic(
                    proposal_root / "ingestion-batch-report.json", report
                )
            payload = {
                "job": args.agent_job,
                "report": str(report_path),
                **report,
                "budget": ledger.snapshot(),
            }
            if report["metrics"]["passed"] is not True:
                exit_code = 2
            if (
                args.agent_job == "ingestion-batch"
                and report["combined_canary_batch_and_prior_failures"]["passed"]
                is not True
            ):
                exit_code = 2
        else:  # pragma: no cover - argparse enforces this boundary.
            raise ValueError(f"unsupported agent job {args.agent_job!r}")
    except Exception as exc:
        raise RuntimeError(_agent_failure_reason(exc, key)) from None
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contraption")
    commands = parser.add_subparsers(dest="command", required=True)

    def assembly_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--spec",
            required=True,
            help="path to a self-contained contraption-4 manifest",
        )

    def controller_input_options(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--controller-input",
            action="append",
            default=[],
            metavar="NAME=JSON",
            help=(
                "set one declared external controller pin to a JSON number or boolean; "
                "repeat for multiple pins"
            ),
        )
        command.add_argument(
            "--controller-input-file",
            metavar="PATH",
            help="strict JSON object containing external controller pin values",
        )

    def simulation_options(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--duration",
            type=float,
            help="simulation seconds; defaults to environment.mission.duration_s, then 1.0",
        )
        command.add_argument("--samples", type=int, default=1)
        command.add_argument(
            "--dt",
            type=float,
            help="physics step; defaults to the greatest common controller subdivision",
        )
        command.add_argument("--seed", type=int, default=20260806)
        command.add_argument(
            "--backend", choices=("numpy", "torch", "auto"), default="numpy"
        )
        command.add_argument("--device")
        command.add_argument("--model-uncertainty", action="store_true")
        command.add_argument("--process-noise", action="store_true")
        controller_input_options(command)

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
        "simulate", help="simulate one canonical resolved contraption"
    )
    assembly_inputs(simulate_command)
    simulation_options(simulate_command)
    simulate_command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    simulate_command.add_argument(
        "--optical-capture",
        action="store_true",
        help="capture optical-observation-1 artifacts at an exact simulation pose",
    )
    simulate_command.add_argument(
        "--optical-scene",
        help="strict optical-scene-1 manifest containing external world geometry",
    )
    simulate_command.add_argument(
        "--optical-sensor",
        action="append",
        default=[],
        metavar="ID",
        help="capture one local or component-qualified optical sensor ID; repeat as needed",
    )
    simulate_command.add_argument("--optical-width", type=int, default=160)
    simulate_command.add_argument("--optical-height", type=int, default=90)
    simulate_command.add_argument(
        "--optical-frame-count",
        type=int,
        help="evenly spaced captures; defaults to environment.mission.camera_frame_count, then 1",
    )
    simulate_command.add_argument(
        "--optical-backend", choices=("numpy", "torch", "auto"), default="numpy"
    )
    simulate_command.add_argument("--optical-device")
    simulate_command.add_argument("--optical-sample-index", type=int, default=0)
    simulate_command.add_argument("--optical-time-index", type=int)
    simulate_command.add_argument("--optical-seed", type=int)
    simulate_command.set_defaults(handler=command_simulate)

    reconstruct = commands.add_parser(
        "optical-reconstruct",
        help="fuse verified depth observations into a sparse Bayesian TSDF/occupancy map",
    )
    reconstruct.add_argument(
        "--sensor",
        action="append",
        required=True,
        metavar="PATH",
        help="optical-sensor-1 descriptor; repeat for heterogeneous observations",
    )
    reconstruct.add_argument(
        "--observation",
        action="append",
        required=True,
        metavar="PATH",
        help="optical-observation-1 manifest to fuse; repeat for multi-view updates",
    )
    reconstruct.add_argument("--output", required=True)
    reconstruct.add_argument("--id", default="optical-reconstruction")
    reconstruct.add_argument("--voxel-size", type=float, default=0.01)
    reconstruct.add_argument("--block-size", type=int, default=8)
    reconstruct.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
    )
    reconstruct.add_argument("--truncation-distance", type=float)
    reconstruct.add_argument("--pixel-stride", type=int, default=1)
    reconstruct.add_argument(
        "--surface-occupancy-threshold", type=float, default=0.55
    )
    reconstruct.add_argument(
        "--surface-maximum-abs-tsdf", type=float, default=0.5
    )
    reconstruct.add_argument(
        "--surface-maximum-occupied-voxels", type=int, default=250_000
    )
    reconstruct.add_argument(
        "--surface-maximum-triangles", type=int, default=2_000_000
    )
    reconstruct.set_defaults(handler=command_optical_reconstruct)

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
        help="compile every resolved controller to complete C99 and fixed-point Verilog",
    )
    assembly_inputs(compile_command)
    compile_command.add_argument(
        "--output", default=str(DEFAULT_OUTPUT / "controllers")
    )
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
    serve.add_argument("--duration", type=float, default=1.0)
    serve.add_argument(
        "--dt",
        type=float,
        help="physics step; defaults to the greatest common controller subdivision",
    )
    serve.add_argument("--seed", type=int, default=20260806)
    serve.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    serve.add_argument("--device")
    controller_input_options(serve)
    serve.set_defaults(handler=command_serve)

    demo = commands.add_parser(
        "demo", help="simulate, view, compile controllers, and plan one contraption"
    )
    assembly_inputs(demo)
    simulation_options(demo)
    demo.add_argument("--output", default=str(DEFAULT_OUTPUT))
    demo.set_defaults(handler=command_demo)
    budget = commands.add_parser("budget", help="show the hard agent-dollar ledger")
    budget.add_argument("--ledger")
    budget.set_defaults(handler=command_budget)
    part_markdown = commands.add_parser(
        "part-markdown",
        help="render a validated part directory as standalone deterministic Markdown",
    )
    part_markdown.add_argument(
        "--part-directory",
        required=True,
        help="domain/category[/device]/instantiations/part directory",
    )
    part_markdown.add_argument(
        "--catalog",
        help="catalog root; inferred from the parent interface when omitted",
    )
    part_markdown.add_argument(
        "--output",
        help="output Markdown path; defaults to README.md in the part directory",
    )
    part_markdown.set_defaults(handler=command_part_markdown)
    canary = commands.add_parser("agent-canary", help="run guarded classification/modeling canaries")
    canary.add_argument("--kind", choices=("both", "classification", "modeling"), default="both")
    canary.add_argument("--ledger")
    canary.add_argument("--env-file")
    canary.set_defaults(handler=command_agent_canary)
    actual = commands.add_parser(
        "agent-run",
        help="run paid, resumable classification, modeling, or promoted ingestion jobs",
    )
    actual_jobs = actual.add_subparsers(dest="agent_job", required=True)

    def actual_common(
        command: argparse.ArgumentParser, *, output_root_required: bool = False
    ) -> None:
        command.add_argument("--ledger")
        command.add_argument("--env-file")
        command.add_argument(
            "--job-file",
            required=True,
            help="path to a portable agent-jobs-1 inventory",
        )
        command.add_argument("--output-root", required=output_root_required)
        command.add_argument(
            "--force",
            action="store_true",
            help="dispatch again even when the exact input hash already completed",
        )
        command.set_defaults(handler=command_agent_run)

    classify_all = actual_jobs.add_parser(
        "classification-all", help="classify every component record in an agent job file"
    )
    actual_common(classify_all)
    modeling_one = actual_jobs.add_parser(
        "modeling-one", help="stage one full modeling proposal without promoting it"
    )
    actual_common(modeling_one)
    modeling_one.add_argument("--target", required=True)
    modeling_one.add_argument("--staging-root")
    ingestion_canary = actual_jobs.add_parser(
        "ingestion-canary",
        help=(
            "run one fresh classification+direct-modeling canary in an isolated "
            "catalog and write the inclusive cost/validation gate"
        ),
    )
    actual_common(ingestion_canary, output_root_required=True)
    ingestion_canary.add_argument(
        "--ledger-limit-usd",
        type=float,
        default=0.50,
        help="lifetime limit for this replay ledger (default: 0.50)",
    )
    ingestion_canary.add_argument("--target", required=True)
    ingestion_canary.add_argument(
        "--prior-failed-ledger",
        action="append",
        default=[],
        help=(
            "dedicated failed ingestion ledger to bind into final cumulative KPIs; "
            "repeat for multiple prior failed runs"
        ),
    )
    ingestion_canary.add_argument("--staging-root")
    ingestion_batch = actual_jobs.add_parser(
        "ingestion-batch",
        help=(
            "run an isolated direct-Responses batch only after a matching canary gate"
        ),
    )
    actual_common(ingestion_batch, output_root_required=True)
    ingestion_batch.add_argument(
        "--ledger-limit-usd",
        type=float,
        default=0.50,
        help="lifetime limit for this replay ledger (default: 0.50)",
    )
    ingestion_batch.add_argument("--canary-report", required=True)
    ingestion_batch.add_argument(
        "--prior-failed-ledger",
        action="append",
        default=[],
        help=(
            "same failed ingestion ledger binding supplied to the canary; repeat "
            "for multiple prior failed runs"
        ),
    )
    ingestion_batch.add_argument(
        "--target",
        action="append",
        help="target id to ingest; repeat, or omit for every non-canary target",
    )
    ingestion_batch.add_argument("--staging-root")
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
