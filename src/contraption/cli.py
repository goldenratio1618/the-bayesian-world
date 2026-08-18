"""Command-line entry points for validation, demos, compilation, and agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np

from .catalog.interfaces import interface_paths, load_interface_catalog
from .control import compile_resolved_controller, control_digest
from .live import LiveApplication, scene_from_result
from .loading import load_contraption
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
PART_IMPORT_CANARY_ROOT = (
    PROJECT_ROOT / "assembled_contraptions" / "examples" / "part_import_canary"
)
DEFAULT_CATALOG = PROJECT_ROOT / "model_catalog"
DEFAULT_OUTPUT = OUTPUT_ROOT / "contraption_run"
AGENT_PROPOSALS = OUTPUT_ROOT / "agent-proposals"
AGENT_STAGING = OUTPUT_ROOT / "agent-staging"


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


def _agent_ledger(path: str | None) -> BudgetLedger:
    return BudgetLedger(
        Path(path).resolve() if path else OUTPUT_ROOT / "agent-budget.json",
        limit_usd=100.0,
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


def command_agent_run(args: argparse.Namespace) -> int:
    """Run paid, non-canary jobs with resumable validated proposal receipts."""

    ledger = _agent_ledger(args.ledger)
    key, dotenv = _agent_key(args.env_file)
    proposal_root = Path(args.output_root).resolve()
    jobs = _load_agent_job_bundle(args.job_file)
    try:
        if args.agent_job == "classification-all":
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
            agent = ModelingAgent(
                ledger,
                Path(args.staging_root).resolve(),
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
        help="run paid, resumable classification or modeling jobs without promotion",
    )
    actual_jobs = actual.add_subparsers(dest="agent_job", required=True)

    def actual_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--ledger")
        command.add_argument("--env-file")
        command.add_argument(
            "--job-file",
            required=True,
            help="path to a portable agent-jobs-1 inventory",
        )
        command.add_argument("--output-root", default=str(AGENT_PROPOSALS))
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
