"""Generic live simulation state for a resolved contraption.

This module contains no device-specific policy.  External pins are discovered
from every resolved controller, simulation is delegated to the canonical
offline simulator, and physical frames are reconstructed by the resolved
assembly itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import re
import threading
from typing import Any

from .control import ExplicitInputSpec, control_digest
from .physics.resolved import ResolvedAssembly
from .physics.simulator import controller_time_step, simulate
from .visualization.viewer import generate_viewer, validate_physical_scene


_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


class LiveRequestError(ValueError):
    """A bounded request error safe to return to a local viewer."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = str(code)


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " and positive" if positive else ""
        raise ValueError(f"{label} must be finite{suffix}")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "detach"):
        return _plain(value.detach().cpu().numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def scene_from_result(
    assembly: ResolvedAssembly,
    result: Any,
    *,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Reconstruct a generic physical scene from one exact simulated sample."""

    if not isinstance(assembly, ResolvedAssembly):
        raise TypeError("scene reconstruction requires a ResolvedAssembly")
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("simulation result has no metadata mapping")
    if metadata.get("assembly_sha256") != assembly.assembly_sha256:
        raise ValueError("simulation result was produced from another assembly")
    if sample_index is None:
        sample_index = metadata.get("pose_frame_sample_index", 0)
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ValueError("pose frame sample index must be an integer")
    scene = _plain(assembly.scene)
    scene["body_pose_frames"] = _plain(
        assembly.body_pose_frames(result, sample_index=sample_index)
    )
    return dict(validate_physical_scene(scene))


@dataclass(frozen=True, slots=True)
class LiveInput:
    """One authored external pin shared by one or more controllers."""

    name: str
    spec: ExplicitInputSpec
    consumers: tuple[str, ...]

    def normalize(self, value: Any) -> bool | float:
        if self.spec.dtype == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"live input {self.name!r} must be boolean")
            return value
        result = _finite(value, f"live input {self.name!r}")
        if not self.spec.bounds.contains(result):
            raise ValueError(
                f"live input {self.name!r}={result} is outside bounds "
                f"[{self.spec.bounds.lower}, {self.spec.bounds.upper}]"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.spec.dtype,
            "default": self.spec.default,
            "minimum": self.spec.bounds.lower,
            "maximum": self.spec.bounds.upper,
            "unit": self.spec.unit,
            "description": self.spec.description,
            "consumers": list(self.consumers),
        }


def _external_inputs(assembly: ResolvedAssembly) -> dict[str, LiveInput]:
    grouped: dict[str, list[tuple[str, ExplicitInputSpec]]] = {}
    for controller_id, controller in sorted(assembly.controllers.items()):
        declared = {item.name: item for item in controller.spec.explicit_inputs}
        for input_name, binding in controller.explicit_input_bindings.items():
            if binding.kind != "external":
                continue
            grouped.setdefault(binding.source, []).append(
                (f"{controller_id}.{input_name}", declared[input_name])
            )
    result: dict[str, LiveInput] = {}
    for source, declarations in sorted(grouped.items()):
        canonical = declarations[0][1]
        conflicts = [
            consumer
            for consumer, declaration in declarations[1:]
            if (
                declaration.dtype,
                declaration.unit,
                declaration.default,
                declaration.bounds,
            )
            != (
                canonical.dtype,
                canonical.unit,
                canonical.default,
                canonical.bounds,
            )
        ]
        if conflicts:
            raise ValueError(
                f"external pin {source!r} has incompatible controller declarations: "
                f"{[declarations[0][0], *conflicts]}"
            )
        if canonical.dtype == "real" and (
            canonical.bounds.lower is None
            or canonical.bounds.upper is None
            or canonical.bounds.upper <= canonical.bounds.lower
        ):
            raise ValueError(
                f"numeric live input {source!r} requires finite increasing bounds"
            )
        result[source] = LiveInput(
            source,
            canonical,
            tuple(consumer for consumer, _declaration in declarations),
        )
    return result


class LiveApplication:
    """Thread-safe live simulation state for any resolved contraption."""

    def __init__(
        self,
        assembly: ResolvedAssembly,
        *,
        duration: float = 1.0,
        dt: float | None = None,
        backend: str = "numpy",
        device: str | None = None,
        seed: int = 20260806,
        initial_inputs: Mapping[str, Any] | None = None,
        simulation: Callable[..., Any] = simulate,
        scene_builder: Callable[[ResolvedAssembly, Any], dict[str, Any]] = scene_from_result,
    ) -> None:
        if not isinstance(assembly, ResolvedAssembly):
            raise TypeError("live simulation requires a canonical ResolvedAssembly")
        periods = tuple(
            float(item.spec.period_s) for item in assembly.controllers.values()
        )
        requested_dt = None if dt is None else _finite(dt, "dt", positive=True)
        self.dt = controller_time_step(periods, requested_dt)
        self.assembly = assembly
        self.duration = _finite(duration, "duration", positive=True)
        self.backend = str(backend)
        self.device = device
        self.seed = int(seed)
        self._simulate = simulation
        self._scene_builder = scene_builder
        self._lock = threading.Lock()
        self._signals = _external_inputs(assembly)
        self._values = {
            name: signal.spec.default for name, signal in self._signals.items()
        }
        if initial_inputs is not None:
            if not isinstance(initial_inputs, Mapping) or any(
                not isinstance(name, str) for name in initial_inputs
            ):
                raise ValueError("initial live controller inputs must be a mapping")
            unknown = sorted(set(initial_inputs) - set(self._signals))
            if unknown:
                raise ValueError(
                    f"unknown initial live controller input(s) {unknown}; declared inputs "
                    f"are {sorted(self._signals)}"
                )
            self._values.update(
                {
                    name: self._signals[name].normalize(value)
                    for name, value in initial_inputs.items()
                }
            )
        self._controllers = tuple(
            {
                "id": controller.id,
                "program_id": controller.spec.id,
                "version": controller.spec.version,
                "sha256": control_digest(controller.spec),
            }
            for _name, controller in sorted(assembly.controllers.items())
        )
        self._result, self._scene = self._run(self._values)

    def _run(self, external_inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = self._simulate(
            self.assembly,
            duration=self.duration,
            dt=self.dt,
            num_samples=1,
            seed=self.seed,
            backend=self.backend,
            device=self.device,
            use_model_uncertainty=False,
            process_noise=False,
            controller_inputs=dict(external_inputs),
        )
        scene = dict(self._scene_builder(self.assembly, result))
        if scene.get("assembly_sha256") != self.assembly.assembly_sha256:
            raise ValueError("live simulation returned a scene for another assembly")
        return result, scene

    def control_schema(self) -> dict[str, Any]:
        with self._lock:
            values = dict(self._values)
        return {
            "schema": "contraption.live-controls/v2",
            "assembly_sha256": self.assembly.assembly_sha256,
            "controllers": [dict(item) for item in self._controllers],
            "inputs": [item.to_dict() for item in self._signals.values()],
            "values": values,
        }

    def simulate_request(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            raise LiveRequestError(400, "invalid_request", "request must be an object")
        if set(value) != {"assembly_sha256", "inputs"}:
            raise LiveRequestError(
                400,
                "invalid_request",
                "request must contain exactly assembly_sha256 and inputs",
            )
        supplied_hash = value["assembly_sha256"]
        if not isinstance(supplied_hash, str) or _HASH.fullmatch(supplied_hash) is None:
            raise LiveRequestError(
                400, "invalid_hash", "assembly_sha256 must be a canonical SHA-256"
            )
        if supplied_hash != self.assembly.assembly_sha256:
            raise LiveRequestError(
                409,
                "assembly_mismatch",
                "request assembly hash does not match the live resolved assembly",
            )
        raw_inputs = value["inputs"]
        if not isinstance(raw_inputs, Mapping) or any(
            not isinstance(key, str) for key in raw_inputs
        ):
            raise LiveRequestError(400, "invalid_controls", "inputs must be an object")
        unknown = sorted(set(raw_inputs) - set(self._signals))
        missing = sorted(set(self._signals) - set(raw_inputs))
        if unknown or missing:
            raise LiveRequestError(
                400,
                "invalid_controls",
                f"external inputs must match exactly; missing={missing}, unknown={unknown}",
            )
        try:
            normalized = {
                name: self._signals[name].normalize(raw_inputs[name])
                for name in self._signals
            }
        except Exception as exc:
            raise LiveRequestError(400, "invalid_controls", str(exc)) from exc
        with self._lock:
            try:
                result, scene = self._run(normalized)
            except Exception as exc:
                raise LiveRequestError(
                    422,
                    "simulation_failed",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            self._values = normalized
            self._result = result
            self._scene = scene
            return dict(scene)

    def viewer_files(self) -> Mapping[str, str]:
        with self._lock:
            result = self._result
        artifact = generate_viewer(
            self.assembly,
            result,
            sample_index=result.metadata.get("pose_frame_sample_index", 0),
            title=str(self.assembly.specification.name),
            live={
                "schema_endpoint": "/api/schema",
                "simulate_endpoint": "/api/simulate",
            },
        )
        return artifact.files


__all__ = [
    "LiveApplication",
    "LiveInput",
    "LiveRequestError",
    "scene_from_result",
]
