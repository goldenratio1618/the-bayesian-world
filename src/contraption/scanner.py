"""End-to-end apartment scanner-robot reference aggregate.

The controller executes the checked, data-only control program in vectorized
form.  Geometry needed by the estimator (orbit radius, look-at pitch, and
phase) is computed here from simulated sensor state; all actuator equations
remain in ``scanner.control.json``.  NumPy arrays and PyTorch tensors stay on
their selected backend throughout the closed-loop simulation.

This module does not assemble the physical PMDL network.  Its contraption-level
entry point therefore requires a hash-bound coverage contract that accounts for
every physical component and connection before the reduced aggregate may run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backend import Backend, infer_backend
from .controls import ControlProgram, evaluate_state_outputs
from .simulator import DifferentialDriveArmModel, SimulationResult, simulate
from .specs import ContraptionSpec


SCANNER_AGGREGATE_MODEL = "contraption.scanner.DifferentialDriveArmModel/v1"
SCANNER_COVERAGE_SCHEMA = "contraption.simulation-coverage/v1"
SCANNER_TOPOLOGY_SCHEMA = "contraption.simulation-topology/v1"
_SCANNER_COMPOSITION = "hand_authored_aggregate_not_pmdl_network"
_REPRESENTATION_KINDS = frozenset(
    {"state", "dynamics_parameter", "controller", "sensor", "geometry_only", "excluded"}
)


class ScannerSimulationCoverageError(ValueError):
    """The physical scanner cannot be represented by the declared aggregate.

    This exception is raised before integration.  It deliberately distinguishes
    an incomplete or stale aggregate-model admission contract from a numerical
    simulation failure.
    """


def _physical_spec(value: ContraptionSpec | Mapping[str, Any]) -> ContraptionSpec:
    if isinstance(value, ContraptionSpec):
        return value
    if not isinstance(value, Mapping):
        raise ScannerSimulationCoverageError(
            "physical_spec must be a ContraptionSpec or serialized contraption object"
        )
    try:
        return ContraptionSpec.from_dict(value)
    except Exception as exc:
        raise ScannerSimulationCoverageError(
            f"physical_spec is not a valid contraption: {type(exc).__name__}: {exc}"
        ) from exc


def _topology_payload(spec: ContraptionSpec) -> dict[str, Any]:
    """Return the canonical physical identity covered by the aggregate model."""

    component_ids = [component.id for component in spec.components]
    connection_ids = [connection.id for connection in spec.connections]
    duplicate_components = sorted(
        {identifier for identifier in component_ids if component_ids.count(identifier) > 1}
    )
    duplicate_connections = sorted(
        {identifier for identifier in connection_ids if connection_ids.count(identifier) > 1}
    )
    if duplicate_components or duplicate_connections:
        raise ScannerSimulationCoverageError(
            "physical_spec has duplicate topology identifiers: "
            f"components={duplicate_components}, connections={duplicate_connections}"
        )
    return {
        "schema": SCANNER_TOPOLOGY_SCHEMA,
        "contraption_id": spec.id,
        "components": sorted(
            ({"id": component.id, "model": component.model} for component in spec.components),
            key=lambda item: item["id"],
        ),
        "connections": sorted(
            (
                {
                    "id": connection.id,
                    "kind": connection.kind,
                    "domain": connection.domain,
                    "endpoints": [
                        f"{endpoint.component}.{endpoint.port}" for endpoint in connection.endpoints
                    ],
                }
                for connection in spec.connections
            ),
            key=lambda item: item["id"],
        ),
    }


def scanner_topology_sha256(
    physical_spec: ContraptionSpec | Mapping[str, Any],
) -> str:
    """Hash component/model and connection topology identity canonically."""

    payload = _topology_payload(_physical_spec(physical_spec))
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_scanner_simulation_coverage(path: str | Path) -> dict[str, Any]:
    """Load a scanner aggregate coverage contract without accepting non-objects."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScannerSimulationCoverageError(
            "scanner simulation coverage file must contain a JSON object"
        )
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ScannerSimulationCoverageError(
            f"{context} has unknown field(s): {', '.join(unknown)}"
        )
    if missing:
        raise ScannerSimulationCoverageError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _string_array(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScannerSimulationCoverageError(f"{context} must be an array of strings")
    result = list(value)
    if (not allow_empty and not result) or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        qualifier = "non-empty " if not allow_empty else ""
        raise ScannerSimulationCoverageError(
            f"{context} must be a {qualifier}array of non-empty strings"
        )
    if len(result) != len(set(result)):
        raise ScannerSimulationCoverageError(f"{context} may not contain duplicates")
    return result


def _validate_representations(value: Any, context: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ScannerSimulationCoverageError(
            f"{context}.representations must be a non-empty array"
        )
    kinds: list[str] = []
    for index, raw in enumerate(value):
        item_context = f"{context}.representations[{index}]"
        if not isinstance(raw, Mapping):
            raise ScannerSimulationCoverageError(f"{item_context} must be an object")
        _strict_keys(
            raw,
            allowed={"kind", "targets", "limitation"},
            required={"kind", "targets"},
            context=item_context,
        )
        kind = raw["kind"]
        if kind not in _REPRESENTATION_KINDS:
            raise ScannerSimulationCoverageError(
                f"{item_context}.kind must be one of {sorted(_REPRESENTATION_KINDS)}, "
                f"got {kind!r}"
            )
        if kind in kinds:
            raise ScannerSimulationCoverageError(
                f"{context} has duplicate representation kind {kind!r}"
            )
        kinds.append(kind)
        targets = _string_array(
            raw["targets"], f"{item_context}.targets", allow_empty=kind == "excluded"
        )
        limitation = raw.get("limitation", "")
        if not isinstance(limitation, str):
            raise ScannerSimulationCoverageError(f"{item_context}.limitation must be a string")
        if kind == "excluded":
            if targets:
                raise ScannerSimulationCoverageError(
                    f"{item_context} is excluded and therefore may not declare targets"
                )
            if not limitation.strip():
                raise ScannerSimulationCoverageError(
                    f"{item_context} exclusion requires a non-empty rationale/limitation"
                )
    if "excluded" in kinds and len(kinds) != 1:
        raise ScannerSimulationCoverageError(
            f"{context} cannot combine 'excluded' with represented behavior"
        )


def _coverage_entries(value: Any, context: str) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScannerSimulationCoverageError(f"coverage.{context} must be an array")
    entries: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(value):
        item_context = f"coverage.{context}[{index}]"
        if not isinstance(raw, Mapping):
            raise ScannerSimulationCoverageError(f"{item_context} must be an object")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ScannerSimulationCoverageError(f"{item_context}.id must be a non-empty string")
        if identifier in entries:
            raise ScannerSimulationCoverageError(
                f"coverage.{context} has duplicate id {identifier!r}"
            )
        entries[identifier] = raw
    return entries


def validate_scanner_simulation_coverage(
    physical_spec: ContraptionSpec | Mapping[str, Any],
    simulation_coverage: Mapping[str, Any],
) -> str:
    """Validate the aggregate-model admission contract and return its topology hash.

    The coverage file is not a PMDL network assembly.  It is a fail-closed
    inventory explaining how every physical component and connection is (or is
    not) represented by one hand-authored aggregate dynamics model.
    """

    spec = _physical_spec(physical_spec)
    topology = _topology_payload(spec)
    if not isinstance(simulation_coverage, Mapping):
        raise ScannerSimulationCoverageError(
            "simulation_coverage must be the parsed coverage JSON object"
        )
    _strict_keys(
        simulation_coverage,
        allowed={
            "schema",
            "contraption_id",
            "aggregate_model",
            "composition",
            "topology_sha256",
            "components",
            "connections",
            "limitations",
        },
        required={
            "schema",
            "contraption_id",
            "aggregate_model",
            "composition",
            "topology_sha256",
            "components",
            "connections",
            "limitations",
        },
        context="simulation_coverage",
    )
    expected_constants = {
        "schema": SCANNER_COVERAGE_SCHEMA,
        "contraption_id": spec.id,
        "aggregate_model": SCANNER_AGGREGATE_MODEL,
        "composition": _SCANNER_COMPOSITION,
    }
    for field, expected in expected_constants.items():
        actual = simulation_coverage[field]
        if actual != expected:
            raise ScannerSimulationCoverageError(
                f"simulation_coverage.{field} drift: expected {expected!r}, got {actual!r}"
            )
    _string_array(simulation_coverage["limitations"], "simulation_coverage.limitations")

    component_entries = _coverage_entries(simulation_coverage["components"], "components")
    physical_components = {component.id: component for component in spec.components}
    missing_components = sorted(set(physical_components) - set(component_entries))
    extra_components = sorted(set(component_entries) - set(physical_components))
    if missing_components or extra_components:
        raise ScannerSimulationCoverageError(
            "component coverage mismatch; "
            f"missing component coverage={missing_components}, stale/extra coverage={extra_components}"
        )
    for identifier, component in physical_components.items():
        entry = component_entries[identifier]
        context = f"coverage component {identifier!r}"
        _strict_keys(
            entry,
            allowed={"id", "model", "representations"},
            required={"id", "model", "representations"},
            context=context,
        )
        if entry["model"] != component.model:
            raise ScannerSimulationCoverageError(
                f"{context} model drift: physical spec references {component.model!r}, "
                f"coverage references {entry['model']!r}"
            )
        _validate_representations(entry["representations"], context)

    connection_entries = _coverage_entries(simulation_coverage["connections"], "connections")
    physical_connections = {connection.id: connection for connection in spec.connections}
    missing_connections = sorted(set(physical_connections) - set(connection_entries))
    extra_connections = sorted(set(connection_entries) - set(physical_connections))
    if missing_connections or extra_connections:
        raise ScannerSimulationCoverageError(
            "connection coverage mismatch; "
            f"missing connection coverage={missing_connections}, "
            f"stale/extra coverage={extra_connections}"
        )
    for identifier, connection in physical_connections.items():
        entry = connection_entries[identifier]
        context = f"coverage connection {identifier!r}"
        _strict_keys(
            entry,
            allowed={"id", "kind", "domain", "endpoints", "representations"},
            required={"id", "kind", "domain", "endpoints", "representations"},
            context=context,
        )
        expected_connection = {
            "kind": connection.kind,
            "domain": connection.domain,
            "endpoints": [
                f"{endpoint.component}.{endpoint.port}" for endpoint in connection.endpoints
            ],
        }
        for field, expected in expected_connection.items():
            if entry[field] != expected:
                raise ScannerSimulationCoverageError(
                    f"{context} {field} drift: physical spec has {expected!r}, "
                    f"coverage has {entry[field]!r}"
                )
        _validate_representations(entry["representations"], context)

    expected_hash = scanner_topology_sha256(spec)
    actual_hash = simulation_coverage["topology_sha256"]
    if actual_hash != expected_hash:
        raise ScannerSimulationCoverageError(
            "simulation_coverage.topology_sha256 is stale or invalid: "
            f"expected {expected_hash!r}, got {actual_hash!r}"
        )
    # Assert that the hashed material and the individually checked material are
    # exactly the same contract, rather than maintaining two drifting views.
    if topology["contraption_id"] != simulation_coverage["contraption_id"]:
        raise AssertionError("validated coverage/topology identity diverged")
    return expected_hash


@dataclass(frozen=True)
class ScannerMission:
    """Physical mission inputs used by both controller and acceptance checks."""

    object_center: tuple[float, float, float] = (0.0, 0.0, 0.35)
    object_side_m: float = 0.40
    orbit_radius_m: float = 0.85
    tangential_speed_m_s: float = 0.12
    duration_s: float = 45.0
    vertical_cycles_per_orbit: float = 2.0
    keep_out_radius_m: float = 0.50
    camera_yaw_offset_rad: float = math.pi / 2.0
    maximum_voltage_v: float = 4.5

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScannerMission":
        cube = value.get("object_bounding_cube", {})
        mission = value.get("mission", {})
        center = tuple(float(item) for item in cube.get("center_m", (0.0, 0.0, 0.35)))
        if len(center) != 3:
            raise ValueError("object_bounding_cube.center_m must have three entries")
        result = cls(
            object_center=center,  # type: ignore[arg-type]
            object_side_m=float(cube.get("side_m", 0.40)),
            orbit_radius_m=float(mission.get("orbit_radius_m", 0.85)),
            tangential_speed_m_s=float(mission.get("tangential_speed_m_s", 0.12)),
            duration_s=float(mission.get("duration_s", 45.0)),
            vertical_cycles_per_orbit=float(mission.get("vertical_cycles_per_orbit", 2.0)),
            keep_out_radius_m=float(value.get("keep_out_radius_m", 0.50)),
            camera_yaw_offset_rad=float(mission.get("camera_yaw_offset_rad", math.pi / 2.0)),
            maximum_voltage_v=float(mission.get("maximum_voltage_v", 4.5)),
        )
        if result.object_side_m <= 0 or result.orbit_radius_m <= result.keep_out_radius_m:
            raise ValueError("scanner geometry must have a positive cube and orbit outside keep-out")
        if result.duration_s <= 0 or result.tangential_speed_m_s < 0:
            raise ValueError("scanner duration must be positive and speed nonnegative")
        return result


class ScannerOrbitController:
    """Backend-preserving adapter from robot state to the restricted controller.

    The normal mission fixes the Moore machine in its ``scanning`` state.  The
    emergency-stop path is deliberately outside that expression evaluation so
    it can dominate even if a sensor expression later becomes invalid.
    """

    def __init__(
        self,
        program: ControlProgram,
        mission: ScannerMission,
        *,
        wheel_radius_m: float = 0.035,
        nominal_traction: float = 0.82,
        base_height_m: float = 0.22,
        arm_length_m: float = 0.18,
        mount_offset_m: float = 0.03,
    ) -> None:
        self.program = program
        self.mission = mission
        self.wheel_radius_m = float(wheel_radius_m)
        self.nominal_traction = float(nominal_traction)
        self.base_height_m = float(base_height_m)
        self.arm_length_m = float(arm_length_m)
        self.mount_offset_m = float(mount_offset_m)
        self.emergency_stop = False

    def reset(self) -> None:
        self.emergency_stop = False

    @staticmethod
    def _wrapped(value: Any, backend: Backend) -> Any:
        return backend.remainder(value + math.pi, 2.0 * math.pi) - math.pi

    def evaluate(self, t: Any, state: Any, backend: Backend) -> Mapping[str, Any]:
        count = int(state.shape[0])
        if self.emergency_stop:
            zeros = backend.zeros((count,))
            return {
                "left_voltage": zeros,
                "right_voltage": zeros,
                "arm_command": state[:, 5],
                "camera_pitch": state[:, 6],
            }

        ox, oy, oz = self.mission.object_center
        dx = state[:, 0] - ox
        dy = state[:, 1] - oy
        radius = backend.sqrt(dx * dx + dy * dy + 1e-12)
        phase = backend.atan2(dy, dx)
        tangent_heading = phase + math.pi / 2.0
        heading_error = self._wrapped(tangent_heading - state[:, 2], backend)
        measured_speed = (
            0.5
            * self.wheel_radius_m
            * self.nominal_traction
            * (state[:, 3] + state[:, 4])
        )

        arm_radial = self.mount_offset_m + self.arm_length_m * backend.cos(state[:, 5])
        arm_yaw = state[:, 2] + self.mission.camera_yaw_offset_rad
        camera_x = state[:, 0] + arm_radial * backend.cos(arm_yaw)
        camera_y = state[:, 1] + arm_radial * backend.sin(arm_yaw)
        camera_z = self.base_height_m + self.arm_length_m * backend.sin(state[:, 5])
        look_dx = ox - camera_x
        look_dy = oy - camera_y
        planar_distance = backend.sqrt(look_dx * look_dx + look_dy * look_dy + 1e-12)
        required_tilt = backend.atan2(oz - camera_z, planar_distance)

        context = {
            "armed": True,
            "reset": False,
            "emergency_stop": False,
            "target_speed": self.mission.tangential_speed_m_s,
            "orbit_radius": self.mission.orbit_radius_m,
            "measured_speed": measured_speed,
            "radius_error": radius - self.mission.orbit_radius_m,
            "heading_error": heading_error,
            "orbit_phase": phase,
            "required_tilt": required_tilt,
            "lift_feedback": state[:, 5],
            "tilt_feedback": state[:, 6],
            "drive_fault": False,
            "brownout": False,
            "time": t,
        }
        outputs = evaluate_state_outputs(self.program, "scanning", context, backend)
        return {
            "left_voltage": outputs["left_voltage"],
            "right_voltage": outputs["right_voltage"],
            "arm_command": outputs["lift_target"],
            "camera_pitch": outputs["tilt_target"],
        }


@dataclass
class ScannerAggregateModel:
    """Low-level hand-authored aggregate used for dynamics-focused tests.

    This object is *not* a PMDL-composed contraption.  Physical contraption
    simulation must enter through :func:`simulate_scanner_robot`, which checks
    the complete simulation-coverage contract before constructing this model.
    """

    dynamics: DifferentialDriveArmModel
    controller: ScannerOrbitController


# Backwards-compatible type spelling.  The aggregate name above is preferred
# because it does not imply that a PMDL network was assembled.
ScannerRobotContraption = ScannerAggregateModel


def load_scanner_mission(path: str | Path) -> ScannerMission:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("scanner parameter file must contain an object")
    return ScannerMission.from_dict(data)


def make_scanner_aggregate_model(
    program: ControlProgram,
    mission: ScannerMission | None = None,
) -> ScannerAggregateModel:
    """Construct only the low-level aggregate dynamics/controller adapter.

    Use this API for isolated unit tests of the reduced equations.  It performs
    no claim of component or connection coverage.
    """

    mission = mission or ScannerMission()
    parameters = {
        "wheel_radius": 0.035,
        "wheel_base": 0.142,
        "motor_gain": 3.2,
        "motor_time_constant": 0.16,
        "rolling_drag": 0.04,
        "traction_left": 0.82,
        "traction_right": 0.82,
        "lateral_slip": 0.0,
        "arm_time_constant": 0.20,
        "camera_time_constant": 0.16,
        "arm_min": -0.35,
        "arm_max": 0.85,
        "camera_pitch_min": -0.70,
        "camera_pitch_max": 0.70,
        "mount_offset": 0.03,
        "arm_length": 0.18,
        "base_height": 0.22,
        "roughness_std": 0.001,
        "contact_jitter_std": 0.004,
    }
    # Newer model revisions expose this mount transform.  Keeping this check
    # allows old serialized experiments to remain runnable.
    if "arm_azimuth_offset" in DifferentialDriveArmModel().default_parameters:
        parameters["arm_azimuth_offset"] = mission.camera_yaw_offset_rad
    dynamics = DifferentialDriveArmModel(parameters)
    controller = ScannerOrbitController(program, mission)
    return ScannerAggregateModel(dynamics=dynamics, controller=controller)


def make_scanner_robot(
    program: ControlProgram,
    mission: ScannerMission | None = None,
) -> ScannerAggregateModel:
    """Compatibility alias for :func:`make_scanner_aggregate_model`.

    Despite the historical name, this returns the reduced aggregate only and
    does not validate a physical contraption coverage contract.
    """

    return make_scanner_aggregate_model(program, mission)


def simulate_scanner_robot(
    program: ControlProgram,
    mission: ScannerMission | None = None,
    *,
    physical_spec: ContraptionSpec | Mapping[str, Any] | None = None,
    simulation_coverage: Mapping[str, Any] | None = None,
    duration: float | None = None,
    dt: float = 0.05,
    num_samples: int = 128,
    seed: int = 20260806,
    backend: str | Backend = "numpy",
    device: str | None = None,
    process_noise: bool = True,
) -> SimulationResult:
    """Run the covered physical scanner through its aggregate approximation.

    The physical spec and its hash-bound coverage contract are mandatory.  A
    missing, extra, excluded-without-rationale, or stale component/connection
    aborts before integration; this function never silently drops topology.
    """

    if physical_spec is None:
        raise ScannerSimulationCoverageError(
            "physical_spec is required for contraption-level scanner simulation; "
            "use make_scanner_aggregate_model only for low-level aggregate tests"
        )
    if simulation_coverage is None:
        raise ScannerSimulationCoverageError(
            "simulation_coverage is required for contraption-level scanner simulation; "
            "the hand-authored aggregate may not silently omit physical elements"
        )
    topology_sha256 = validate_scanner_simulation_coverage(
        physical_spec, simulation_coverage
    )

    mission = mission or ScannerMission()
    robot = make_scanner_aggregate_model(program, mission)
    distributions = {
        "wheel_radius": {"mean": 0.035, "std": 0.0007, "lower": 0.032, "upper": 0.038},
        "wheel_base": {"mean": 0.142, "std": 0.003, "lower": 0.13, "upper": 0.155},
        "traction_left": {"mean": 0.82, "std": 0.08, "lower": 0.45, "upper": 1.0},
        "traction_right": {"mean": 0.82, "std": 0.08, "lower": 0.45, "upper": 1.0},
        "motor_gain": {"mean": 3.2, "std": 0.16, "lower": 2.5, "upper": 3.8},
    }
    initial = [mission.orbit_radius_m, 0.0, math.pi / 2.0, 0.0, 0.0, 0.25, 0.0]
    result = simulate(
        robot,
        duration=mission.duration_s if duration is None else float(duration),
        dt=dt,
        parameter_distribution=distributions,
        initial_state=initial,
        num_samples=num_samples,
        seed=seed,
        backend=backend,
        device=device,
        process_noise=process_noise,
    )
    return replace(
        result,
        metadata={
            **dict(result.metadata),
            "simulation_scope": "validated_hand_authored_scanner_aggregate",
            "physical_contraption_id": simulation_coverage["contraption_id"],
            "simulation_coverage_topology_sha256": topology_sha256,
            "pmdl_network_composed": False,
        },
    )


def scanner_metrics(
    result: SimulationResult,
    mission: ScannerMission | None = None,
    *,
    warmup_s: float = 2.0,
) -> dict[str, Any]:
    """Compute geometric coverage, tracking, pointing, and safety evidence."""

    mission = mission or ScannerMission()
    numerical = infer_backend(result.samples)
    states = numerical.to_numpy(result.samples)
    outputs = numerical.to_numpy(result.output_samples)
    times = numerical.to_numpy(result.time)
    start = int(np.searchsorted(times, warmup_s, side="left"))
    start = min(start, states.shape[1] - 1)
    states = states[:, start:, :]
    outputs = outputs[:, start:, :]

    ox, oy, oz = mission.object_center
    radius = np.hypot(states[..., 0] - ox, states[..., 1] - oy)
    radius_error = radius - mission.orbit_radius_m
    camera_x = outputs[..., result.output_names.index("camera_x")]
    camera_y = outputs[..., result.output_names.index("camera_y")]
    camera_z = outputs[..., result.output_names.index("camera_z")]
    yaw = states[..., result.state_names.index("yaw")]
    pitch = states[..., result.state_names.index("camera_pitch")]
    optical_yaw = yaw + mission.camera_yaw_offset_rad
    optical = np.stack(
        (
            np.cos(pitch) * np.cos(optical_yaw),
            np.cos(pitch) * np.sin(optical_yaw),
            np.sin(pitch),
        ),
        axis=-1,
    )
    target = np.stack((ox - camera_x, oy - camera_y, oz - camera_z), axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    cosine = np.sum(optical * target, axis=-1) / np.maximum(target_norm, 1e-12)
    pointing_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    # A conservative planar keep-out protects the complete object cube and a
    # margin.  Collision probability is the fraction of sampled missions that
    # ever enter it after warmup.
    collision_by_sample = np.any(radius <= mission.keep_out_radius_m, axis=1)
    collision_count = int(np.count_nonzero(collision_by_sample))
    sample_count = int(states.shape[0])
    collision_estimate = collision_count / sample_count
    # One-sided 95% Wilson upper confidence bound.  A zero observed count in a
    # small Monte Carlo batch is not evidence for an arbitrarily tiny risk.
    z_95_one_sided = 1.6448536269514722
    z2 = z_95_one_sided * z_95_one_sided
    denominator = 1.0 + z2 / sample_count
    center = (collision_estimate + z2 / (2.0 * sample_count)) / denominator
    half_width = (
        z_95_one_sided
        / denominator
        * math.sqrt(
            collision_estimate * (1.0 - collision_estimate) / sample_count
            + z2 / (4.0 * sample_count * sample_count)
        )
    )
    collision_upper_95 = min(1.0, center + half_width)
    phases = np.unwrap(np.arctan2(states[..., 1] - oy, states[..., 0] - ox), axis=1)
    coverage = np.abs(phases[:, -1] - phases[:, 0])
    values = {
        "sample_count": sample_count,
        "evaluated_duration_s": float(times[-1] - times[start]),
        "orbit_radius_rmse_m": float(np.sqrt(np.mean(radius_error * radius_error))),
        "orbit_radius_p95_absolute_error_m": float(np.quantile(np.abs(radius_error), 0.95)),
        "camera_pointing_p95_deg": float(np.quantile(pointing_deg, 0.95)),
        "camera_pointing_max_deg": float(np.max(pointing_deg)),
        "collision_count": collision_count,
        "collision_probability": collision_estimate,
        "collision_probability_upper_95_wilson": collision_upper_95,
        "minimum_chassis_keepout_clearance_m": float(np.min(radius - mission.keep_out_radius_m)),
        "camera_height_range_m": [float(np.min(camera_z)), float(np.max(camera_z))],
        "median_orbit_coverage_deg": float(np.degrees(np.median(coverage))),
    }
    values["acceptance"] = {
        "orbit_radius_rmse": values["orbit_radius_rmse_m"] <= 0.12,
        "camera_pointing_p95": values["camera_pointing_p95_deg"] <= 6.0,
        "collision_probability": values["collision_probability_upper_95_wilson"]
        <= 0.001,
    }
    values["accepted"] = all(values["acceptance"].values())
    return values


__all__ = [
    "SCANNER_AGGREGATE_MODEL",
    "SCANNER_COVERAGE_SCHEMA",
    "ScannerAggregateModel",
    "ScannerMission",
    "ScannerOrbitController",
    "ScannerRobotContraption",
    "ScannerSimulationCoverageError",
    "load_scanner_simulation_coverage",
    "load_scanner_mission",
    "make_scanner_aggregate_model",
    "make_scanner_robot",
    "scanner_metrics",
    "scanner_topology_sha256",
    "simulate_scanner_robot",
    "validate_scanner_simulation_coverage",
]
