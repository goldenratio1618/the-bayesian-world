"""Canonical scanner-robot runtime built from a :class:`ResolvedAssembly`.

This module contains mission policy, not another physical model.  Every state,
parameter, control target, connector frame, and body transform used here is
discovered from the verified component-package/PMDL closure.  The generic PMDL
descriptor simulator advances that closure; the physical resolver reconstructs
and validates every displayed configuration from the same state trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backend import Backend, infer_backend
from .controls import (
    ControlProgram,
    ControllerRuntime,
    evaluate_state_outputs,
    load_control_program,
)
from .dsl import load_model
from .paths import asset_root
from .physical import ComponentPackageRegistry
from .resolved import DynamicsCompletenessRecord, ResolvedAssembly, resolve_assembly
from .simulator import SimulationResult, simulate


class ScannerRuntimeError(ValueError):
    """The resolved closure cannot support the scanner mission unambiguously."""


_SCANNER_DYNAMICS_GATE_IDS = frozenset(
    {
        "fixed_payload_mass_inertia",
        "moving_arm_camera_inertial_derivation",
        "servo_case_reaction_coupling",
        "caster_floor_contact",
        "full_body_keepout",
        "controller_sensor_observation_binding",
        "electrical_supply_and_fault_coupling",
    }
)
_SCANNER_DYNAMICS_SCOPE = "bare_chassis_planar_plus_component_local_dynamics"
_BARE_CHASSIS_MASS_KG = 0.160


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "detach"):
        return _plain(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScannerRuntimeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ScannerRuntimeError(f"{context} must be finite")
    return result


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScannerRuntimeError(f"{context} must be an object")
    return value


def _vector3(value: Any, context: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ScannerRuntimeError(f"{context} must contain exactly three numbers")
    return tuple(  # type: ignore[return-value]
        _number(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _require_scanner_dynamics_completeness(
    assembly: ResolvedAssembly,
) -> DynamicsCompletenessRecord:
    """Validate the scanner's exact, hash-bound incomplete-dynamics contract."""

    record = assembly.dynamics_completeness
    if record.status != "incomplete" or record.modeled_scope != _SCANNER_DYNAMICS_SCOPE:
        raise ScannerRuntimeError(
            "scanner dynamics_completeness must declare status='incomplete' and "
            f"modeled_scope={_SCANNER_DYNAMICS_SCOPE!r}"
        )
    gate_ids = {gate.id for gate in record.gates}
    if gate_ids != _SCANNER_DYNAMICS_GATE_IDS or any(
        gate.status != "open" for gate in record.gates
    ):
        raise ScannerRuntimeError(
            "scanner dynamics_completeness must contain exactly these open gates: "
            + ", ".join(sorted(_SCANNER_DYNAMICS_GATE_IDS))
        )

    basis = _mapping(record.parameter_basis, "scanner dynamics parameter_basis")
    required_basis = {
        "component",
        "mass_parameter",
        "mass_scope",
        "yaw_inertia_parameter",
        "yaw_inertia_estimate",
    }
    if set(basis) != required_basis:
        raise ScannerRuntimeError(
            "scanner dynamics parameter_basis fields must be exactly "
            + ", ".join(sorted(required_basis))
        )
    if basis["component"] != "chassis":
        raise ScannerRuntimeError("scanner bare dynamics basis must reference component 'chassis'")
    components = [
        component
        for component in assembly.specification.components
        if component.id == basis["component"]
    ]
    chassis = _one(components, "scanner bare-chassis dynamics component")
    mass_parameter = str(basis["mass_parameter"])
    inertia_parameter = str(basis["yaw_inertia_parameter"])
    if mass_parameter != "mass" or inertia_parameter != "yaw_inertia":
        raise ScannerRuntimeError(
            "scanner bare dynamics basis must bind chassis mass and yaw_inertia parameters"
        )
    if basis["mass_scope"] != "published_bare_chassis_without_batteries":
        raise ScannerRuntimeError(
            "scanner chassis mass scope must be published_bare_chassis_without_batteries"
        )
    try:
        mass = _number(chassis.parameters[mass_parameter], "scanner bare chassis mass")
        yaw_inertia = _number(
            chassis.parameters[inertia_parameter], "scanner bare chassis yaw inertia"
        )
    except KeyError as exc:
        raise ScannerRuntimeError(
            f"scanner chassis is missing hash-bound parameter {exc.args[0]!r}"
        ) from exc
    if not math.isclose(mass, _BARE_CHASSIS_MASS_KG, rel_tol=0.0, abs_tol=1e-12):
        raise ScannerRuntimeError(
            f"scanner bare chassis mass must be {_BARE_CHASSIS_MASS_KG:.17g} kg, got {mass:.17g}"
        )

    estimate = _mapping(
        basis["yaw_inertia_estimate"], "scanner yaw_inertia_estimate"
    )
    required_estimate = {
        "kind",
        "package_body",
        "package_solid",
        "formula",
        "source",
    }
    if set(estimate) != required_estimate:
        raise ScannerRuntimeError(
            "scanner yaw_inertia_estimate fields must be exactly "
            + ", ".join(sorted(required_estimate))
        )
    if estimate["kind"] != "uniform_box_about_center_z":
        raise ScannerRuntimeError(
            "scanner yaw inertia must declare uniform_box_about_center_z derivation"
        )
    component_package = assembly.packages[chassis.package]
    try:
        body = component_package.body_map[str(estimate["package_body"])]
    except KeyError as exc:
        raise ScannerRuntimeError(
            f"scanner yaw inertia references missing package body {exc.args[0]!r}"
        ) from exc
    solids = [solid for solid in body.solids if solid.id == estimate["package_solid"]]
    solid = _one(solids, "scanner yaw inertia package solid")
    if solid.geometry.kind != "box":
        raise ScannerRuntimeError(
            "scanner yaw inertia uniform-box derivation requires canonical box geometry"
        )
    dimensions = tuple(float(value) for value in solid.geometry.dimensions_m)
    if estimate["formula"] != "mass * (length^2 + width^2) / 12":
        raise ScannerRuntimeError("scanner yaw inertia derivation formula is not canonical")
    if not isinstance(estimate["source"], str) or not estimate["source"].strip():
        raise ScannerRuntimeError("scanner yaw inertia derivation requires a source")
    if not math.isclose(
        yaw_inertia,
        mass * (dimensions[0] ** 2 + dimensions[1] ** 2) / 12.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ScannerRuntimeError(
            "scanner chassis yaw_inertia does not equal its hash-bound bare-box "
            "derivation from the referenced canonical solid: expected "
            f"{mass * (dimensions[0] ** 2 + dimensions[1] ** 2) / 12.0:.17g}, "
            f"got {yaw_inertia:.17g}"
        )
    return record


@dataclass(frozen=True, slots=True)
class ScannerMission:
    """Hash-bound scanner scenario read from the canonical contraption source."""

    object_center: tuple[float, float, float]
    object_side_m: float
    orbit_radius_m: float
    tangential_speed_m_s: float
    duration_s: float
    keep_out_radius_m: float

    def __post_init__(self) -> None:
        center = _vector3(self.object_center, "scanner mission object_center")
        object.__setattr__(self, "object_center", center)
        for name in (
            "object_side_m",
            "orbit_radius_m",
            "tangential_speed_m_s",
            "duration_s",
            "keep_out_radius_m",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), f"scanner mission {name}"))
        if self.object_side_m <= 0.0:
            raise ScannerRuntimeError("scanner mission object_side_m must be positive")
        if self.keep_out_radius_m < 0.0:
            raise ScannerRuntimeError("scanner mission keep_out_radius_m must be nonnegative")
        if self.orbit_radius_m <= self.keep_out_radius_m:
            raise ScannerRuntimeError(
                "scanner mission orbit_radius_m must be greater than keep_out_radius_m"
            )
        if self.tangential_speed_m_s < 0.0 or self.duration_s <= 0.0:
            raise ScannerRuntimeError(
                "scanner mission speed must be nonnegative and duration must be positive"
            )

    @classmethod
    def from_assembly(cls, assembly: ResolvedAssembly) -> "ScannerMission":
        if not isinstance(assembly, ResolvedAssembly):
            raise TypeError("scanner mission requires a ResolvedAssembly")
        environment = _mapping(assembly.specification.environment, "contraption.environment")
        cube = _mapping(
            environment.get("object_bounding_cube"),
            "contraption.environment.object_bounding_cube",
        )
        mission = _mapping(environment.get("mission"), "contraption.environment.mission")
        required_cube = {"center_m", "side_m"}
        required_mission = {
            "orbit_radius_m",
            "tangential_speed_m_s",
            "duration_s",
            "keep_out_radius_m",
        }
        missing_cube = sorted(required_cube - set(cube))
        missing_mission = sorted(required_mission - set(mission))
        if missing_cube or missing_mission:
            raise ScannerRuntimeError(
                "canonical scanner scenario is incomplete; "
                f"missing object fields={missing_cube}, missing mission fields={missing_mission}"
            )
        return cls(
            _vector3(cube["center_m"], "object_bounding_cube.center_m"),
            _number(cube["side_m"], "object_bounding_cube.side_m"),
            _number(mission["orbit_radius_m"], "mission.orbit_radius_m"),
            _number(mission["tangential_speed_m_s"], "mission.tangential_speed_m_s"),
            _number(mission["duration_s"], "mission.duration_s"),
            _number(mission["keep_out_radius_m"], "mission.keep_out_radius_m"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_center_m": list(self.object_center),
            "object_side_m": self.object_side_m,
            "orbit_radius_m": self.orbit_radius_m,
            "tangential_speed_m_s": self.tangential_speed_m_s,
            "duration_s": self.duration_s,
            "keep_out_radius_m": self.keep_out_radius_m,
        }


@dataclass(frozen=True, slots=True)
class _ScannerLayout:
    root_component: str
    chassis_position_x: str
    chassis_position_y: str
    chassis_yaw: str
    chassis_forward_speed: str
    left_wheel: str
    right_wheel: str
    left_wheel_rate: str
    right_wheel_rate: str
    left_wheel_coordinate: str
    right_wheel_coordinate: str
    lift_coordinate: str
    tilt_coordinate: str
    lift_feedback: str
    tilt_feedback: str
    camera_component: str
    camera_optical_connector: str
    camera_tilt_axis_connector: str
    output_bindings: Mapping[str, str]
    telemetry_outputs: tuple[str, ...]
    output_bounds: Mapping[str, tuple[float, float]]
    output_defaults: Mapping[str, float]
    output_slew_per_second: Mapping[str, float]

    def state_map(self) -> dict[str, str]:
        return {
            "position_x": self.chassis_position_x,
            "position_y": self.chassis_position_y,
            "yaw": self.chassis_yaw,
            "forward_speed": self.chassis_forward_speed,
            "left_wheel_rate": self.left_wheel_rate,
            "right_wheel_rate": self.right_wheel_rate,
            "left_wheel_coordinate": self.left_wheel_coordinate,
            "right_wheel_coordinate": self.right_wheel_coordinate,
            "lift_coordinate": self.lift_coordinate,
            "tilt_coordinate": self.tilt_coordinate,
            "lift_feedback": self.lift_feedback,
            "tilt_feedback": self.tilt_feedback,
        }


def _one(values: Sequence[Any], context: str) -> Any:
    if len(values) != 1:
        raise ScannerRuntimeError(
            f"{context} must resolve uniquely; found {len(values)} candidate(s)"
        )
    return values[0]


def _state_name(assembly: ResolvedAssembly, component: str, local_name: str) -> str:
    qualified = f"{component}.{local_name}"
    if qualified not in assembly.system.state_names:
        raise ScannerRuntimeError(
            f"resolved scanner requires PMDL unknown {qualified!r}, but it is absent"
        )
    return qualified


def _binding(assembly: ResolvedAssembly, component: str, connector: str) -> str | None:
    key = f"{component}.{connector}"
    try:
        return assembly.connector_bindings[key]
    except KeyError as exc:
        raise ScannerRuntimeError(
            f"scanner topology references connector {key!r} without a PMDL binding"
        ) from exc


def _endpoint_with_model_port(
    assembly: ResolvedAssembly, connection: Any, component: str, model_port: str
) -> bool:
    return any(
        endpoint.component == component
        and _binding(assembly, endpoint.component, endpoint.port) == model_port
        for endpoint in connection.endpoints
    )


def _other_component(connection: Any, component: str) -> str:
    others = [
        endpoint.component
        for endpoint in connection.endpoints
        if endpoint.component != component
    ]
    return _one(list(dict.fromkeys(others)), f"connection {connection.id!r} peer component")


def _revolute_for_component(assembly: ResolvedAssembly, component: str) -> Any:
    candidates = [
        attachment
        for attachment in assembly.physical.attachments
        if attachment.kind == "revolute"
        and component in {attachment.parent.component, attachment.child.component}
    ]
    return _one(candidates, f"revolute attachment for component {component!r}")


def _attachment_peer(attachment: Any, component: str) -> str:
    peers = [
        endpoint.component
        for endpoint in (attachment.parent, attachment.child)
        if endpoint.component != component
    ]
    return _one(
        list(dict.fromkeys(peers)),
        f"peer component on attachment {attachment.id!r}",
    )


def _position_feedback_for_attachment(
    assembly: ResolvedAssembly, attachment: Any, payload_component: str
) -> str:
    servo_components = [
        endpoint.component
        for endpoint in (attachment.parent, attachment.child)
        if endpoint.component != payload_component
    ]
    servo = _one(
        list(dict.fromkeys(servo_components)),
        f"servo peer for attachment {attachment.id!r}",
    )
    model = assembly.component_models[servo]
    feedback_ports = [
        port
        for port in model.signal_ports
        if port.name == "position_measurement" and port.direction == "output"
    ]
    feedback = _one(
        feedback_ports,
        f"position feedback PMDL output for servo {servo!r}",
    )
    nets = [
        connection
        for connection in assembly.specification.connections
        if connection.kind == "signal"
        and connection.domain == "signal"
        and _endpoint_with_model_port(
            assembly, connection, servo, feedback.name
        )
    ]
    _one(nets, f"declared position-feedback signal net for servo {servo!r}")
    return _state_name(assembly, servo, feedback.name)


def _derive_layout(assembly: ResolvedAssembly) -> _ScannerLayout:
    if not isinstance(assembly, ResolvedAssembly):
        raise TypeError("scanner runtime requires a ResolvedAssembly")
    _require_scanner_dynamics_completeness(assembly)
    root = assembly.physical.root_component
    chassis_model = assembly.component_models[root]
    required_chassis_states = {"position_x", "position_y", "yaw", "forward_speed"}
    actual_chassis_states = {state.name for state in chassis_model.states}
    missing = sorted(required_chassis_states - actual_chassis_states)
    if missing:
        raise ScannerRuntimeError(
            f"physical root {root!r} does not expose scanner chassis states {missing}"
        )

    left_net = _one(
        [
            connection
            for connection in assembly.specification.connections
            if connection.kind == "power"
            and connection.domain == "mechanical"
            and _endpoint_with_model_port(assembly, connection, root, "left_drive")
        ],
        "mechanical net bound to chassis left_drive",
    )
    right_net = _one(
        [
            connection
            for connection in assembly.specification.connections
            if connection.kind == "power"
            and connection.domain == "mechanical"
            and _endpoint_with_model_port(assembly, connection, root, "right_drive")
        ],
        "mechanical net bound to chassis right_drive",
    )
    left_wheel = _other_component(left_net, root)
    right_wheel = _other_component(right_net, root)
    if left_wheel == right_wheel:
        raise ScannerRuntimeError("left and right chassis contacts resolve to the same wheel")

    def wheel_details(wheel: str) -> tuple[str, str]:
        attachment = _revolute_for_component(assembly, wheel)
        if attachment.coordinate is None:
            raise ScannerRuntimeError(f"wheel {wheel!r} attachment has no coordinate")
        if attachment.coordinate not in assembly.system.state_names:
            raise ScannerRuntimeError(
                f"wheel {wheel!r} coordinate {attachment.coordinate!r} is not a PMDL unknown"
            )
        wheel_package = assembly.packages[
            next(item.package for item in assembly.specification.components if item.id == wheel)
        ]
        axle_connectors = [
            connector for connector in wheel_package.connectors if connector.model_port == "axle"
        ]
        axle = _one(axle_connectors, f"wheel {wheel!r} axle connector")
        wheel_model = assembly.component_models[wheel]
        axle_ports = [port for port in wheel_model.power_ports if port.name == axle.model_port]
        axle_port = _one(axle_ports, f"wheel {wheel!r} axle PMDL port")
        return _state_name(assembly, wheel, axle_port.flow), attachment.coordinate

    left_rate, left_coordinate = wheel_details(left_wheel)
    right_rate, right_coordinate = wheel_details(right_wheel)

    if assembly.controller is None:
        raise ScannerRuntimeError("resolved scanner requires a verified canonical controller")
    output_bindings = dict(assembly.controller_output_bindings)
    telemetry_outputs = tuple(assembly.controller_telemetry_outputs)
    expected_actuator_outputs = {
        "left_voltage",
        "right_voltage",
        "lift_target",
        "tilt_target",
    }
    if set(output_bindings) != expected_actuator_outputs:
        raise ScannerRuntimeError(
            "scanner controller actuator outputs must be exactly "
            f"{sorted(expected_actuator_outputs)}; found={sorted(output_bindings)}"
        )
    if telemetry_outputs != ("record_video",):
        raise ScannerRuntimeError(
            "scanner controller telemetry output must be exactly ['record_video']; "
            f"found={list(telemetry_outputs)}"
        )
    if set(output_bindings.values()) != set(assembly.system.control_names):
        raise ScannerRuntimeError(
            "controller output bindings must drive exactly the assembled controls; "
            f"bound={sorted(output_bindings.values())}, "
            f"assembled={sorted(assembly.system.control_names)}"
        )

    def controlled_component(output_name: str) -> str:
        source = output_bindings[output_name]
        binding = _one(
            [
                control
                for control in assembly.specification.controls
                if control.source == source
            ],
            f"assembled control binding for controller output {output_name!r}",
        )
        model_port = _binding(
            assembly, binding.target.component, binding.target.port
        )
        if model_port != "position_command":
            raise ScannerRuntimeError(
                f"controller output {output_name!r} targets "
                f"{binding.target.component}.{binding.target.port}, bound to "
                f"PMDL port {model_port!r}, not 'position_command'"
            )
        return binding.target.component

    lift_servo = controlled_component("lift_target")
    tilt_servo = controlled_component("tilt_target")
    if lift_servo == tilt_servo:
        raise ScannerRuntimeError("lift and tilt outputs target the same servo component")
    arm_attachment = _revolute_for_component(assembly, lift_servo)
    camera_attachment = _revolute_for_component(assembly, tilt_servo)
    arm = _attachment_peer(arm_attachment, lift_servo)
    camera = _attachment_peer(camera_attachment, tilt_servo)
    if arm_attachment.coordinate is None or camera_attachment.coordinate is None:
        raise ScannerRuntimeError("scanner arm/camera revolute joints require coordinates")
    for coordinate in (arm_attachment.coordinate, camera_attachment.coordinate):
        if coordinate not in assembly.system.state_names:
            raise ScannerRuntimeError(
                f"physical joint coordinate {coordinate!r} is absent from assembled PMDL states"
            )
    lift_feedback = _position_feedback_for_attachment(
        assembly, arm_attachment, arm
    )
    tilt_feedback = _position_feedback_for_attachment(
        assembly, camera_attachment, camera
    )
    output_bounds: dict[str, tuple[float, float]] = {}
    output_defaults: dict[str, float] = {}
    output_slew: dict[str, float] = {}
    for output_name, source in output_bindings.items():
        try:
            lower_raw, upper_raw = assembly.system.control_bounds[source]
            default_raw = assembly.system.control_defaults[source]
            slew_raw = assembly.system.control_slew_rates[source]
        except KeyError as exc:
            raise ScannerRuntimeError(
                f"assembled control {source!r} must expose default, bounds, and slew rate"
            ) from exc
        if lower_raw is None or upper_raw is None:
            raise ScannerRuntimeError(
                f"assembled control {source!r} requires finite lower and upper bounds"
            )
        lower = _number(lower_raw, f"assembled control {source!r} minimum")
        upper = _number(upper_raw, f"assembled control {source!r} maximum")
        default = _number(default_raw, f"assembled control {source!r} default")
        slew = _number(slew_raw, f"assembled control {source!r} slew_per_second")
        if lower >= upper or not lower <= default <= upper or slew <= 0.0:
            raise ScannerRuntimeError(
                f"assembled control {source!r} has invalid bounds/default/slew"
            )
        output_bounds[output_name] = (lower, upper)
        output_defaults[output_name] = default
        output_slew[output_name] = slew

    # These controller parameters appear inside the canonical ControlProgram's
    # expressions.  They are intentionally checked against the hash-bound
    # assembled control contract so stale internal clamps cannot disagree with
    # the actual actuator admission limits.
    controller_limits = {
        "voltage_limit": output_bounds["left_voltage"][1],
        "lift_min": output_bounds["lift_target"][0],
        "lift_max": output_bounds["lift_target"][1],
        "tilt_min": output_bounds["tilt_target"][0],
        "tilt_max": output_bounds["tilt_target"][1],
        "lift_midpoint": output_defaults["lift_target"],
    }
    for parameter_name, expected in controller_limits.items():
        try:
            actual = _number(
                assembly.controller.parameters[parameter_name],
                f"controller parameter {parameter_name!r}",
            )
        except KeyError as exc:
            raise ScannerRuntimeError(
                f"canonical scanner controller lacks parameter {parameter_name!r}"
            ) from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ScannerRuntimeError(
                f"controller parameter {parameter_name!r}={actual} disagrees with "
                f"assembled control contract value {expected}"
            )
    right_voltage_bounds = output_bounds["right_voltage"]
    voltage_limit = controller_limits["voltage_limit"]
    if (
        output_bounds["left_voltage"] != (-voltage_limit, voltage_limit)
        or right_voltage_bounds != (-voltage_limit, voltage_limit)
    ):
        raise ScannerRuntimeError(
            "canonical voltage_limit requires identical symmetric left/right control bounds"
        )

    camera_package = assembly.packages[
        next(item.package for item in assembly.specification.components if item.id == camera)
    ]
    optical_connector = _one(
        [
            connector
            for connector in camera_package.connectors
            if connector.interface == "camera-view-axis"
            and connector.domain == "optical"
            and connector.body is not None
            and connector.local_pose is not None
        ],
        f"camera {camera!r} spatial optical-axis connector",
    )
    camera_optical_connector = f"{camera}.{optical_connector.id}"
    # Revolute coordinates are positive rotations about the parent connector's
    # local +Z axis in the physical resolver.  Retain that canonical connector
    # identity rather than copying a camera-axis or mounting-angle constant.
    camera_tilt_axis_connector = (
        f"{camera_attachment.parent.component}.{camera_attachment.parent.connector}"
    )

    return _ScannerLayout(
        root,
        _state_name(assembly, root, "position_x"),
        _state_name(assembly, root, "position_y"),
        _state_name(assembly, root, "yaw"),
        _state_name(assembly, root, "forward_speed"),
        left_wheel,
        right_wheel,
        left_rate,
        right_rate,
        left_coordinate,
        right_coordinate,
        arm_attachment.coordinate,
        camera_attachment.coordinate,
        lift_feedback,
        tilt_feedback,
        camera,
        camera_optical_connector,
        camera_tilt_axis_connector,
        output_bindings,
        telemetry_outputs,
        output_bounds,
        output_defaults,
        output_slew,
    )


class ScannerAssemblyController:
    """Adapter from resolved sensor state to the verified data-only controller."""

    _EXTERNAL_INPUTS = frozenset(
        {"armed", "reset", "emergency_stop", "target_speed", "orbit_radius"}
    )
    _SENSOR_INPUTS = frozenset(
        {
            "measured_speed",
            "radius_error",
            "heading_error",
            "orbit_phase",
            "required_tilt",
            "lift_feedback",
            "tilt_feedback",
            "drive_fault",
            "brownout",
        }
    )

    def __init__(
        self,
        assembly: ResolvedAssembly,
        mission: ScannerMission | None = None,
        *,
        external_inputs: Mapping[str, Any] | None = None,
    ) -> None:
        self.assembly_sha256 = assembly.assembly_sha256
        self.mission = mission or ScannerMission.from_assembly(assembly)
        canonical_mission = ScannerMission.from_assembly(assembly)
        if self.mission != canonical_mission:
            raise ScannerRuntimeError(
                "scanner mission overrides change the hash-bound scenario; update and "
                "re-resolve the contraption specification instead"
            )
        if assembly.controller is None:
            raise ScannerRuntimeError("resolved scanner has no verified ControlProgram")
        self._assembly = assembly
        self._state_names = tuple(assembly.system.state_names)
        self.program: ControlProgram = assembly.controller
        self.layout = _derive_layout(assembly)
        declared_external = {
            signal.name for signal in self.program.inputs if signal.source == "external"
        }
        declared_sensors = {
            signal.name for signal in self.program.inputs if signal.source == "sensor"
        }
        if declared_external != self._EXTERNAL_INPUTS or declared_sensors != self._SENSOR_INPUTS:
            raise ScannerRuntimeError(
                "scanner controller input contract mismatch; "
                f"external={sorted(declared_external)}, sensors={sorted(declared_sensors)}"
            )

        def expression_inputs(expression: Any) -> set[str]:
            if expression.op == "ref" and isinstance(expression.value, str):
                prefix, separator, name = expression.value.partition(".")
                return {name} if separator and prefix in {"external", "sensor"} else set()
            result: set[str] = set()
            for argument in expression.args:
                result.update(expression_inputs(argument))
            return result

        state_machine_inputs: set[str] = set()
        for state in self.program.states:
            for transition in state.transitions:
                state_machine_inputs.update(expression_inputs(transition.condition))
            for update in state.updates.values():
                state_machine_inputs.update(expression_inputs(update))
        shared_state_machine_inputs = {
            "armed",
            "reset",
            "emergency_stop",
            "drive_fault",
            "brownout",
        }
        unsupported_state_inputs = sorted(
            state_machine_inputs - shared_state_machine_inputs
        )
        if unsupported_state_inputs:
            raise ScannerRuntimeError(
                "one scanner controller runtime cannot select different states for "
                "Monte Carlo samples; transition/update inputs must be shared, but "
                f"found sample-varying input(s) {unsupported_state_inputs}"
            )
        input_specs = {signal.name: signal for signal in self.program.inputs}
        if input_specs["target_speed"].default != self.mission.tangential_speed_m_s:
            raise ScannerRuntimeError(
                "controller target_speed default disagrees with canonical mission speed"
            )
        if input_specs["orbit_radius"].default != self.mission.orbit_radius_m:
            raise ScannerRuntimeError(
                "controller orbit_radius default disagrees with canonical mission radius"
            )
        supplied = {} if external_inputs is None else dict(external_inputs)
        unknown = sorted(set(supplied) - declared_external)
        if unknown:
            raise ScannerRuntimeError(
                f"unknown scanner external input(s): {unknown}; "
                f"declared={sorted(declared_external)}"
            )
        defaults: dict[str, Any] = {
            "armed": True,
            "reset": False,
            "emergency_stop": False,
            "target_speed": self.mission.tangential_speed_m_s,
            "orbit_radius": self.mission.orbit_radius_m,
        }
        defaults.update(supplied)
        self.external_inputs = {
            name: input_specs[name].normalize(value, f"external input {name!r}")
            for name, value in defaults.items()
        }
        self._indices = {
            name: assembly.system.state_names.index(state_name)
            for name, state_name in self.layout.state_map().items()
        }
        self._runtime = ControllerRuntime(self.program)
        self._last_time_s: float | None = None
        self._applied_outputs: dict[str, Any] = {}
        self._trace: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._runtime.reset()
        self._last_time_s = None
        self._applied_outputs = {}
        self._trace = []

    @staticmethod
    def _wrapped(value: Any, backend: Backend) -> Any:
        return backend.remainder(value + math.pi, 2.0 * math.pi) - math.pi

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, (bool, int, float)):
            return value
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if np.asarray(array).size == 0:
            raise ScannerRuntimeError("controller input batch may not be empty")
        item = np.asarray(array).reshape(-1)[0]
        return bool(item) if np.asarray(array).dtype.kind == "b" else float(item)

    def _validated_runtime_inputs(
        self, context: Mapping[str, Any], count: int
    ) -> dict[str, Any]:
        """Validate every sample before selecting shared state-machine inputs."""

        validated: dict[str, Any] = {}
        for signal in self.program.inputs:
            value = context[signal.name]
            if hasattr(value, "detach"):
                array = np.asarray(value.detach().cpu().numpy())
            else:
                array = np.asarray(value)
            if array.ndim == 0:
                items = [array.item()] * count
            elif array.size == count:
                items = list(array.reshape(-1))
            else:
                raise ScannerRuntimeError(
                    f"controller input {signal.name!r} has {array.size} values for "
                    f"a {count}-sample simulation batch"
                )
            normalized = [
                signal.normalize(item.item() if hasattr(item, "item") else item,
                                 f"controller input {signal.name!r} sample {index}")
                for index, item in enumerate(items)
            ]
            # Scanner transitions use only shared external booleans and the
            # currently ideal shared fault flags.  Actuator expressions remain
            # vectorized; the runtime needs one scalar solely for transitions.
            validated[signal.name] = normalized[0]
        return validated

    def _required_tilt_from_physical_configuration(
        self, state: Any, backend: Backend
    ) -> Any:
        """Return the closest look-at joint coordinate from canonical poses.

        The physical resolver owns every mounting translation, quaternion, and
        revolute coordinate.  For each sample we reconstruct and validate that
        exact assembly, then project the target ray and declared optical +Z
        axis onto the plane normal to the resolved tilt-joint +Z axis.  The
        signed rotation between those projections is the required correction.
        No scanner linkage length, camera height, or mounting angle is copied
        into this controller adapter.
        """

        if hasattr(state, "detach"):
            rows = np.asarray(state.detach().cpu().numpy(), dtype=np.float64)
        else:
            rows = np.asarray(state, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self._state_names):
            raise ScannerRuntimeError(
                "scanner controller state must be [sample, resolved PMDL state]"
            )
        required: list[float] = []
        for sample_index, row in enumerate(rows):
            configured = self._assembly.configuration_from_state(
                row, state_names=self._state_names
            )
            try:
                optical_pose = configured.connector_poses[
                    self.layout.camera_optical_connector
                ]
                joint_pose = configured.connector_poses[
                    self.layout.camera_tilt_axis_connector
                ]
            except KeyError as exc:
                raise ScannerRuntimeError(
                    "resolved physical configuration omitted the scanner camera "
                    f"connector {exc.args[0]!r}"
                ) from exc
            optical = _rotate_z(optical_pose.rotation_quaternion_wxyz)
            joint_axis = _rotate_z(joint_pose.rotation_quaternion_wxyz)
            target = np.asarray(self.mission.object_center, dtype=np.float64) - np.asarray(
                optical_pose.translation_m, dtype=np.float64
            )
            target_norm = float(np.linalg.norm(target))
            axis_norm = float(np.linalg.norm(joint_axis))
            if target_norm <= 1e-12 or axis_norm <= 1e-12:
                raise ScannerRuntimeError(
                    "camera look-at geometry is singular at "
                    f"sample={sample_index}: target_norm={target_norm}, "
                    f"joint_axis_norm={axis_norm}"
                )
            target /= target_norm
            joint_axis /= axis_norm
            optical_projection = optical - joint_axis * float(np.dot(joint_axis, optical))
            target_projection = target - joint_axis * float(np.dot(joint_axis, target))
            optical_norm = float(np.linalg.norm(optical_projection))
            target_projection_norm = float(np.linalg.norm(target_projection))
            if optical_norm <= 1e-12 or target_projection_norm <= 1e-12:
                raise ScannerRuntimeError(
                    "camera look-at target cannot be resolved about the declared tilt "
                    f"axis at sample={sample_index}"
                )
            optical_projection /= optical_norm
            target_projection /= target_projection_norm
            correction = math.atan2(
                float(
                    np.dot(
                        joint_axis,
                        np.cross(optical_projection, target_projection),
                    )
                ),
                float(np.dot(optical_projection, target_projection)),
            )
            current_tilt = float(row[self._indices["tilt_coordinate"]])
            required.append(current_tilt + correction)
        return backend.asarray(required)

    def _context(self, t: float, state: Any, backend: Backend) -> dict[str, Any]:
        x = state[:, self._indices["position_x"]]
        y = state[:, self._indices["position_y"]]
        yaw = state[:, self._indices["yaw"]]
        lift_feedback = state[:, self._indices["lift_feedback"]]
        tilt_feedback = state[:, self._indices["tilt_feedback"]]
        dx = x - self.mission.object_center[0]
        dy = y - self.mission.object_center[1]
        radius = backend.sqrt(dx * dx + dy * dy + 1e-18)
        phase = backend.atan2(dy, dx)
        heading_error = self._wrapped(phase + math.pi / 2.0 - yaw, backend)
        # The chassis PMDL explicitly owns forward_speed.  Reading it avoids a
        # second wheel-radius/rolling conversion in mission code.
        measured_speed = state[:, self._indices["forward_speed"]]
        required_tilt = self._required_tilt_from_physical_configuration(state, backend)
        return {
            **self.external_inputs,
            "measured_speed": measured_speed,
            "radius_error": radius - float(self.external_inputs["orbit_radius"]),
            "heading_error": heading_error,
            "orbit_phase": phase,
            "required_tilt": required_tilt,
            "lift_feedback": lift_feedback,
            "tilt_feedback": tilt_feedback,
            "drive_fault": False,
            "brownout": False,
            "time": t,
            "time_in_state": self._runtime.time_in_state,
            "dt": 0.0 if self._last_time_s is None else t - self._last_time_s,
            **{
                f"register.{name}": value
                for name, value in self._runtime.registers.items()
            },
        }

    def _slew(
        self,
        outputs: Mapping[str, Any],
        elapsed: float,
        count: int,
        backend: Backend,
        *,
        bypass: bool,
    ) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        for output_name in self.layout.output_bindings:
            value = backend.asarray(outputs[output_name])
            if len(value.shape) == 0:
                value = backend.broadcast_to(value, (count,))
            lower, upper = self.layout.output_bounds[output_name]
            value = backend.clip(value, lower, upper)
            previous = self._applied_outputs.get(output_name)
            if previous is None:
                previous = backend.full(
                    (count,), self.layout.output_defaults[output_name]
                )
            if not bypass:
                maximum_change = self.layout.output_slew_per_second[output_name] * elapsed
                value = backend.minimum(
                    backend.maximum(value, previous - maximum_change),
                    previous + maximum_change,
                )
            applied[output_name] = value
        self._applied_outputs = applied
        return applied

    def evaluate(self, t: Any, state: Any, backend: Backend) -> Mapping[str, Any]:
        time_s = _number(t, "scanner controller time")
        if self._last_time_s is not None and time_s < self._last_time_s - 1e-12:
            raise ScannerRuntimeError("scanner controller time moved backwards")
        if self._last_time_s is not None and abs(time_s - self._last_time_s) <= 1e-12:
            return {
                self.layout.output_bindings[name]: value
                for name, value in self._applied_outputs.items()
            }

        count = int(state.shape[0])
        context = self._context(time_s, state, backend)
        elapsed = 0.0 if self._last_time_s is None else time_s - self._last_time_s
        active_state = self._runtime.state
        outputs = evaluate_state_outputs(self.program, active_state, context, backend)
        scalar_inputs = self._validated_runtime_inputs(context, count)
        self._runtime.step(scalar_inputs, max(elapsed, 1e-12))
        emergency_override = bool(
            self.external_inputs["emergency_stop"]
            or context["drive_fault"]
            or context["brownout"]
        )
        if emergency_override:
            # The state transition is canonical ControlProgram behavior; only
            # the immediate timing override bypasses ordinary actuator slew.
            outputs = evaluate_state_outputs(
                self.program, self._runtime.state, context, backend
            )
        applied = self._slew(
            outputs, elapsed, count, backend, bypass=emergency_override
        )
        telemetry = {
            name: self._scalar(outputs[name])
            for name in self.layout.telemetry_outputs
        }
        self._trace.append(
            {
                "time_s": time_s,
                "active_state": active_state,
                "next_state": self._runtime.state,
                "telemetry": telemetry,
                "applied_controls": {
                    self.layout.output_bindings[name]: _plain(value)
                    for name, value in applied.items()
                },
                "emergency_slew_override": emergency_override,
            }
        )
        self._last_time_s = time_s
        return {
            self.layout.output_bindings[name]: value
            for name, value in applied.items()
        }

    def trace(self) -> list[dict[str, Any]]:
        return _plain(self._trace)


def load_scanner_assembly(root: str | Path | None = None) -> ResolvedAssembly:
    """Load and resolve the bundled scanner example as one verified closure."""

    project = asset_root() if root is None else Path(root).expanduser().resolve()
    example = project / "examples" / "scanner_robot"
    specification = json.loads((example / "contraption.json").read_text(encoding="utf-8"))
    packages = ComponentPackageRegistry.load(example / "component_packages.json")
    models = {}
    for model_path in sorted((project / "models" / "scanner").glob("*.pmdl")):
        model = load_model(model_path)
        if model.id in models:
            raise ScannerRuntimeError(f"duplicate scanner PMDL model id {model.id!r}")
        models[model.id] = model
    if not models:
        raise ScannerRuntimeError(
            f"no scanner PMDL models found under {project / 'models' / 'scanner'}"
        )
    program_paths = sorted((example / "controls").glob("*.json"))
    programs: dict[str, ControlProgram] = {}
    for program_path in program_paths:
        program = load_control_program(program_path)
        if program.name in programs:
            raise ScannerRuntimeError(
                f"duplicate scanner control-program id {program.name!r}"
            )
        programs[program.name] = program
    if not programs:
        raise ScannerRuntimeError(f"no scanner control program found under {example / 'controls'}")
    assembly = resolve_assembly(
        specification,
        packages,
        models,
        control_programs=programs,
    )
    _require_scanner_dynamics_completeness(assembly)
    return assembly


@dataclass(frozen=True, slots=True)
class _ScannerSimulationRuntime:
    """Minimal adapter that makes canonical validation part of simulation.

    It deliberately carries no physical equations or geometry.  The generic
    simulator discovers ``system`` and ``controller`` structurally, and calls
    this post-result hook before returning any trajectory.
    """

    assembly: ResolvedAssembly
    controller: ScannerAssemblyController

    @property
    def system(self) -> Any:
        return self.assembly.system

    def validate_simulation_step(
        self,
        *,
        step_index: int,
        time_s: float,
        state: Any,
        state_names: Sequence[str],
        backend: Backend,
    ) -> None:
        del backend  # The canonical validator owns safe NumPy/Torch conversion.
        try:
            self.assembly.validate_simulation_state(
                state,
                state_names=state_names,
                step_index=step_index,
                time_s=time_s,
                require_initial_configuration=step_index == 0,
            )
        except ValueError as exc:
            raise ScannerRuntimeError(
                f"accepted scanner state rejected: {exc}"
            ) from exc

    def validate_simulation_result(self, result: SimulationResult) -> None:
        self.assembly.validate_simulation_result(result)


def simulate_scanner_robot(
    assembly: ResolvedAssembly,
    mission: ScannerMission | None = None,
    *,
    external_inputs: Mapping[str, Any] | None = None,
    visualization_sample_index: int = 0,
    duration: float | None = None,
    dt: float = 0.05,
    num_samples: int = 1,
    seed: int = 0,
    backend: str | Backend = "numpy",
    device: str | None = None,
    dtype: Any | None = None,
    use_model_uncertainty: bool = False,
    process_noise: bool = False,
    **simulator_options: Any,
) -> SimulationResult:
    """Simulate the component PMDL assembly and emit validated physical frames."""

    if not isinstance(assembly, ResolvedAssembly):
        raise TypeError(
            "simulate_scanner_robot requires a ResolvedAssembly; resolve the canonical "
            "contraption/packages/PMDL closure first"
        )
    _require_scanner_dynamics_completeness(assembly)
    canonical_mission = ScannerMission.from_assembly(assembly)
    if mission is not None and mission != canonical_mission:
        raise ScannerRuntimeError(
            "scanner mission overrides would not be covered by assembly_sha256; "
            "update the canonical contraption and resolve it again"
        )
    mission = canonical_mission
    controller = ScannerAssemblyController(
        assembly, mission, external_inputs=external_inputs
    )
    horizon = mission.duration_s if duration is None else _number(duration, "duration")
    if horizon <= 0.0:
        raise ScannerRuntimeError("duration must be positive")
    if (
        isinstance(visualization_sample_index, bool)
        or not isinstance(visualization_sample_index, int)
        or visualization_sample_index < 0
        or visualization_sample_index >= num_samples
    ):
        raise ScannerRuntimeError(
            "visualization_sample_index must identify one actual simulation "
            f"sample in [0, {num_samples})"
        )
    runtime = _ScannerSimulationRuntime(assembly, controller)
    result = simulate(
        runtime,
        duration=horizon,
        dt=dt,
        num_samples=num_samples,
        seed=seed,
        backend=backend,
        device=device,
        dtype=dtype,
        use_model_uncertainty=use_model_uncertainty,
        process_noise=process_noise,
        **simulator_options,
    )
    layout = controller.layout
    return replace(
        result,
        metadata={
            **dict(result.metadata),
            "contraption_id": assembly.specification.id,
            "assembly_sha256": assembly.assembly_sha256,
            "pmdl_sha256": assembly.system.pmdl_sha256,
            "simulation_scope": "component_pmdl_resolved_assembly",
            "pmdl_network_composed": True,
            "pose_frame_statistic": "actual_sample",
            "pose_frame_sample_index": visualization_sample_index,
            "scanner_external_inputs": dict(controller.external_inputs),
            "scanner_controller": {
                "name": controller.program.name,
                "version": controller.program.version,
                "output_bindings": dict(layout.output_bindings),
                "telemetry_outputs": list(layout.telemetry_outputs),
            },
            "scanner_control_frames": controller.trace(),
            "scanner_emergency_override": {
                "behavior": "canonical_emergency_state_outputs_apply_immediately",
                "drive_behavior": "zero_left_and_right_voltage_immediately",
                "joint_behavior": "hold_current_lift_and_tilt_feedback",
                "bypasses_slew_limit": True,
                "reason": "emergency stop has safety priority over normal actuator slew",
            },
            "scanner_sensor_assumptions": {
                "measured_speed": {
                    "source": layout.chassis_forward_speed,
                    "fidelity": "ideal_simulated_state_estimator_proxy",
                    "reason": "the current assembly has no localization/IMU estimator component",
                },
                "planar_localization": {
                    "sources": [
                        layout.chassis_position_x,
                        layout.chassis_position_y,
                        layout.chassis_yaw,
                    ],
                    "fidelity": "ideal_simulated_state_estimator_proxy",
                    "reason": "the current assembly has no localization/IMU estimator component",
                },
                "lift_feedback": {
                    "source": layout.lift_feedback,
                    "fidelity": "declared_servo_position_measurement_signal_net",
                },
                "tilt_feedback": {
                    "source": layout.tilt_feedback,
                    "fidelity": "declared_servo_position_measurement_signal_net",
                },
                "required_tilt": {
                    "source": [
                        layout.camera_optical_connector,
                        layout.camera_tilt_axis_connector,
                    ],
                    "fidelity": "exact_resolved_connector_geometry_with_known_target",
                },
                "drive_fault": {
                    "value": False,
                    "reason": "driver fault generation is outside current component PMDL fidelity",
                },
                "brownout": {
                    "value": False,
                    "reason": "brownout thresholding is outside current component PMDL fidelity",
                },
            },
        },
    )


def scanner_physical_scene(
    assembly: ResolvedAssembly,
    result: SimulationResult,
    *,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Reconstruct a viewer scene from the assembly and exact result samples."""

    if result.metadata.get("assembly_sha256") != assembly.assembly_sha256:
        raise ScannerRuntimeError("simulation result was produced from a different assembly")
    if sample_index is None:
        sample_index = result.metadata.get("pose_frame_sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ScannerRuntimeError(
            "pose_frame_sample_index must be an integer identifying an actual sample"
        )
    # The resolver admits the complete ensemble, then reconstructs one exact
    # selected path.  No serialized scene or duplicate pose metadata is trusted.
    frames = assembly.body_pose_frames(result, sample_index=sample_index)
    scene = _plain(assembly.scene)
    scene["body_pose_frames"] = _plain(frames)
    return scene


def _wilson_upper(successes: int, count: int) -> float:
    if count < 1:
        raise ScannerRuntimeError("scanner metrics require at least one sample")
    estimate = successes / count
    z = 1.6448536269514722
    z2 = z * z
    denominator = 1.0 + z2 / count
    center = (estimate + z2 / (2.0 * count)) / denominator
    half = (
        z
        / denominator
        * math.sqrt(estimate * (1.0 - estimate) / count + z2 / (4.0 * count * count))
    )
    return min(1.0, center + half)


def _rotate_z(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(item) for item in quaternion)
    return np.asarray(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dtype=np.float64,
    )


def scanner_metrics(
    assembly: ResolvedAssembly,
    result: SimulationResult,
    *,
    warmup_s: float = 0.0,
) -> dict[str, Any]:
    """Compute ensemble evidence from canonical physical configurations.

    Viewer frames contain one explicitly selected actual sample.  They are not
    an ensemble-statistics source.  Metrics therefore require the exact
    :class:`ResolvedAssembly` and resolve every sample/time state through its
    physical closure before calculating pointing or keep-out clearance.
    """

    if not isinstance(assembly, ResolvedAssembly):
        raise TypeError("scanner_metrics requires a ResolvedAssembly")
    dynamics_completeness = _require_scanner_dynamics_completeness(assembly)
    metadata = _mapping(result.metadata, "simulation metadata")
    if metadata.get("assembly_sha256") != assembly.assembly_sha256:
        raise ScannerRuntimeError("metrics assembly differs from the simulated assembly")
    if metadata.get("pmdl_sha256") != assembly.system.pmdl_sha256:
        raise ScannerRuntimeError("metrics refuse a result with a stale PMDL hash")
    mission = ScannerMission.from_assembly(assembly)
    external_inputs = _mapping(
        metadata.get("scanner_external_inputs"), "scanner_external_inputs"
    )
    target_radius = _number(
        external_inputs.get("orbit_radius"), "scanner external orbit_radius"
    )
    warmup = _number(warmup_s, "warmup_s")
    if warmup < 0.0:
        raise ScannerRuntimeError("warmup_s must be nonnegative")

    numerical = infer_backend(result.samples)
    samples = np.asarray(numerical.to_numpy(result.samples), dtype=np.float64)
    times = np.asarray(numerical.to_numpy(result.time), dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != len(assembly.system.state_names):
        raise ScannerRuntimeError(
            "scanner metric samples do not match the resolved PMDL state inventory"
        )
    if tuple(result.state_names) != tuple(assembly.system.state_names):
        raise ScannerRuntimeError("scanner metric state_names differ from the assembly")
    start = min(int(np.searchsorted(times, warmup, side="left")), len(times) - 1)
    sample_count, time_count, _ = samples.shape
    root_positions = np.empty((sample_count, time_count, 3), dtype=np.float64)
    pointing = np.empty((sample_count, time_count), dtype=np.float64)
    camera_heights = np.empty((sample_count, time_count), dtype=np.float64)
    camera_connector = _derive_layout(assembly).camera_optical_connector
    target_center = np.asarray(mission.object_center, dtype=np.float64)
    for sample_index in range(sample_count):
        for time_index in range(time_count):
            configured = assembly.configuration_from_state(
                samples[sample_index, time_index],
                state_names=result.state_names,
            )
            root_pose = configured.component_pose(assembly.physical.root_component)
            root_positions[sample_index, time_index] = root_pose.translation_m
            try:
                optical_pose = configured.connector_poses[camera_connector]
            except KeyError as exc:
                raise ScannerRuntimeError(
                    f"physical configuration lacks optical connector {camera_connector!r}"
                ) from exc
            position = np.asarray(optical_pose.translation_m, dtype=np.float64)
            optical = _rotate_z(optical_pose.rotation_quaternion_wxyz)
            target = target_center - position
            target_norm = float(np.linalg.norm(target))
            if target_norm <= 1e-12:
                raise ScannerRuntimeError(
                    "camera coincides with the scanner target center at "
                    f"sample={sample_index}, time_index={time_index}"
                )
            cosine = float(np.dot(optical, target) / target_norm)
            pointing[sample_index, time_index] = math.degrees(
                math.acos(max(-1.0, min(1.0, cosine)))
            )
            camera_heights[sample_index, time_index] = position[2]

    x = root_positions[:, start:, 0]
    y = root_positions[:, start:, 1]
    radius = np.hypot(x - mission.object_center[0], y - mission.object_center[1])
    error = radius - target_radius
    root_keepout_violation_by_sample = np.any(
        radius <= mission.keep_out_radius_m, axis=1
    )
    root_keepout_violation_count = int(
        np.count_nonzero(root_keepout_violation_by_sample)
    )
    phases = np.unwrap(
        np.arctan2(y - mission.object_center[1], x - mission.object_center[0]), axis=1
    )
    evaluated_pointing = pointing[:, start:]
    evaluated_heights = camera_heights[:, start:]
    root_keepout_violation_probability = root_keepout_violation_count / sample_count
    acceptance = _mapping(
        assembly.specification.metadata.get("acceptance", {}),
        "scanner acceptance",
    )
    required_acceptance = {
        "orbit_radius_rmse_max_m",
        "camera_pointing_p95_max_deg",
        "root_keepout_violation_probability_max",
    }
    missing_acceptance = sorted(required_acceptance - set(acceptance))
    if missing_acceptance:
        raise ScannerRuntimeError(
            "hash-bound scanner acceptance contract is missing field(s): "
            + ", ".join(missing_acceptance)
        )
    orbit_limit = _number(
        acceptance["orbit_radius_rmse_max_m"],
        "scanner acceptance orbit_radius_rmse_max_m",
    )
    pointing_limit = _number(
        acceptance["camera_pointing_p95_max_deg"],
        "scanner acceptance camera_pointing_p95_max_deg",
    )
    root_keepout_violation_limit = _number(
        acceptance["root_keepout_violation_probability_max"],
        "scanner acceptance root_keepout_violation_probability_max",
    )
    if orbit_limit < 0.0 or not 0.0 <= pointing_limit <= 180.0:
        raise ScannerRuntimeError(
            "scanner orbit/pointing acceptance limits are outside physical ranges"
        )
    if not 0.0 <= root_keepout_violation_limit <= 1.0:
        raise ScannerRuntimeError(
            "scanner root_keepout_violation_probability_max must be within [0, 1]"
        )
    root_keepout_violation_upper = _wilson_upper(
        root_keepout_violation_count, sample_count
    )
    radius_rmse = float(np.sqrt(np.mean(error * error)))
    pointing_p95 = float(np.quantile(evaluated_pointing, 0.95))
    values = {
        "assembly_sha256": metadata["assembly_sha256"],
        "pmdl_sha256": metadata["pmdl_sha256"],
        "sample_count": sample_count,
        "evaluated_duration_s": float(times[-1] - times[start]),
        "target_orbit_radius_m": target_radius,
        "orbit_radius_rmse_m": radius_rmse,
        "orbit_radius_p95_absolute_error_m": float(np.quantile(np.abs(error), 0.95)),
        "camera_pointing_p95_deg": pointing_p95,
        "camera_pointing_max_deg": float(np.max(evaluated_pointing)),
        "camera_height_range_m": [
            float(np.min(evaluated_heights)),
            float(np.max(evaluated_heights)),
        ],
        "root_keepout_violation_count": root_keepout_violation_count,
        "root_keepout_violation_probability": root_keepout_violation_probability,
        "root_keepout_violation_probability_upper_95_wilson": root_keepout_violation_upper,
        "minimum_root_keepout_clearance_m": float(
            np.min(radius - mission.keep_out_radius_m)
        ),
        "median_orbit_coverage_deg": float(
            np.degrees(np.median(np.abs(phases[:, -1] - phases[:, 0])))
        ),
        "validated_physical_configuration_count": sample_count * time_count,
        "dynamics_completeness": {
            "status": dynamics_completeness.status,
            "modeled_scope": dynamics_completeness.modeled_scope,
            "open_gates": [
                gate.to_dict() for gate in dynamics_completeness.open_gates
            ],
        },
    }
    values["acceptance"] = {
        "orbit_radius_rmse": radius_rmse <= orbit_limit,
        "camera_pointing_p95": pointing_p95 <= pointing_limit,
        "root_keepout_violation_probability": (
            root_keepout_violation_upper <= root_keepout_violation_limit
        ),
        "dynamics_completeness": dynamics_completeness.complete,
    }
    values["accepted"] = all(values["acceptance"].values())
    return values


__all__ = [
    "ScannerAssemblyController",
    "ScannerMission",
    "ScannerRuntimeError",
    "load_scanner_assembly",
    "scanner_metrics",
    "scanner_physical_scene",
    "simulate_scanner_robot",
]
