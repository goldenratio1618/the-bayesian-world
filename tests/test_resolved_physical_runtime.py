from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from contraption.physics.controls import load_control_program
from contraption.catalog.instantiations import (
    PartInstantiationRegistry,
    StaticPartSpec,
)
from contraption.catalog.interfaces import load_interface_catalog
from contraption.physics.dsl import ModelRegistry
from contraption.physics.resolved import ResolutionError, resolve_assembly
from contraption.applications.scanner import ScannerAssemblyController, ScannerRuntimeError
from contraption.physics.simulator import SimulationResult, simulate
from contraption.physics.specs import FrozenDict, ModelSpec
from contraption.physics.uq import summarize_samples


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "scanner_robot"


def model_digest(model: ModelSpec) -> str:
    return "sha256:" + hashlib.sha256(model.to_json().encode("utf-8")).hexdigest()


def scanner_inputs():
    specification = json.loads((EXAMPLE / "contraption.json").read_text())
    catalog_root = ROOT / "model_catalog"
    interfaces = load_interface_catalog(catalog_root)
    models = ModelRegistry()
    models.load_directory(catalog_root, interfaces=interfaces)
    instantiations = PartInstantiationRegistry.load_catalog(catalog_root, models=models)
    program = load_control_program(EXAMPLE / "controls" / "scanner.control.json")
    return specification, instantiations, models, {program.name: program}


def synthetic_result(assembly) -> SimulationResult:
    initial = np.asarray(assembly.system.initial_state, dtype=float)
    samples = np.broadcast_to(initial, (2, 2, initial.size)).copy()
    # Only sample one moves.  This lets the all-sample admission test prove
    # that rendering sample zero cannot conceal a bad sample one.
    samples[1, 1, assembly.system.state_names.index("chassis.position_x")] += 0.01
    samples[1, 1, assembly.system.state_names.index("left-wheel.angle")] = 0.2
    samples[1, 1, assembly.system.state_names.index("right-wheel.angle")] = -0.2
    times = np.asarray([0.0, 0.1], dtype=float)
    output_names: tuple[str, ...] = ()
    output_samples = np.zeros((2, 2, len(output_names)), dtype=float)
    return SimulationResult(
        time=times,
        state_names=tuple(assembly.system.state_names),
        samples=samples,
        output_names=output_names,
        output_samples=output_samples,
        summary=summarize_samples(samples),
        output_summary=summarize_samples(output_samples),
        metadata={
            "assembly_sha256": assembly.assembly_sha256,
            "pmdl_sha256": assembly.system.pmdl_sha256,
            "sample_count": 2,
        },
    )


class ResolvedPhysicalRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        inputs = scanner_inputs()
        cls.assembly = resolve_assembly(
            inputs[0], inputs[1], inputs[2], control_programs=inputs[3]
        )

    def test_part_measures_root_states_and_controller_are_one_closure(self) -> None:
        assembly = self.assembly
        wheel = assembly.parts["scanner.wheel.v1"]
        chassis = assembly.parts["scanner.romi_chassis.v1"]
        self.assertAlmostEqual(
            wheel.measure_parameter(wheel.parameter_binding_map["radius"]), 0.035
        )
        self.assertAlmostEqual(
            chassis.measure_parameter(chassis.parameter_binding_map["wheel_base"]),
            0.15,
        )
        self.assertEqual(assembly.physical.root_state_binding.x, "chassis.position_x")
        self.assertEqual(assembly.controller.name, "scanner_orbit_controller")
        self.assertEqual(
            dict(assembly.controller_output_bindings),
            {
                "left_voltage": "scanner.left_voltage",
                "right_voltage": "scanner.right_voltage",
                "lift_target": "scanner.lift_angle",
                "tilt_target": "scanner.tilt_angle",
            },
        )
        self.assertEqual(assembly.controller_telemetry_outputs, ("record_video",))

    def test_configuration_from_state_uses_canonical_bindings(self) -> None:
        assembly = self.assembly
        configured = assembly.configuration_from_state(assembly.system.initial_state)
        self.assertEqual(configured.body_poses, assembly.physical.body_poses)
        self.assertEqual(configured.connector_poses, assembly.physical.connector_poses)

        state = np.asarray(assembly.system.initial_state, dtype=float)
        state[assembly.system.state_names.index("left-wheel.angle")] = 0.4
        state[assembly.system.state_names.index("right-wheel.angle")] = -0.4
        configured = assembly.configuration_from_state(state)
        for wheel, drive in (
            ("left-wheel.translation", "chassis.left_drive"),
            ("right-wheel.translation", "chassis.right_drive"),
        ):
            self.assertLess(
                configured.connector_poses[wheel].angular_distance(
                    configured.connector_poses[drive]
                ),
                1e-12,
            )

        with self.assertRaisesRegex(ResolutionError, "missing binding"):
            assembly.configuration_from_state({"chassis.position_x": 0.85})

    def test_live_state_validation_is_batch_capable_and_contextual(self) -> None:
        assembly = self.assembly
        initial = np.asarray(assembly.system.initial_state, dtype=float)
        configured = assembly.validate_simulation_state(
            initial,
            state_names=assembly.system.state_names,
            sample_index=3,
            step_index=0,
            time_s=0.0,
            require_initial_configuration=True,
        )
        self.assertEqual(configured.body_poses, assembly.physical.body_poses)

        batch = np.broadcast_to(initial, (2, initial.size)).copy()
        batch[1, assembly.system.state_names.index("chassis.position_x")] += 0.01
        configurations = assembly.validate_simulation_state(
            batch,
            state_names=assembly.system.state_names,
            step_index=4,
            time_s=0.2,
        )
        self.assertEqual(len(configurations), 2)
        self.assertNotEqual(
            configurations[0].body_poses["chassis/body"],
            configurations[1].body_poses["chassis/body"],
        )

        batch[1, assembly.system.state_names.index("arm-linkage.angle")] = 0.1
        with self.assertRaisesRegex(
            ResolutionError,
            r"sample=1, step_index=5, time_s=0.25.*arm-lift-joint.*holonomic coordinate mismatch",
        ):
            assembly.validate_simulation_state(
                batch,
                state_names=assembly.system.state_names,
                step_index=5,
                time_s=0.25,
            )

        changed_initial = initial.copy()
        changed_initial[assembly.system.state_names.index("chassis.position_x")] += 0.01
        with self.assertRaisesRegex(
            ResolutionError, r"sample=9, step_index=0.*hash-bound static assembly scene"
        ):
            assembly.validate_simulation_state(
                changed_initial,
                state_names=assembly.system.state_names,
                sample_index=9,
                step_index=0,
                time_s=0.0,
                require_initial_configuration=True,
            )

    def test_generic_simulator_calls_the_physical_step_hook_at_t0(self) -> None:
        assembly = replace(self.assembly, controller=None)
        changed_initial = np.asarray(assembly.system.initial_state, dtype=float).copy()
        changed_initial[
            assembly.system.state_names.index("chassis.position_x")
        ] += 0.01
        with self.assertRaisesRegex(
            ResolutionError,
            r"step_index=0, time_s=0.*hash-bound static assembly scene",
        ):
            simulate(
                assembly,
                times=(0.0, 0.001),
                initial_state=changed_initial,
                controls=dict(assembly.system.control_defaults),
                num_samples=1,
                use_model_uncertainty=False,
                process_noise=False,
            )

    def test_camera_target_is_exactly_reachable_across_lift_sweep(self) -> None:
        assembly = self.assembly
        target = np.asarray(
            assembly.specification.environment["object_bounding_cube"]["center_m"],
            dtype=float,
        )
        initial = dict(
            zip(
                assembly.system.state_names,
                assembly.system.initial_state,
                strict=True,
            )
        )
        lower, upper = assembly.system.control_bounds["scanner.tilt_angle"]
        assert lower is not None and upper is not None

        for lift_angle in (-0.2, 0.0, 0.25, 0.5, 0.7):
            with self.subTest(lift_angle=lift_angle):
                state = dict(initial)
                state["arm-linkage.angle"] = lift_angle
                state["lift-servo.angle"] = lift_angle
                state["camera.angle"] = 0.0
                state["tilt-servo.angle"] = 0.0
                configured = assembly.configuration_from_state(state)

                optical = configured.connector_poses["camera.optical_axis"]
                pivot = configured.connector_poses["tilt-servo.shaft"]
                optical_origin = np.asarray(optical.translation_m, dtype=float)
                pivot_origin = np.asarray(pivot.translation_m, dtype=float)
                view = (
                    np.asarray(optical.apply((0.0, 0.0, 1.0)), dtype=float)
                    - optical_origin
                )
                tilt_axis = (
                    np.asarray(pivot.apply((0.0, 0.0, 1.0)), dtype=float)
                    - pivot_origin
                )
                pivot_to_target = target - pivot_origin
                pivot_to_target /= np.linalg.norm(pivot_to_target)

                # A one-axis camera can look at the target iff the target ray
                # lies in the plane normal to the declared revolute axis.
                self.assertLess(abs(float(np.dot(pivot_to_target, tilt_axis))), 1e-12)
                required_tilt = math.atan2(
                    float(np.dot(tilt_axis, np.cross(view, pivot_to_target))),
                    float(np.dot(view, pivot_to_target)),
                )
                self.assertGreaterEqual(required_tilt, lower)
                self.assertLessEqual(required_tilt, upper)

                state["camera.angle"] = required_tilt
                state["tilt-servo.angle"] = required_tilt
                pointed = assembly.configuration_from_state(state)
                pointed_optical = pointed.connector_poses["camera.optical_axis"]
                pointed_origin = np.asarray(
                    pointed_optical.translation_m, dtype=float
                )
                pointed_view = (
                    np.asarray(
                        pointed_optical.apply((0.0, 0.0, 1.0)), dtype=float
                    )
                    - pointed_origin
                )
                target_ray = target - pointed_origin
                target_ray /= np.linalg.norm(target_ray)
                pointing_error = math.acos(
                    float(np.clip(np.dot(pointed_view, target_ray), -1.0, 1.0))
                )
                self.assertLess(pointing_error, 5e-8)

    def test_exact_sample_frames_and_all_sample_validation(self) -> None:
        assembly = self.assembly
        result = synthetic_result(assembly)
        assembly.validate_simulation_result(result)
        wrapper = assembly.body_pose_frames(result, sample_index=1)
        self.assertEqual(wrapper["assembly_sha256"], assembly.assembly_sha256)
        self.assertEqual(len(wrapper["frames"]), 2)
        self.assertEqual(
            wrapper["frames"][0]["body_poses"],
            assembly.scene["body_poses"],
        )
        self.assertEqual(
            wrapper["frames"][0]["connector_poses"],
            assembly.scene["connector_poses"],
        )
        self.assertNotEqual(
            wrapper["frames"][1]["body_poses"]["chassis/body"],
            wrapper["frames"][0]["body_poses"]["chassis/body"],
        )

        mismatched_samples = np.asarray(result.samples).copy()
        mismatched_samples[
            1, 1, assembly.system.state_names.index("arm-linkage.angle")
        ] = 0.1
        holonomic_mismatch = replace(
            result,
            samples=mismatched_samples,
            summary=summarize_samples(mismatched_samples),
        )
        with self.assertRaisesRegex(
            ResolutionError,
            r"sample=1, time_index=1.*arm-lift-joint.*holonomic coordinate mismatch",
        ):
            assembly.validate_simulation_result(holonomic_mismatch)

        wheel = assembly.physical.parts["scanner.wheel.v1"]
        uncountered = replace(
            wheel,
            connectors=tuple(
                replace(connector, kinematics=None)
                if connector.id == "translation"
                else connector
                for connector in wheel.connectors
            ),
        )
        parts = dict(assembly.physical.parts)
        parts[wheel.id] = uncountered
        bad_physical = replace(
            assembly.physical,
            parts=FrozenDict(parts),
        )
        bad_assembly = replace(assembly, physical=bad_physical)
        with self.assertRaisesRegex(
            ResolutionError, r"sample=1, time_index=1.*right-floor-contact|sample=1, time_index=1.*left-floor-contact"
        ):
            bad_assembly.body_pose_frames(result, sample_index=0)

    def test_result_provenance_sample_selection_and_means_fail_closed(self) -> None:
        assembly = self.assembly
        result = synthetic_result(assembly)
        stale = replace(
            result,
            metadata={**result.metadata, "assembly_sha256": "sha256:" + "0" * 64},
        )
        with self.assertRaisesRegex(ResolutionError, "assembly_sha256.*stale"):
            assembly.validate_simulation_result(stale)
        with self.assertRaisesRegex(ResolutionError, "outside"):
            assembly.body_pose_frames(result, sample_index=2)
        with self.assertRaisesRegex(ResolutionError, "means are not accepted"):
            assembly.body_pose_frames(result.mean, sample_index=0)  # type: ignore[arg-type]

        wrong_initial_samples = np.asarray(result.samples).copy()
        wrong_initial_samples[
            1, 0, assembly.system.state_names.index("chassis.position_x")
        ] += 0.01
        wrong_initial = replace(
            result,
            samples=wrong_initial_samples,
            summary=summarize_samples(wrong_initial_samples),
        )
        with self.assertRaisesRegex(
            ResolutionError,
            r"sample=1, time_index=0.*hash-bound static assembly scene",
        ):
            assembly.validate_simulation_result(wrong_initial)

    def test_controller_reference_and_geometry_parameter_drift_fail_closed(self) -> None:
        specification, packages, models, programs = scanner_inputs()
        with self.assertRaisesRegex(ResolutionError, "no parsed control-program registry"):
            resolve_assembly(specification, packages, models)

        stale = copy.deepcopy(specification)
        stale["controller"]["sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ResolutionError, "content hash mismatch"):
            resolve_assembly(stale, packages, models, control_programs=programs)

        missing_output = copy.deepcopy(specification)
        del missing_output["controller"]["output_bindings"]["left_voltage"]
        with self.assertRaisesRegex(ResolutionError, "output coverage mismatch"):
            resolve_assembly(
                missing_output, packages, models, control_programs=programs
            )

        changed_values = []
        for item in packages.values():
            if item.id == "scanner.wheel.v1":
                changed_parameters = dict(item.model_instance.parameters)
                changed_parameters["radius"] = 0.04
                item = replace(
                    item,
                    model_instance=replace(
                        item.model_instance,
                        parameters=FrozenDict(changed_parameters),
                    ),
                )
            changed_values.append(item)
        wrong_radius = PartInstantiationRegistry(changed_values)
        with self.assertRaisesRegex(
            ResolutionError, r"radius.*disagrees with part physical measure"
        ):
            resolve_assembly(
                specification, wrong_radius, models, control_programs=programs
            )

        missing_joint_state = copy.deepcopy(specification)
        arm_joint = next(
            connection
            for connection in missing_joint_state["connections"]
            if connection["id"] == "arm-lift-joint"
        )
        arm_joint["joint"]["coordinate_bindings"] = [
            arm_joint["joint"]["coordinate_bindings"][0]
        ]
        with self.assertRaisesRegex(
            ResolutionError, r"coordinate-binding coverage mismatch.*lift-servo.angle"
        ):
            resolve_assembly(
                missing_joint_state,
                packages,
                models,
                control_programs=programs,
            )

    def test_scanner_dynamics_completeness_and_inertia_basis_fail_closed(self) -> None:
        specification, packages, models, programs = scanner_inputs()

        missing_record = copy.deepcopy(specification)
        del missing_record["metadata"]["dynamics_completeness"]
        with self.assertRaisesRegex(
            ResolutionError, "requires metadata.dynamics_completeness"
        ):
            resolve_assembly(
                missing_record, packages, models, control_programs=programs
            )

        unknown_field = copy.deepcopy(specification)
        unknown_field["metadata"]["dynamics_completeness"]["silent_override"] = True
        with self.assertRaisesRegex(ResolutionError, "unknown.*silent_override"):
            resolve_assembly(
                unknown_field, packages, models, control_programs=programs
            )

        missing_gate = copy.deepcopy(specification)
        missing_gate["metadata"]["dynamics_completeness"]["gates"].pop()
        missing_gate_assembly = resolve_assembly(
            missing_gate, packages, models, control_programs=programs
        )
        with self.assertRaisesRegex(
            ScannerRuntimeError, "exactly these open gates"
        ):
            ScannerAssemblyController(missing_gate_assembly)

        chassis = packages["scanner.romi_chassis.v1"]
        static_data = chassis.static.to_dict()
        static_data["bodies"][0]["solids"][0]["geometry"]["dimensions_m"][0] = 0.17
        changed_chassis = replace(
            chassis, static=StaticPartSpec.from_dict(static_data)
        )
        changed_parts = PartInstantiationRegistry(
            changed_chassis if item.id == chassis.id else item
            for item in packages.values()
        )
        changed_assembly = resolve_assembly(
            specification,
            changed_parts,
            models,
            control_programs=programs,
        )
        with self.assertRaisesRegex(
            ScannerRuntimeError, "yaw_inertia.*canonical solid"
        ):
            ScannerAssemblyController(changed_assembly)

    def test_geometry_bound_pmdl_uncertainty_is_refused(self) -> None:
        specification, _packages, models, programs = scanner_inputs()
        raw_model = json.loads(
            (
                ROOT
                / "model_catalog"
                / "mechanical"
                / "wheels"
                / "rolling_drive_wheels"
                / "rolling_wheel.pmdl"
            ).read_text()
        )
        next(
            parameter
            for parameter in raw_model["parameters"]
            if parameter["name"] == "radius"
        )["uncertainty"] = {
            "distribution": "normal",
            "parameters": {"std": 0.001},
        }
        changed_model = ModelSpec.from_dict(raw_model)
        changed_models = dict(models)
        changed_models[changed_model.id] = changed_model

        wheel = _packages["scanner.wheel.v1"]
        changed_wheel = replace(
            wheel,
            model_instance=replace(
                wheel.model_instance,
                model=replace(
                    wheel.model_instance.model,
                    sha256=model_digest(changed_model),
                ),
            ),
        )
        changed_parts = PartInstantiationRegistry(
            changed_wheel if item.id == wheel.id else item
            for item in _packages.values()
        )
        with self.assertRaisesRegex(
            ResolutionError, "may not declare independent PMDL uncertainty"
        ):
            resolve_assembly(
                specification,
                changed_parts,
                changed_models,
                control_programs=programs,
            )

    def test_root_pose_initial_state_disagreement_is_refused(self) -> None:
        specification, packages, models, programs = scanner_inputs()
        specification["physical_root"]["pose"]["translation_m"][0] = 0.9
        with self.assertRaisesRegex(
            ResolutionError, "root pose x=.*disagrees with initial PMDL state"
        ):
            resolve_assembly(
                specification, packages, models, control_programs=programs
            )


if __name__ == "__main__":
    unittest.main()
