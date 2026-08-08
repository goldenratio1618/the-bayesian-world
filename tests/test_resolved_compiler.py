"""Same-source C99 derivation from complete canonical resolved assemblies."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np

from contraption.assembly import assemble_contraption
from contraption.compiler import (
    IRValidationError,
    OnlineModelIR,
    compile_resolved_assembly,
)
from contraption.controls import ControlProgram
from contraption.dsl import parse_model
from contraption.resolved import ResolvedAssembly


CANONICAL_HASH = "sha256:" + "b" * 64
TEST_GEOMETRY = {
    "kind": "box",
    "dimensions": [0.01, 0.01, 0.01],
    "unit": "m",
    "metadata": {
        "provenance_kind": "estimated",
        "source": "nonphysical compiler unit-test fixture",
    },
}


def _controlled_model(*, singular: bool = False):
    parameters = [
        {"name": "decay", "unit": "Hz", "default": 1.0},
    ]
    relation = "x_dot + decay * x - command"
    if singular:
        parameters = [
            {"name": "time_scale", "unit": "s^4", "default": 1.0},
            {"name": "time_constant", "unit": "s", "default": 1.0},
        ]
        relation = "time_scale * x_dot ** 5 + x / time_constant - command"
    return parse_model(
        {
            "format": "pmdl-1",
            "id": "control.singular.test" if singular else "control.first_order.test",
            "name": "Controlled PMDL test system",
            "version": "1.0.0",
            "domains": ["control"],
            "category": "controlled-state",
            "signal_ports": [
                {"name": "command", "direction": "input", "unit": "Hz"}
            ],
            "states": [
                {
                    "name": "x",
                    "unit": "1",
                    "initial": 0.0,
                    "derivative": "x_dot",
                }
            ],
            "parameters": parameters,
            "relations": [{"name": "dynamics", "expression": relation}],
            "validity": {
                "ranges": {"x": {"lower": -10.0, "upper": 10.0}},
                "assumptions": ["Local compiler test"],
                "max_timestep": 0.05,
            },
        }
    )


def _assembled_system(*, canonical: bool = True, singular: bool = False):
    model = _controlled_model(singular=singular)
    specification = {
        "format": "contraption-1",
        "id": "test.c99-source",
        "name": "Resolved compiler source",
        "version": "1.0.0",
        "components": [
            {"id": "plant", "model": model.id, "geometry": TEST_GEOMETRY}
        ],
        "controls": [
            {
                "id": "command",
                "source": "external.command",
                "target": "plant.command",
                "settings": {"unit": "Hz", "default": 2.0},
                "external": True,
            }
        ],
    }
    return assemble_contraption(
        specification,
        component_models={"plant": model},
        canonical_assembly_sha256=CANONICAL_HASH if canonical else None,
    )


def _controller() -> tuple[ControlProgram, dict]:
    program = ControlProgram.from_dict(
        {
            "name": "fixture_controller",
            "version": "1.0.0",
            "inputs": [
                {
                    "name": "target",
                    "type": "number",
                    "source": "external",
                    "default": 2.0,
                    "unit": "Hz",
                }
            ],
            "outputs": [
                {
                    "name": "command",
                    "type": "number",
                    "source": "output",
                    "default": 2.0,
                    "unit": "Hz",
                }
            ],
            "states": [
                {
                    "name": "run",
                    "outputs": {"command": {"ref": "external.target"}},
                }
            ],
            "initial_state": "run",
        }
    )
    payload = json.dumps(
        program.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    reference = {
        "id": program.name,
        "version": program.version,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "output_bindings": {"command": "external.command"},
        "telemetry_outputs": (),
    }
    return program, reference


def _resolved(
    system=None, *, controller: bool = True, incomplete_dynamics: bool = False
) -> ResolvedAssembly:
    selected = _assembled_system() if system is None else system
    program, reference = _controller()
    assembly = object.__new__(ResolvedAssembly)
    dynamics_completeness = (
        {
            "schema": "contraption.dynamics-completeness/v1",
            "status": "incomplete",
            "modeled_scope": "fixture_local_dynamics_only",
            "parameter_basis": {},
            "gates": [
                {
                    "id": "fixture_missing_reaction",
                    "status": "open",
                    "reason": "Fixture omits a reaction path.",
                }
            ],
        }
        if incomplete_dynamics
        else {
            "schema": "contraption.dynamics-completeness/v1",
            "status": "complete",
            "modeled_scope": "complete_test_fixture",
            "parameter_basis": {},
            "gates": [],
        }
    )
    metadata = {"dynamics_completeness": dynamics_completeness}
    values = {
        "specification": SimpleNamespace(
            controller=reference if controller else None,
            metadata=metadata,
        ),
        "packages": None,
        "component_models": None,
        "connector_bindings": None,
        "controller": program if controller else None,
        "physical": SimpleNamespace(assembly_sha256=CANONICAL_HASH),
        "system": selected,
    }
    for name, value in values.items():
        object.__setattr__(assembly, name, value)
    return assembly


class ResolvedCompilerTests(unittest.TestCase):
    def test_derives_expected_implicit_linearization_and_embeds_hashes(self) -> None:
        system = _assembled_system()
        resolved = _resolved(system)

        artifact = compile_resolved_assembly(
            resolved,
            model_name="resolved_plant",
            expected_assembly_sha256=CANONICAL_HASH,
            expected_pmdl_sha256=system.pmdl_sha256,
        )

        model = artifact.manifest["model"]
        np.testing.assert_allclose(model["A"], [[-1.0]], atol=1e-7)
        np.testing.assert_allclose(model["B"], [[1.0]], atol=1e-7)
        np.testing.assert_allclose(model["dynamics_bias"], [0.0], atol=1e-7)
        self.assertEqual(model["state_names"], ["plant.x"])
        self.assertEqual(model["input_names"], ["external.command"])
        self.assertEqual(artifact.manifest["assembly_sha256"], CANONICAL_HASH)
        self.assertEqual(artifact.manifest["pmdl_sha256"], system.pmdl_sha256)
        self.assertEqual(
            artifact.manifest["controller"],
            {
                name: resolved.specification.controller[name]
                for name in ("id", "version", "sha256")
            },
        )
        self.assertFalse(artifact.manifest["controller_execution"]["emitted"])
        self.assertEqual(artifact.manifest["derivation"]["dF_dq_rank"], 2)
        self.assertIn(CANONICAL_HASH, artifact.header)
        self.assertIn(system.pmdl_sha256, artifact.source)
        self.assertIn(resolved.specification.controller["sha256"], artifact.source)
        self.assertIn("Controller execution: NOT EMITTED", artifact.header)

        syntax = artifact.syntax_check()
        if syntax.compiler is not None:
            self.assertTrue(syntax.ok, syntax.stderr)

    def test_operating_point_matches_local_affine_dynamics(self) -> None:
        system = _assembled_system()

        artifact = compile_resolved_assembly(
            _resolved(system),
            operating_state={"plant.x": 1.0},
            operating_controls={"external.command": 3.0},
        )

        model = artifact.manifest["model"]
        np.testing.assert_allclose(model["A"], [[-1.0]], atol=1e-7)
        np.testing.assert_allclose(model["B"], [[1.0]], atol=1e-7)
        np.testing.assert_allclose(model["dynamics_bias"], [0.0], atol=1e-7)
        np.testing.assert_allclose(
            artifact.manifest["derivation"]["operating_state_derivative"],
            [2.0],
            atol=1e-8,
        )

    def test_refuses_authored_ir_uncanonical_system_and_hash_mismatch(self) -> None:
        system = _assembled_system()
        resolved = _resolved(system)
        artifact = compile_resolved_assembly(resolved)
        authored = OnlineModelIR.from_dict(artifact.manifest["model"])

        with self.assertRaisesRegex(IRValidationError, "refuses authored OnlineModelIR"):
            compile_resolved_assembly(authored)
        with self.assertRaisesRegex(IRValidationError, "bare AssembledPMDLSystem"):
            compile_resolved_assembly(system)
        with self.assertRaisesRegex(IRValidationError, "physical/PMDL assembly hash mismatch"):
            compile_resolved_assembly(_resolved(_assembled_system(canonical=False)))
        with self.assertRaisesRegex(IRValidationError, "assembly hash mismatch"):
            compile_resolved_assembly(
                resolved,
                expected_assembly_sha256="sha256:" + "c" * 64,
            )

    def test_stale_controller_hash_is_rejected(self) -> None:
        resolved = _resolved()
        reference = dict(resolved.specification.controller)
        reference["sha256"] = "sha256:" + "d" * 64
        object.__setattr__(
            resolved, "specification", SimpleNamespace(controller=reference)
        )
        with self.assertRaisesRegex(IRValidationError, "controller content hash"):
            compile_resolved_assembly(resolved)

    def test_no_controller_is_explicit_in_manifest_and_source(self) -> None:
        artifact = compile_resolved_assembly(_resolved(controller=False))
        self.assertIsNone(artifact.manifest["controller"])
        self.assertIn("Controller: none", artifact.source)
        self.assertIn("Controller execution: NOT APPLICABLE", artifact.source)
        self.assertEqual(
            artifact.manifest["controller_execution"]["contract"],
            "no_controller_declared",
        )
        self.assertFalse(artifact.manifest["controller_execution"]["emitted"])

    def test_incomplete_dynamics_are_prominent_in_manifest_and_source(self) -> None:
        artifact = compile_resolved_assembly(
            _resolved(incomplete_dynamics=True), model_name="incomplete_fixture"
        )
        completeness = artifact.manifest["dynamics_completeness"]
        self.assertEqual(completeness["status"], "incomplete")
        self.assertEqual(
            completeness["gates"][0]["id"], "fixture_missing_reaction"
        )
        self.assertEqual(
            artifact.manifest["model"]["metadata"]["dynamics_completeness"],
            completeness,
        )
        self.assertIn("Dynamics completeness: INCOMPLETE", artifact.source)
        self.assertIn("fixture_missing_reaction", artifact.header)

    def test_missing_mandatory_dynamics_record_is_a_compiler_validation_error(self) -> None:
        resolved = _resolved()
        object.__setattr__(
            resolved,
            "specification",
            SimpleNamespace(
                controller=resolved.specification.controller,
                metadata={},
            ),
        )
        with self.assertRaisesRegex(
            IRValidationError, "mandatory dynamics_completeness record"
        ):
            compile_resolved_assembly(resolved)

    def test_singular_implicit_derivative_is_rejected_with_rank_diagnostic(self) -> None:
        with self.assertRaisesRegex(
            IRValidationError,
            r"singular dF/d\[xdot,a\]|ill-conditioned",
        ):
            compile_resolved_assembly(_resolved(_assembled_system(singular=True)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
