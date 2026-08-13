from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil

import pytest

from contraption import (
    ContraptionLoadError,
    ResolutionError,
    ResolvedControllerOutputBinding,
    load_contraption,
    resolve_assembly,
)
from contraption.control import control_digest, parse_control
from contraption.loading import _load_catalogs
from contraption.physics.specs import ContraptionSpec
from contraption.verification import parse_verification


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "assembled_contraptions" / "scanner"
TEST_SYSTEMS = ROOT / "assembled_contraptions" / "examples" / "test_systems"


def _scanner_bundle(tmp_path: Path) -> tuple[Path, dict]:
    source = json.loads((SCANNER / "contraption.json").read_text(encoding="utf-8"))
    manifest = copy.deepcopy(source)
    catalog = os.path.relpath(ROOT / "model_catalog", tmp_path).replace(os.sep, "/")
    manifest["catalogs"] = [{"path": catalog}]
    for collection in ("controllers", "verifications"):
        for link in manifest[collection]:
            relative = Path(link["program"]["path"])
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SCANNER / relative, target)
    path = tmp_path / "contraption.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path, manifest


def _write(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _projected_scanner_controller(
    tmp_path: Path,
    original_link: dict,
    authored: dict,
    identifier: str,
    output_names: set[str],
    filename: str,
) -> dict:
    artifact = copy.deepcopy(authored)
    artifact["id"] = identifier
    artifact["name"] = identifier
    artifact["implicit_inputs"] = []
    artifact.pop("observer", None)
    for derived in artifact["derived"]:
        derived["expression"] = derived["expression"].replace(
            "sqrt(implicit.forward_speed.variance)", "0.0 * input.target_speed"
        ).replace(
            "implicit.forward_speed.mean", "0.0 * input.target_speed"
        )
    artifact["outputs"] = [
        output for output in artifact["outputs"] if output["name"] in output_names
    ]
    for mode in artifact["modes"]:
        mode["outputs"] = {
            name: expression
            for name, expression in mode["outputs"].items()
            if name in output_names
        }
    program = parse_control(artifact)
    artifact_path = tmp_path / "controls" / filename
    artifact_path.write_text(program.to_json(indent=2) + "\n", encoding="utf-8")
    return {
        "id": identifier,
        "program": {
            "path": f"controls/{filename}",
            "sha256": control_digest(program),
        },
        "explicit_inputs": copy.deepcopy(original_link["explicit_inputs"]),
        "implicit_inputs": {},
        "outputs": {
            name: target
            for name, target in original_link["outputs"].items()
            if name in output_names
        },
    }


def test_load_contraption_resolves_closed_scanner_bundle(tmp_path: Path) -> None:
    path, _ = _scanner_bundle(tmp_path)

    assembly = load_contraption(path)

    assert assembly.specification.format == "contraption-4"
    assert tuple(assembly.controllers) == ("scanner_orbit_controller",)
    assert tuple(assembly.verifications) == ("scanner.orbit_acceptance",)
    controller = assembly.controllers["scanner_orbit_controller"]
    signal_outputs = {
        name: binding
        for name, binding in controller.output_bindings.items()
        if binding.kind == "signal"
    }
    assert {binding.source for binding in signal_outputs.values()} == set(
        assembly.system.control_names
    )
    assert controller.output_bindings["record_video"].kind == "external"
    assert controller.output_bindings["record_video"].source == "record_video"
    assert controller.output_bindings["record_video"].state_name is None
    for binding in controller.explicit_input_bindings.values():
        if binding.kind == "external":
            assert binding.state_name is None
            assert binding.state_index is None
        else:
            assert binding.state_name is not None
            assert binding.state_index is not None
            assert assembly.system.state_names[binding.state_index] == binding.state_name
    verification = assembly.verifications["scanner.orbit_acceptance"]
    for binding in verification.input_bindings.values():
        assert assembly.system.state_names[binding.state_index] == binding.state_name


def test_load_contraption_checks_canonical_controller_hash(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    manifest["controllers"][0]["program"]["sha256"] = "sha256:" + "0" * 64
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="canonical content hash mismatch"):
        load_contraption(path)


def test_load_contraption_checks_canonical_verification_hash(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    manifest["verifications"][0]["program"]["sha256"] = "sha256:" + "0" * 64
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="canonical content hash mismatch"):
        load_contraption(path)


def test_resolve_assembly_rejects_mutated_controller_artifact(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    specification = ContraptionSpec.from_dict(manifest)
    controller_path = tmp_path / manifest["controllers"][0]["program"]["path"]
    artifact = json.loads(controller_path.read_text(encoding="utf-8"))
    artifact["parameters"][0]["default"] += 0.01
    controller = parse_control(artifact)
    models, instantiations = _load_catalogs((ROOT / "model_catalog",))

    with pytest.raises(ResolutionError, match="controller artifact.*hash mismatch"):
        resolve_assembly(
            specification,
            instantiations,
            models,
            controller_specs={controller.id: controller},
        )


def test_resolve_assembly_rejects_mutated_verification_artifact() -> None:
    root = TEST_SYSTEMS / "rc_circuit"
    manifest = json.loads((root / "contraption.json").read_text(encoding="utf-8"))
    specification = ContraptionSpec.from_dict(manifest)
    artifact = json.loads((root / "verification.verify").read_text(encoding="utf-8"))
    artifact["parameters"][0]["value"] += 1.0
    verification = parse_verification(artifact)
    models, instantiations = _load_catalogs((root / "catalog",))

    with pytest.raises(ResolutionError, match="verification artifact.*hash mismatch"):
        resolve_assembly(
            specification,
            instantiations,
            models,
            verification_specs={verification.id: verification},
        )


def test_observer_free_bundle_requires_dynamics_completeness(tmp_path: Path) -> None:
    source = TEST_SYSTEMS / "rc_circuit"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "contraption.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["metadata"]["dynamics_completeness"]
    _write(path, manifest)

    with pytest.raises(
        ContraptionLoadError,
        match="requires metadata.dynamics_completeness",
    ):
        load_contraption(path)


def test_load_contraption_rejects_absolute_catalog_path(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    manifest["catalogs"] = [{"path": str((ROOT / "model_catalog").resolve())}]
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="must be a non-empty relative path"):
        load_contraption(path)


def test_load_contraption_rejects_duplicate_actuator_drive(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    outputs = manifest["controllers"][0]["outputs"]
    outputs["right_voltage"] = outputs["left_voltage"]
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="is driven by both"):
        load_contraption(path)


def test_load_contraption_rejects_artifact_parent_escape(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    manifest["controllers"][0]["program"]["path"] = "../controller.control"
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="contained relative path"):
        load_contraption(path)


def test_controller_sensor_requires_signal_binding(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    manifest["controllers"][0]["explicit_inputs"]["heading_x"] = {
        "external": "heading_x"
    }
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="requires a signal binding"):
        load_contraption(path)


def test_verification_input_must_resolve_to_pmdl_state(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    verification = manifest["verifications"][0]
    first_input = next(iter(verification["inputs"]))
    verification["inputs"][first_input] = "chassis.not_a_state"
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="unknown PMDL state"):
        load_contraption(path)


def test_load_contraption_resolves_multiple_controllers(tmp_path: Path) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    original_link = manifest["controllers"][0]
    original_path = tmp_path / original_link["program"]["path"]
    authored = json.loads(original_path.read_text(encoding="utf-8"))

    drive_link = _projected_scanner_controller(
        tmp_path,
        original_link,
        authored,
        "scanner_orbit_controller",
        {"left_voltage", "right_voltage", "lift_target"},
        "drive.control",
    )
    tilt_link = _projected_scanner_controller(
        tmp_path,
        original_link,
        authored,
        "scanner_tilt_controller", {"tilt_target"}, "tilt.control"
    )
    manifest["controllers"] = [drive_link, tilt_link]
    _write(path, manifest)

    assembly = load_contraption(path)

    assert tuple(assembly.controllers) == (
        "scanner_orbit_controller",
        "scanner_tilt_controller",
    )
    assert set(assembly.system.control_names) == {
        *assembly.controllers["scanner_orbit_controller"].plant_output_bindings.values(),
        *assembly.controllers["scanner_tilt_controller"].plant_output_bindings.values(),
    }


def test_load_contraption_rejects_duplicate_external_output_across_controllers(
    tmp_path: Path,
) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    original_link = manifest["controllers"][0]
    authored = json.loads(
        (tmp_path / original_link["program"]["path"]).read_text(encoding="utf-8")
    )
    manifest["controllers"] = [
        _projected_scanner_controller(
            tmp_path,
            original_link,
            authored,
            "camera_controller_a",
            {"record_video"},
            "camera_a.control",
        ),
        _projected_scanner_controller(
            tmp_path,
            original_link,
            authored,
            "camera_controller_b",
            {"record_video"},
            "camera_b.control",
        ),
    ]
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="is exposed by both"):
        load_contraption(path)


def test_controller_outputs_require_typed_signal_or_external_binding(
    tmp_path: Path,
) -> None:
    path, manifest = _scanner_bundle(tmp_path)
    manifest["controllers"][0]["outputs"]["left_voltage"] = (
        "control-board.left_command"
    )
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="must be an object"):
        load_contraption(path)

    path, manifest = _scanner_bundle(tmp_path)
    manifest["controllers"][0]["outputs"]["record_video"] = {
        "signal": "control-board.left_command",
        "external": "record_video",
    }
    _write(path, manifest)

    with pytest.raises(ContraptionLoadError, match="exactly one of signal/external"):
        load_contraption(path)


@pytest.mark.parametrize(
    ("kind", "source", "state_name"),
    [
        ("telemetry", "record_video", None),
        ("external", "", None),
        ("signal", "controller.command", None),
        ("external", "record_video", "plant.command"),
    ],
)
def test_resolved_controller_output_binding_is_strict(
    kind: str, source: str, state_name: str | None
) -> None:
    with pytest.raises(ResolutionError):
        ResolvedControllerOutputBinding(kind, source, state_name)
