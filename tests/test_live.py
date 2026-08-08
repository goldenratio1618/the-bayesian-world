from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from contraption.controls import SignalSpec
from contraption.live import (
    LiveRequestError,
    LiveScannerApplication,
    make_live_handler,
)
from contraption.resolved import ResolvedAssembly


ASSEMBLY_HASH = "sha256:" + "a" * 64
CONTROLLER_HASH = "sha256:" + "c" * 64


def _scene() -> dict:
    pose = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    provenance = {
        "kind": "estimated",
        "source": "live-server unit fixture",
        "reference": None,
    }
    body_poses = {"part/body": pose}
    connector_poses: dict[str, dict] = {}
    return {
        "schema": "contraption.physical-scene/v1",
        "assembly_sha256": ASSEMBLY_HASH,
        "contraption_id": "live-fixture",
        "components": [
            {
                "id": "part",
                "package": "fixture.part",
                "model": "fixture_model",
                "physical_role": "part",
                "bodies": [
                    {
                        "id": "body",
                        "local_pose": pose,
                        "solids": [
                            {
                                "id": "case",
                                "geometry": {
                                    "kind": "box",
                                    "dimensions_m": [0.1, 0.1, 0.1],
                                    "mesh_uri": None,
                                },
                                "local_pose": pose,
                                "provenance": provenance,
                            }
                        ],
                    }
                ],
                "connectors": [],
            }
        ],
        "connections": [],
        "body_poses": body_poses,
        "connector_poses": connector_poses,
        "body_pose_frames": {
            "assembly_sha256": ASSEMBLY_HASH,
            "frames": [
                {
                    "time_s": 0.0,
                    "body_poses": body_poses,
                    "connector_poses": connector_poses,
                }
            ],
        },
    }


def _assembly() -> ResolvedAssembly:
    assembly = object.__new__(ResolvedAssembly)
    controller_reference = {
        "id": "fixture.controller",
        "version": "1.0.0",
        "sha256": CONTROLLER_HASH,
        "output_bindings": {},
        "telemetry_outputs": ("status",),
    }
    controller = SimpleNamespace(
        name="fixture.controller",
        version="1.0.0",
        inputs=(
            SignalSpec(
                "armed",
                value_type="boolean",
                source="external",
                default=False,
                unit="1",
                description="enable the fixture",
            ),
            SignalSpec(
                "target_speed",
                source="external",
                default=0.1,
                minimum=0.0,
                maximum=0.25,
                unit="m/s",
                description="fixture speed",
            ),
            SignalSpec(
                "measured_speed",
                source="sensor",
                default=0.0,
                minimum=-1.0,
                maximum=1.0,
                unit="m/s",
            ),
        ),
    )
    values = {
        "specification": SimpleNamespace(controller=controller_reference),
        "packages": None,
        "component_models": None,
        "connector_bindings": None,
        "controller": controller,
        "physical": SimpleNamespace(assembly_sha256=ASSEMBLY_HASH),
        "system": None,
    }
    for name, value in values.items():
        object.__setattr__(assembly, name, value)
    return assembly


class _SimulationStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, _assembly: ResolvedAssembly, **options):
        self.calls.append(options)
        if options["external_inputs"]["target_speed"] == 0.2:
            raise RuntimeError("deliberate simulator failure")
        return SimpleNamespace(metadata={"pose_frame_sample_index": 0})


class LiveScannerApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulate = _SimulationStub()
        self.scene_patch = mock.patch(
            "contraption.live.scanner_physical_scene",
            side_effect=lambda _assembly, _result: _scene(),
        )
        self.scene_patch.start()
        self.addCleanup(self.scene_patch.stop)
        self.application = LiveScannerApplication(
            _assembly(), simulate=self.simulate, duration=0.1, dt=0.05
        )

    @staticmethod
    def valid_inputs(*, target_speed: float = 0.15) -> dict:
        return {"armed": True, "target_speed": target_speed}

    def request(self, *, inputs: dict | None = None, digest: str = ASSEMBLY_HASH) -> dict:
        return {
            "assembly_sha256": digest,
            "inputs": self.valid_inputs() if inputs is None else inputs,
        }

    def test_schema_is_derived_only_from_external_controller_inputs(self) -> None:
        schema = self.application.control_schema()
        self.assertEqual(schema["assembly_sha256"], ASSEMBLY_HASH)
        self.assertEqual(
            schema["controller"],
            {
                "id": "fixture.controller",
                "version": "1.0.0",
                "sha256": CONTROLLER_HASH,
            },
        )
        self.assertEqual(
            [item["name"] for item in schema["inputs"]],
            ["armed", "target_speed"],
        )
        self.assertNotIn("measured_speed", schema["values"])

    def test_valid_controls_rerun_python_and_return_only_the_scene(self) -> None:
        response = self.application.simulate_request(self.request())
        self.assertEqual(response["schema"], "contraption.physical-scene/v1")
        self.assertEqual(response["assembly_sha256"], ASSEMBLY_HASH)
        self.assertIn("body_pose_frames", response)
        self.assertNotIn("inputs", response)
        self.assertEqual(
            self.simulate.calls[-1]["external_inputs"],
            {"armed": True, "target_speed": 0.15},
        )

    def test_viewer_is_generated_from_the_assembly_and_actual_result(self) -> None:
        artifact = SimpleNamespace(files={"index.html": "canonical viewer"})
        with mock.patch(
            "contraption.live.generate_viewer", return_value=artifact
        ) as generate:
            files = self.application.viewer_files()
        self.assertEqual(files, artifact.files)
        positional = generate.call_args.args
        self.assertIs(positional[0], self.application.assembly)
        self.assertIs(positional[1], self.application._result)
        self.assertEqual(generate.call_args.kwargs["sample_index"], 0)

    def test_stale_hash_and_invalid_control_sets_fail_before_simulation(self) -> None:
        initial_calls = len(self.simulate.calls)
        cases = (
            (self.request(digest="sha256:" + "b" * 64), 409, "assembly_mismatch"),
            (self.request(inputs={"armed": True}), 400, "invalid_controls"),
            (
                self.request(
                    inputs={"armed": True, "target_speed": 0.1, "extra": 2.0}
                ),
                400,
                "invalid_controls",
            ),
            (
                self.request(inputs={"armed": 1, "target_speed": 0.1}),
                400,
                "invalid_controls",
            ),
            (
                self.request(inputs={"armed": True, "target_speed": 1.0}),
                400,
                "invalid_controls",
            ),
        )
        for request, status, code in cases:
            with self.subTest(code=code, request=request):
                with self.assertRaises(LiveRequestError) as caught:
                    self.application.simulate_request(request)
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(caught.exception.code, code)
        self.assertEqual(len(self.simulate.calls), initial_calls)

    def test_simulation_failure_is_loud_and_does_not_publish_partial_state(self) -> None:
        before = self.application.control_schema()["values"]
        with self.assertRaises(LiveRequestError) as caught:
            self.application.simulate_request(
                self.request(inputs=self.valid_inputs(target_speed=0.2))
            )
        self.assertEqual(caught.exception.status, 422)
        self.assertEqual(caught.exception.code, "simulation_failed")
        self.assertIn("deliberate simulator failure", str(caught.exception))
        self.assertEqual(self.application.control_schema()["values"], before)

    def test_http_errors_are_structured_and_hash_bound(self) -> None:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_live_handler(self.application)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(server.shutdown)

        def post_bytes(payload: bytes) -> tuple[int, dict]:
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/simulate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                response = urlopen(request, timeout=2.0)
            except HTTPError as exc:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            with response:
                return response.status, json.loads(response.read().decode("utf-8"))

        def post(value: dict) -> tuple[int, dict]:
            return post_bytes(json.dumps(value).encode("utf-8"))

        status, body = post(self.request(digest="sha256:" + "b" * 64))
        self.assertEqual(status, 409)
        self.assertEqual(body["schema"], "contraption.live-error/v1")
        self.assertEqual(body["assembly_sha256"], ASSEMBLY_HASH)
        self.assertEqual(body["code"], "assembly_mismatch")

        status, body = post(
            self.request(inputs=self.valid_inputs(target_speed=0.2))
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["code"], "simulation_failed")

        calls_before_invalid_json = len(self.simulate.calls)
        invalid_payloads = (
            (
                b'{"assembly_sha256":"'
                + ASSEMBLY_HASH.encode("ascii")
                + b'","assembly_sha256":"'
                + ASSEMBLY_HASH.encode("ascii")
                + b'","inputs":{"armed":true,"target_speed":0.1}}'
            ),
            (
                b'{"assembly_sha256":"'
                + ASSEMBLY_HASH.encode("ascii")
                + b'","inputs":{"armed":true,"target_speed":NaN}}'
            ),
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                status, body = post_bytes(payload)
                self.assertEqual(status, 400)
                self.assertEqual(body["code"], "invalid_json")
        self.assertEqual(len(self.simulate.calls), calls_before_invalid_json)


if __name__ == "__main__":
    unittest.main()
