"""Same-origin live viewer server backed by canonical Python simulation.

The browser is a display and input surface only.  A control POST is validated
against the resolved data-only controller, simulated in Python through the same
``ResolvedAssembly``, and answered with a complete hash-bound physical scene.
No derivative, kinematics, or placement logic is implemented in JavaScript.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
import threading
from typing import Any
from urllib.parse import urlsplit

from ..applications.scanner import simulate_scanner_robot
from ..physics.resolved import ResolvedAssembly
from .scanner_scene import scanner_physical_scene
from .viewer import generate_viewer, validate_physical_scene


_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_REQUEST_BYTES = 64 * 1024


class LiveRequestError(ValueError):
    """A bounded HTTP error safe to return to the local viewer."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = str(code)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LiveRequestError(400, "invalid_request", f"{label} must be an object")
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " and positive" if positive else ""
        raise ValueError(f"{label} must be finite{suffix}")
    return result


class LiveScannerApplication:
    """Thread-safe state for one exact resolved assembly and live viewer."""

    def __init__(
        self,
        assembly: ResolvedAssembly,
        *,
        duration: float | None = None,
        dt: float = 0.05,
        backend: str = "numpy",
        device: str | None = None,
        seed: int = 20260806,
        simulate: Callable[..., Any] = simulate_scanner_robot,
    ) -> None:
        if not isinstance(assembly, ResolvedAssembly):
            raise TypeError("live server requires a canonical ResolvedAssembly")
        if assembly.controller is None or assembly.specification.controller is None:
            raise ValueError("live server requires a hash-bound resolved controller")
        self.assembly = assembly
        self.duration = None if duration is None else _finite(duration, "duration", positive=True)
        self.dt = _finite(dt, "dt", positive=True)
        self.backend = str(backend)
        self.device = device
        self.seed = int(seed)
        self._simulate = simulate
        self._lock = threading.Lock()
        self._signals = {
            signal.name: signal
            for signal in assembly.controller.inputs
            if signal.source == "external"
        }
        self._values = {
            name: signal.default for name, signal in sorted(self._signals.items())
        }
        if not self._signals:
            raise ValueError("resolved controller declares no external live inputs")
        unrenderable = sorted(
            signal.name
            for signal in self._signals.values()
            if signal.value_type == "number"
            and (
                signal.minimum is None
                or signal.maximum is None
                or signal.maximum <= signal.minimum
            )
        )
        if unrenderable:
            raise ValueError(
                "numeric live inputs require finite increasing minimum/maximum "
                f"bounds for an exact UI control: {unrenderable}"
            )
        reference = {
            name: str(assembly.specification.controller[name])
            for name in ("id", "version", "sha256")
        }
        if (
            reference["id"] != assembly.controller.name
            or reference["version"] != assembly.controller.version
            or _HASH.fullmatch(str(reference["sha256"])) is None
        ):
            raise ValueError("resolved controller identity does not match its hash reference")
        self._controller_reference = reference
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
            external_inputs=dict(external_inputs),
        )
        scene = scanner_physical_scene(self.assembly, result)
        normalized = validate_physical_scene(scene)
        if normalized["assembly_sha256"] != self.assembly.assembly_sha256:
            raise ValueError("live simulation returned a scene for another assembly")
        return result, dict(normalized)

    def control_schema(self) -> dict[str, Any]:
        with self._lock:
            values = dict(self._values)
        return {
            "schema": "contraption.live-controls/v1",
            "assembly_sha256": self.assembly.assembly_sha256,
            "controller": dict(self._controller_reference),
            "inputs": [
                {
                    "name": signal.name,
                    "type": signal.value_type,
                    "default": signal.default,
                    "minimum": signal.minimum,
                    "maximum": signal.maximum,
                    "unit": signal.unit,
                    "description": signal.description,
                }
                for signal in sorted(self._signals.values(), key=lambda item: item.name)
            ],
            "values": values,
        }

    def simulate_request(self, value: Any) -> dict[str, Any]:
        request = _object(value, "request")
        if set(request) != {"assembly_sha256", "inputs"}:
            raise LiveRequestError(
                400,
                "invalid_request",
                "request must contain exactly assembly_sha256 and inputs",
            )
        supplied_hash = request["assembly_sha256"]
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
        raw_inputs = _object(request["inputs"], "inputs")
        unknown = sorted(set(raw_inputs) - set(self._signals))
        missing = sorted(set(self._signals) - set(raw_inputs))
        if unknown or missing:
            raise LiveRequestError(
                400,
                "invalid_controls",
                f"external controls must match exactly; missing={missing}, unknown={unknown}",
            )
        try:
            normalized = {
                name: self._signals[name].normalize(
                    raw_inputs[name], f"external control {name!r}"
                )
                for name in sorted(self._signals)
            }
        except Exception as exc:
            raise LiveRequestError(400, "invalid_controls", str(exc)) from exc
        with self._lock:
            try:
                result, scene = self._run(normalized)
            except LiveRequestError:
                raise
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
            title="Apartment scanner robot — live canonical simulation",
            live={
                "schema_endpoint": "/api/schema",
                "simulate_endpoint": "/api/simulate",
            },
        )
        return artifact.files


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def make_live_handler(application: LiveScannerApplication) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one live application."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "ContraptionLive/1"

        def _send(
            self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8"
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, error: LiveRequestError) -> None:
            self._send(
                error.status,
                _json_bytes(
                    {
                        "schema": "contraption.live-error/v1",
                        "assembly_sha256": application.assembly.assembly_sha256,
                        "code": error.code,
                        "error": str(error),
                    }
                ),
            )

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlsplit(self.path).path
            if path == "/api/schema":
                self._send(200, _json_bytes(application.control_schema()))
                return
            filenames = {
                "/": "index.html",
                "/index.html": "index.html",
                "/viewer.js": "viewer.js",
                "/style.css": "style.css",
                "/viewer-data.json": "viewer-data.json",
            }
            filename = filenames.get(path)
            if filename is None:
                self._error(LiveRequestError(404, "not_found", "resource not found"))
                return
            files = application.viewer_files()
            body = files[filename].encode("utf-8")
            content_type = {
                "index.html": "text/html; charset=utf-8",
                "viewer.js": "text/javascript; charset=utf-8",
                "style.css": "text/css; charset=utf-8",
                "viewer-data.json": "application/json; charset=utf-8",
            }[filename]
            self._send(200, body, content_type)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if urlsplit(self.path).path != "/api/simulate":
                self._error(LiveRequestError(404, "not_found", "resource not found"))
                return
            if self.headers.get_content_type() != "application/json":
                self._error(
                    LiveRequestError(415, "content_type", "Content-Type must be application/json")
                )
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length < 1 or length > _MAX_REQUEST_BYTES:
                self._error(
                    LiveRequestError(
                        413,
                        "request_size",
                        f"request body must be 1..{_MAX_REQUEST_BYTES} bytes",
                    )
                )
                return
            try:
                value = json.loads(
                    self.rfile.read(length).decode("utf-8"),
                    object_pairs_hook=_strict_json_object,
                    parse_constant=_reject_json_constant,
                )
            except UnicodeDecodeError as exc:
                self._error(LiveRequestError(400, "invalid_json", str(exc)))
                return
            except (json.JSONDecodeError, ValueError) as exc:
                self._error(LiveRequestError(400, "invalid_json", str(exc)))
                return
            try:
                scene = application.simulate_request(value)
            except LiveRequestError as exc:
                self._error(exc)
                return
            self._send(HTTPStatus.OK, _json_bytes(scene))

        def log_message(self, format: str, *args: Any) -> None:
            # Preserve stdlib request logging while keeping the exact local bind
            # visible.  No request bodies or controller values are logged.
            super().log_message(format, *args)

    return Handler


def serve_live_scanner(
    application: LiveScannerApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "live viewer defaults to loopback; non-loopback exposure requires a "
            "separately authenticated deployment"
        )
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError("port must be an integer from 1 through 65535")
    server = ThreadingHTTPServer((host, port), make_live_handler(application))
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = [
    "LiveRequestError",
    "LiveScannerApplication",
    "make_live_handler",
    "serve_live_scanner",
]
