"""Same-origin HTTP transport for :mod:`contraption.live`."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.parse import urlsplit

from ..live import LiveApplication, LiveRequestError


_MAX_REQUEST_BYTES = 64 * 1024


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


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


def make_live_handler(application: LiveApplication) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one generic live application."""

    if not isinstance(application, LiveApplication):
        raise TypeError("make_live_handler requires a LiveApplication")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ContraptionLive/2"

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str = "application/json; charset=utf-8",
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
                self._send(HTTPStatus.OK, _json_bytes(application.control_schema()))
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
            self._send(HTTPStatus.OK, body, content_type)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if urlsplit(self.path).path != "/api/simulate":
                self._error(LiveRequestError(404, "not_found", "resource not found"))
                return
            if self.headers.get_content_type() != "application/json":
                self._error(
                    LiveRequestError(
                        415, "content_type", "Content-Type must be application/json"
                    )
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
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._error(LiveRequestError(400, "invalid_json", str(exc)))
                return
            try:
                scene = application.simulate_request(value)
            except LiveRequestError as exc:
                self._error(exc)
                return
            self._send(HTTPStatus.OK, _json_bytes(scene))

    return Handler


def serve_live(
    application: LiveApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "live viewer defaults to loopback; non-loopback exposure requires a "
            "separately authenticated deployment"
        )
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 through 65535")
    server = ThreadingHTTPServer((host, port), make_live_handler(application))
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["make_live_handler", "serve_live"]
