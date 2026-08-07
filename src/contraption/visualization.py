"""Generate a self-contained, dependency-free contraption browser viewer.

The renderer is intentionally separate from simulation.  It consumes only a
JSON-compatible specification, an optional serialized trajectory, and an
optional constrained online IR.  The resulting HTML performs no network
requests and contains its JavaScript, CSS, model geometry, wiring graph, and
trajectory inline.  The bundled application supports mouse/touch rotation,
wheel zoom, playback/scrubbing, external-control sliders, global/per-component
transparency, component meshes/primitives, and an electrical connection view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping


class VisualizationError(ValueError):
    """Raised when an input cannot be serialized into the offline viewer."""


from .paths import asset_root


_ASSET_DIRECTORY = asset_root() / "web"


def _convert_object(value: Any, *, include_samples: bool = True) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VisualizationError("viewer data cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {str(key): _convert_object(item, include_samples=include_samples) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_object(item, include_samples=include_samples) for item in value]
    if hasattr(value, "tolist"):
        return _convert_object(value.tolist(), include_samples=include_samples)
    if hasattr(value, "item"):
        try:
            return _convert_object(value.item(), include_samples=include_samples)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict(include_samples=include_samples)
        except TypeError:
            converted = value.to_dict()
        return _convert_object(converted, include_samples=include_samples)
    if is_dataclass(value):
        return _convert_object(asdict(value), include_samples=include_samples)
    raise VisualizationError(f"unsupported viewer data type: {type(value).__name__}")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    converted = _convert_object(value)
    if not isinstance(converted, Mapping):
        raise VisualizationError(f"{label} must serialize to an object")
    return converted


def _assets() -> tuple[str, str, str]:
    template_path = _ASSET_DIRECTORY / "viewer.html"
    script_path = _ASSET_DIRECTORY / "viewer.js"
    style_path = _ASSET_DIRECTORY / "style.css"
    missing = [str(path) for path in (template_path, script_path, style_path) if not path.is_file()]
    if missing:
        raise VisualizationError(f"viewer assets are missing: {', '.join(missing)}")
    return (
        template_path.read_text(encoding="utf-8"),
        script_path.read_text(encoding="utf-8"),
        style_path.read_text(encoding="utf-8"),
    )


def _script_json(value: Mapping[str, Any]) -> str:
    # JSON is parsed from a script text node. Escaping HTML-significant and line
    # separator characters prevents a value such as ``</script>`` from ending
    # that node or changing the surrounding document.
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


@dataclass(frozen=True)
class VisualizationArtifact:
    """Standalone page plus its inspectable source assets and normalized data."""

    title: str
    html: str
    javascript: str
    stylesheet: str
    data: Mapping[str, Any]

    @property
    def data_json(self) -> str:
        return json.dumps(self.data, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @property
    def files(self) -> Mapping[str, str]:
        return {
            "index.html": self.html,
            "viewer.js": self.javascript,
            "style.css": self.stylesheet,
            "viewer-data.json": self.data_json,
        }

    def write(self, destination: str | Path) -> Mapping[str, Path]:
        path = Path(destination)
        if path.suffix.lower() in {".html", ".htm"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.html, encoding="utf-8")
            return {path.name: path}
        if path.suffix:
            raise VisualizationError("viewer output must be an HTML file or directory")
        path.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}
        for filename, contents in self.files.items():
            target = path / filename
            target.write_text(contents, encoding="utf-8")
            results[filename] = target
        return results


def generate_viewer(
    specification: Mapping[str, Any] | Any,
    trajectory: Mapping[str, Any] | Any | None = None,
    output: str | Path | None = None,
    *,
    title: str | None = None,
    runtime_model: Mapping[str, Any] | Any | None = None,
) -> VisualizationArtifact:
    """Create the offline viewer and optionally write it to a file/directory.

    ``trajectory`` accepts the simulator's ``SimulationResult`` directly; its
    ``to_dict(include_samples=True)`` representation is embedded.  Supplying a
    constrained ``runtime_model`` (for example ``OnlineModelIR``) lets the
    browser integrate its affine state model locally when external sliders
    change, without a server or network dependency.
    """

    spec = _object(specification, "specification")
    simulation = {} if trajectory is None else _object(trajectory, "trajectory")
    runtime = None if runtime_model is None else _object(runtime_model, "runtime_model")
    default_title = str(spec.get("name", spec.get("id", "Contraption viewer")))
    page_title = default_title if title is None else str(title)
    if not page_title.strip():
        raise VisualizationError("viewer title must not be empty")
    payload: dict[str, Any] = {
        "schema": "contraption.viewer/v1",
        "title": page_title,
        "specification": spec,
        "simulation": simulation,
        "runtime": runtime,
    }
    template, script, style = _assets()
    required_markers = ("@@TITLE@@", "@@STYLE@@", "@@DATA@@", "@@SCRIPT@@")
    missing = [marker for marker in required_markers if marker not in template]
    if missing:
        raise VisualizationError(f"viewer template is missing markers: {', '.join(missing)}")
    escaped_title = html.escape(page_title, quote=True).encode(
        "ascii", "xmlcharrefreplace"
    ).decode("ascii")
    standalone = (
        template.replace("@@STYLE@@", style)
        .replace("@@DATA@@", _script_json(payload))
        .replace("@@SCRIPT@@", script)
        .replace("@@TITLE@@", escaped_title)
    )
    artifact = VisualizationArtifact(page_title, standalone, script, style, payload)
    if output is not None:
        artifact.write(output)
    return artifact


generate_visualization = generate_viewer
build_viewer = generate_viewer


__all__ = [
    "VisualizationArtifact",
    "VisualizationError",
    "build_viewer",
    "generate_viewer",
    "generate_visualization",
]
