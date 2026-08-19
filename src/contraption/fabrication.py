"""Strict fabrication records shared by parts, assemblies, and build plans.

The records in this module deliberately distinguish a connector-side
capability/requirement from the implementation selected by one contraption.
Missing information is data: an omitted record or ``status: missing`` never
causes a fastener, bearing, conductor, or route to be inferred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any
from urllib.parse import urlparse


class FabricationSpecError(ValueError):
    """A fabrication record is malformed, contradictory, or incomplete."""


COMMON_STANDARD_FAMILIES = frozenset(
    {
        "iso_metric_thread",
        "unified_thread",
        "iso_bearing",
        "abma_bearing",
        "awg",
        "iec_conductor",
        "ipc_land_pattern",
        "manufacturer_connector",
        "manufacturer_package",
        "manufacturer_fastener",
        "manufacturer_cable",
        "extension",
    }
)

_FABRICATION_KINDS = frozenset(
    {"fixed_mount", "rotary_support", "electrical_termination", "optical_alignment", "other"}
)
_STATUSES = frozenset({"missing", "partial", "specified"})
_ROLES = frozenset({"internal", "external", "plug", "receptacle", "neutral"})
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_METRIC_THREAD = re.compile(
    r"^M(?P<diameter>[0-9]+(?:\.[0-9]+)?)x(?P<pitch>[0-9]+(?:\.[0-9]+)?)(?:-[0-9A-Za-z]+)?$"
)


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FabricationSpecError(f"{context} must be an object with string keys")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FabricationSpecError(f"{context} must be an array")
    return value


def _keys(
    value: Mapping[str, Any],
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise FabricationSpecError(f"{context} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise FabricationSpecError(f"{context} is missing fields: {', '.join(missing)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise FabricationSpecError(f"{context} must be a nonempty trimmed string")
    return value


def _optional_text(value: Any, context: str) -> str | None:
    return None if value is None else _text(value, context)


def _number(value: Any, context: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FabricationSpecError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FabricationSpecError(f"{context} must be finite")
    if positive and result <= 0:
        raise FabricationSpecError(f"{context} must be positive")
    if nonnegative and result < 0:
        raise FabricationSpecError(f"{context} must be nonnegative")
    return result


def _optional_number(
    value: Any,
    context: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    return None if value is None else _number(value, context, positive=positive, nonnegative=nonnegative)


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FabricationSpecError(f"{context} must be a positive integer")
    return value


def _url(value: Any, context: str) -> str:
    result = _text(value, context)
    parsed = urlparse(result)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise FabricationSpecError(f"{context} must be an absolute HTTP(S) URL")
    return result


@dataclass(frozen=True, slots=True)
class StandardReferenceSpec:
    """Auditable standard identity plus normalized dimensions where applicable."""

    family: str
    authority: str
    document: str
    designation: str
    revision: str | None = None
    role: str = "neutral"
    uri: str | None = None
    nominal_diameter_m: float | None = None
    pitch_m: float | None = None
    gauge_awg: int | None = None

    def __post_init__(self) -> None:
        if self.family not in COMMON_STANDARD_FAMILIES:
            raise FabricationSpecError(
                f"standard.family must be one of {sorted(COMMON_STANDARD_FAMILIES)}"
            )
        _text(self.authority, "standard.authority")
        _text(self.document, "standard.document")
        _text(self.designation, "standard.designation")
        if self.revision is not None:
            _text(self.revision, "standard.revision")
        if self.role not in _ROLES:
            raise FabricationSpecError(f"standard.role must be one of {sorted(_ROLES)}")
        if self.uri is not None:
            _url(self.uri, "standard.uri")
        if self.family == "extension" and self.uri is None:
            raise FabricationSpecError("extension standards require an authoritative uri")
        diameter = _optional_number(
            self.nominal_diameter_m, "standard.nominal_diameter_m", positive=True
        )
        pitch = _optional_number(self.pitch_m, "standard.pitch_m", positive=True)
        object.__setattr__(self, "nominal_diameter_m", diameter)
        object.__setattr__(self, "pitch_m", pitch)
        if self.gauge_awg is not None and (
            isinstance(self.gauge_awg, bool)
            or not isinstance(self.gauge_awg, int)
            or self.gauge_awg < 0
            or self.gauge_awg > 40
        ):
            raise FabricationSpecError("standard.gauge_awg must be an integer from 0 through 40")
        if self.family in {"iso_metric_thread", "unified_thread"} and (
            diameter is None or pitch is None
        ):
            raise FabricationSpecError(
                f"{self.family} requires nominal_diameter_m and pitch_m"
            )
        if self.family == "iso_metric_thread":
            match = _METRIC_THREAD.fullmatch(self.designation)
            if match is None:
                raise FabricationSpecError(
                    "iso_metric_thread designation must resemble 'M3x0.5' or 'M3x0.5-6H'"
                )
            declared_diameter = float(match.group("diameter")) / 1000.0
            declared_pitch = float(match.group("pitch")) / 1000.0
            if not math.isclose(diameter, declared_diameter, rel_tol=0.0, abs_tol=1e-12):
                raise FabricationSpecError(
                    "iso_metric_thread nominal_diameter_m disagrees with its designation"
                )
            if not math.isclose(pitch, declared_pitch, rel_tol=0.0, abs_tol=1e-12):
                raise FabricationSpecError(
                    "iso_metric_thread pitch_m disagrees with its designation"
                )
        if self.family == "awg" and self.gauge_awg is None:
            raise FabricationSpecError("awg standards require gauge_awg")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StandardReferenceSpec":
        data = _object(value, "standard")
        allowed = {
            "family",
            "authority",
            "document",
            "designation",
            "revision",
            "role",
            "uri",
            "nominal_diameter_m",
            "pitch_m",
            "gauge_awg",
        }
        required = {"family", "authority", "document", "designation"}
        _keys(data, allowed, required, "standard")
        gauge = data.get("gauge_awg")
        if gauge is not None and (isinstance(gauge, bool) or not isinstance(gauge, int)):
            raise FabricationSpecError("standard.gauge_awg must be an integer")
        return cls(
            _text(data["family"], "standard.family"),
            _text(data["authority"], "standard.authority"),
            _text(data["document"], "standard.document"),
            _text(data["designation"], "standard.designation"),
            _optional_text(data.get("revision"), "standard.revision"),
            _text(data.get("role", "neutral"), "standard.role"),
            None if data.get("uri") is None else _url(data["uri"], "standard.uri"),
            _optional_number(data.get("nominal_diameter_m"), "standard.nominal_diameter_m", positive=True),
            _optional_number(data.get("pitch_m"), "standard.pitch_m", positive=True),
            gauge,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "family": self.family,
            "authority": self.authority,
            "document": self.document,
            "designation": self.designation,
            "role": self.role,
        }
        for name in ("revision", "uri", "nominal_diameter_m", "pitch_m", "gauge_awg"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class FabricationEvidenceSpec:
    kind: str
    source: str
    locator: str | None = None
    sha256: str | None = None
    page: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"component_input", "datasheet", "vendor_page", "catalog", "drawing", "measurement", "manual"}:
            raise FabricationSpecError("fabrication evidence.kind is unsupported")
        _text(self.source, "fabrication evidence.source")
        if self.locator is not None:
            _text(self.locator, "fabrication evidence.locator")
        if self.sha256 is not None and _SHA256.fullmatch(self.sha256) is None:
            raise FabricationSpecError("fabrication evidence.sha256 must be a prefixed SHA-256")
        if self.page is not None and (
            isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1
        ):
            raise FabricationSpecError("fabrication evidence.page must be a positive integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FabricationEvidenceSpec":
        data = _object(value, "fabrication evidence")
        _keys(data, {"kind", "source", "locator", "sha256", "page"}, {"kind", "source"}, "fabrication evidence")
        return cls(
            _text(data["kind"], "fabrication evidence.kind"),
            _text(data["source"], "fabrication evidence.source"),
            _optional_text(data.get("locator"), "fabrication evidence.locator"),
            _optional_text(data.get("sha256"), "fabrication evidence.sha256"),
            data.get("page"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "source": self.source}
        for name in ("locator", "sha256", "page"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class RetentionSpec:
    method: str
    hardware: StandardReferenceSpec | None = None
    quantity: int | None = None
    torque_n_m: float | None = None
    locking_method: str | None = None
    installation_process: str | None = None

    _METHODS = frozenset(
        {"threaded_fastener", "threaded_engagement", "press_fit", "snap_fit", "clip", "pin", "key", "clamp", "adhesive", "weld", "solder", "crimp", "integral", "none", "other"}
    )
    _LOCKING = frozenset(
        {"none", "locknut", "prevailing_torque", "threadlocker", "lock_washer", "safety_wire", "staking", "adhesive", "other"}
    )

    def __post_init__(self) -> None:
        if self.method not in self._METHODS:
            raise FabricationSpecError(f"retention.method must be one of {sorted(self._METHODS)}")
        if self.hardware is not None and not isinstance(self.hardware, StandardReferenceSpec):
            raise FabricationSpecError("retention.hardware must be a standard reference")
        if self.quantity is not None:
            _positive_integer(self.quantity, "retention.quantity")
        torque = _optional_number(self.torque_n_m, "retention.torque_n_m", nonnegative=True)
        object.__setattr__(self, "torque_n_m", torque)
        if self.locking_method is not None and self.locking_method not in self._LOCKING:
            raise FabricationSpecError(f"retention.locking_method must be one of {sorted(self._LOCKING)}")
        if self.installation_process is not None:
            _text(self.installation_process, "retention.installation_process")

    @property
    def missing_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.method in {"threaded_fastener", "threaded_engagement"}:
            for name in ("hardware", "quantity", "torque_n_m", "locking_method"):
                if getattr(self, name) is None:
                    missing.append(name)
        elif self.method not in {"integral", "none"}:
            if self.installation_process is None:
                missing.append("installation_process")
            if self.method in {"clip", "pin", "key", "clamp", "snap_fit", "other"} and self.hardware is None:
                missing.append("hardware")
        return tuple(missing)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetentionSpec":
        data = _object(value, "retention")
        _keys(data, {"method", "hardware", "quantity", "torque_n_m", "locking_method", "installation_process"}, {"method"}, "retention")
        quantity = data.get("quantity")
        if quantity is not None:
            quantity = _positive_integer(quantity, "retention.quantity")
        return cls(
            _text(data["method"], "retention.method"),
            None if data.get("hardware") is None else StandardReferenceSpec.from_dict(_object(data["hardware"], "retention.hardware")),
            quantity,
            _optional_number(data.get("torque_n_m"), "retention.torque_n_m", nonnegative=True),
            _optional_text(data.get("locking_method"), "retention.locking_method"),
            _optional_text(data.get("installation_process"), "retention.installation_process"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"method": self.method}
        if self.hardware is not None:
            result["hardware"] = self.hardware.to_dict()
        for name in ("quantity", "torque_n_m", "locking_method", "installation_process"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class BearingSpec:
    method: str
    standard: StandardReferenceSpec | None = None
    designation: str | None = None
    bore_diameter_m: float | None = None
    outer_diameter_m: float | None = None
    width_m: float | None = None
    radial_clearance_m: float | None = None
    axial_retention: RetentionSpec | None = None
    lubrication: str | None = None

    _METHODS = frozenset({"rolling_element", "plain_bushing", "direct_running", "flexure"})

    def __post_init__(self) -> None:
        if self.method not in self._METHODS:
            raise FabricationSpecError(f"bearing.method must be one of {sorted(self._METHODS)}")
        if self.standard is not None and not isinstance(self.standard, StandardReferenceSpec):
            raise FabricationSpecError("bearing.standard must be a standard reference")
        if self.designation is not None:
            _text(self.designation, "bearing.designation")
        for name in ("bore_diameter_m", "outer_diameter_m", "width_m"):
            object.__setattr__(self, name, _optional_number(getattr(self, name), f"bearing.{name}", positive=True))
        object.__setattr__(self, "radial_clearance_m", _optional_number(self.radial_clearance_m, "bearing.radial_clearance_m", nonnegative=True))
        if self.axial_retention is not None and not isinstance(self.axial_retention, RetentionSpec):
            raise FabricationSpecError("bearing.axial_retention must be a retention record")
        if self.lubrication is not None:
            _text(self.lubrication, "bearing.lubrication")

    @property
    def missing_fields(self) -> tuple[str, ...]:
        if self.method == "flexure":
            return () if self.standard is not None or self.designation is not None else ("standard_or_designation",)
        missing = [
            name
            for name in ("standard", "designation", "bore_diameter_m", "outer_diameter_m", "width_m", "radial_clearance_m", "axial_retention", "lubrication")
            if getattr(self, name) is None
        ]
        if self.axial_retention is not None:
            missing.extend(f"axial_retention.{name}" for name in self.axial_retention.missing_fields)
        return tuple(missing)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BearingSpec":
        data = _object(value, "bearing")
        names = {"method", "standard", "designation", "bore_diameter_m", "outer_diameter_m", "width_m", "radial_clearance_m", "axial_retention", "lubrication"}
        _keys(data, names, {"method"}, "bearing")
        return cls(
            _text(data["method"], "bearing.method"),
            None if data.get("standard") is None else StandardReferenceSpec.from_dict(_object(data["standard"], "bearing.standard")),
            _optional_text(data.get("designation"), "bearing.designation"),
            _optional_number(data.get("bore_diameter_m"), "bearing.bore_diameter_m", positive=True),
            _optional_number(data.get("outer_diameter_m"), "bearing.outer_diameter_m", positive=True),
            _optional_number(data.get("width_m"), "bearing.width_m", positive=True),
            _optional_number(data.get("radial_clearance_m"), "bearing.radial_clearance_m", nonnegative=True),
            None if data.get("axial_retention") is None else RetentionSpec.from_dict(_object(data["axial_retention"], "bearing.axial_retention")),
            _optional_text(data.get("lubrication"), "bearing.lubrication"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"method": self.method}
        if self.standard is not None:
            result["standard"] = self.standard.to_dict()
        if self.axial_retention is not None:
            result["axial_retention"] = self.axial_retention.to_dict()
        for name in ("designation", "bore_diameter_m", "outer_diameter_m", "width_m", "radial_clearance_m", "lubrication"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class ConductorSpec:
    """Known conductor details.

    Individual values are optional so evidence-backed fragments can be retained
    without inventing the rest of a construction-ready wire specification.  A
    conductor record must still contain at least one known value; its absent
    values are exposed through :attr:`missing_fields` and therefore remain
    explicit on a partial connector fabrication record.
    """

    standard: StandardReferenceSpec | None = None
    conductor_count: int | None = None
    material: str | None = None
    cross_section_m2: float | None = None
    insulation_standard: StandardReferenceSpec | None = None
    voltage_rating_v: float | None = None
    temperature_rating_k: float | None = None

    def __post_init__(self) -> None:
        if all(
            getattr(self, name) is None
            for name in (
                "standard",
                "conductor_count",
                "material",
                "cross_section_m2",
                "insulation_standard",
                "voltage_rating_v",
                "temperature_rating_k",
            )
        ):
            raise FabricationSpecError("conductor must contain at least one known value")
        for name in ("standard", "insulation_standard"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, StandardReferenceSpec):
                raise FabricationSpecError(f"conductor.{name} must be a standard reference")
        if self.conductor_count is not None:
            _positive_integer(self.conductor_count, "conductor.conductor_count")
        if self.material is not None and self.material not in {"copper", "tinned_copper", "silver_plated_copper", "aluminum", "other"}:
            raise FabricationSpecError("conductor.material is unsupported")
        for name in ("cross_section_m2", "voltage_rating_v", "temperature_rating_k"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _number(value, f"conductor.{name}", positive=True))

    @property
    def missing_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "standard",
                "conductor_count",
                "material",
                "cross_section_m2",
                "insulation_standard",
                "voltage_rating_v",
                "temperature_rating_k",
            )
            if getattr(self, name) is None
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConductorSpec":
        data = _object(value, "conductor")
        names = {"standard", "conductor_count", "material", "cross_section_m2", "insulation_standard", "voltage_rating_v", "temperature_rating_k"}
        _keys(data, names, set(), "conductor")
        return cls(
            None if data.get("standard") is None else StandardReferenceSpec.from_dict(_object(data["standard"], "conductor.standard")),
            None if data.get("conductor_count") is None else _positive_integer(data["conductor_count"], "conductor.conductor_count"),
            _optional_text(data.get("material"), "conductor.material"),
            _optional_number(data.get("cross_section_m2"), "conductor.cross_section_m2", positive=True),
            None if data.get("insulation_standard") is None else StandardReferenceSpec.from_dict(_object(data["insulation_standard"], "conductor.insulation_standard")),
            _optional_number(data.get("voltage_rating_v"), "conductor.voltage_rating_v", positive=True),
            _optional_number(data.get("temperature_rating_k"), "conductor.temperature_rating_k", positive=True),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in (
            "standard",
            "conductor_count",
            "material",
            "cross_section_m2",
            "insulation_standard",
            "voltage_rating_v",
            "temperature_rating_k",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value.to_dict() if isinstance(value, StandardReferenceSpec) else value
        return result


@dataclass(frozen=True, slots=True)
class TerminationSpec:
    method: str
    hardware: StandardReferenceSpec | None
    installation_process: str
    contact_part_number: str | None = None
    housing_part_number: str | None = None
    pin: str | None = None
    contact_pitch_m: float | None = None
    pad_dimensions_m: tuple[float, float] | None = None

    _METHODS = frozenset({"crimp", "solder", "screw", "spring", "push_fit", "pcb_land", "weld", "insulation_displacement", "plug", "other"})

    def __post_init__(self) -> None:
        if self.method not in self._METHODS:
            raise FabricationSpecError(f"termination.method must be one of {sorted(self._METHODS)}")
        if self.hardware is not None and not isinstance(self.hardware, StandardReferenceSpec):
            raise FabricationSpecError("termination.hardware must be a standard reference")
        _text(self.installation_process, "termination.installation_process")
        for name in ("contact_part_number", "housing_part_number", "pin"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), f"termination.{name}")
        object.__setattr__(self, "contact_pitch_m", _optional_number(self.contact_pitch_m, "termination.contact_pitch_m", positive=True))
        if self.pad_dimensions_m is not None:
            if len(self.pad_dimensions_m) != 2:
                raise FabricationSpecError("termination.pad_dimensions_m must have two values")
            object.__setattr__(self, "pad_dimensions_m", tuple(_number(item, f"termination.pad_dimensions_m[{index}]", positive=True) for index, item in enumerate(self.pad_dimensions_m)))

    @property
    def missing_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.method in {"crimp", "plug", "insulation_displacement"}:
            for name in ("hardware", "contact_part_number", "housing_part_number", "pin"):
                if getattr(self, name) is None:
                    missing.append(name)
        elif self.method in {"screw", "spring", "push_fit", "other"} and self.hardware is None:
            missing.append("hardware")
        elif self.method == "pcb_land":
            for name in ("hardware", "contact_pitch_m", "pad_dimensions_m"):
                if getattr(self, name) is None:
                    missing.append(name)
        return tuple(missing)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TerminationSpec":
        data = _object(value, "termination")
        names = {"method", "hardware", "installation_process", "contact_part_number", "housing_part_number", "pin", "contact_pitch_m", "pad_dimensions_m"}
        _keys(data, names, {"method", "installation_process"}, "termination")
        dimensions = data.get("pad_dimensions_m")
        parsed_dimensions = None
        if dimensions is not None:
            sequence = _sequence(dimensions, "termination.pad_dimensions_m")
            if len(sequence) != 2:
                raise FabricationSpecError("termination.pad_dimensions_m must have two values")
            parsed_dimensions = tuple(_number(item, f"termination.pad_dimensions_m[{index}]", positive=True) for index, item in enumerate(sequence))
        return cls(
            _text(data["method"], "termination.method"),
            None if data.get("hardware") is None else StandardReferenceSpec.from_dict(_object(data["hardware"], "termination.hardware")),
            _text(data["installation_process"], "termination.installation_process"),
            _optional_text(data.get("contact_part_number"), "termination.contact_part_number"),
            _optional_text(data.get("housing_part_number"), "termination.housing_part_number"),
            _optional_text(data.get("pin"), "termination.pin"),
            _optional_number(data.get("contact_pitch_m"), "termination.contact_pitch_m", positive=True),
            parsed_dimensions,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"method": self.method, "installation_process": self.installation_process}
        if self.hardware is not None:
            result["hardware"] = self.hardware.to_dict()
        for name in ("contact_part_number", "housing_part_number", "pin", "contact_pitch_m"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.pad_dimensions_m is not None:
            result["pad_dimensions_m"] = list(self.pad_dimensions_m)
        return result


@dataclass(frozen=True, slots=True)
class ProtectionSpec:
    kind: str
    standard: StandardReferenceSpec | None = None
    part_number: str | None = None
    current_rating_a: float | None = None
    voltage_rating_v: float | None = None

    _KINDS = frozenset({"none", "fuse", "circuit_breaker", "current_limiter", "fusible_link", "other"})

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise FabricationSpecError(f"protection.kind must be one of {sorted(self._KINDS)}")
        if self.standard is not None and not isinstance(self.standard, StandardReferenceSpec):
            raise FabricationSpecError("protection.standard must be a standard reference")
        if self.part_number is not None:
            _text(self.part_number, "protection.part_number")
        object.__setattr__(self, "current_rating_a", _optional_number(self.current_rating_a, "protection.current_rating_a", positive=True))
        object.__setattr__(self, "voltage_rating_v", _optional_number(self.voltage_rating_v, "protection.voltage_rating_v", positive=True))

    @property
    def missing_fields(self) -> tuple[str, ...]:
        if self.kind == "none":
            return ()
        return tuple(name for name in ("standard", "part_number", "current_rating_a", "voltage_rating_v") if getattr(self, name) is None)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtectionSpec":
        data = _object(value, "protection")
        _keys(data, {"kind", "standard", "part_number", "current_rating_a", "voltage_rating_v"}, {"kind"}, "protection")
        return cls(
            _text(data["kind"], "protection.kind"),
            None if data.get("standard") is None else StandardReferenceSpec.from_dict(_object(data["standard"], "protection.standard")),
            _optional_text(data.get("part_number"), "protection.part_number"),
            _optional_number(data.get("current_rating_a"), "protection.current_rating_a", positive=True),
            _optional_number(data.get("voltage_rating_v"), "protection.voltage_rating_v", positive=True),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.standard is not None:
            result["standard"] = self.standard.to_dict()
        for name in ("part_number", "current_rating_a", "voltage_rating_v"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class RouteSpec:
    topology: str
    routed_length_m: float
    minimum_bend_radius_m: float
    service_loop_m: float
    strain_relief: str
    waypoints: tuple[str, ...] = ()

    _TOPOLOGIES = frozenset({"point_to_point", "daisy_chain", "star", "tree", "bus", "captive_lead"})
    _STRAIN_RELIEF = frozenset({"none", "clamp", "gland", "tie_mount", "overmold", "boot", "lacing", "other"})

    def __post_init__(self) -> None:
        if self.topology not in self._TOPOLOGIES:
            raise FabricationSpecError(f"route.topology must be one of {sorted(self._TOPOLOGIES)}")
        object.__setattr__(self, "routed_length_m", _number(self.routed_length_m, "route.routed_length_m", positive=True))
        object.__setattr__(self, "minimum_bend_radius_m", _number(self.minimum_bend_radius_m, "route.minimum_bend_radius_m", nonnegative=True))
        object.__setattr__(self, "service_loop_m", _number(self.service_loop_m, "route.service_loop_m", nonnegative=True))
        if self.strain_relief not in self._STRAIN_RELIEF:
            raise FabricationSpecError(f"route.strain_relief must be one of {sorted(self._STRAIN_RELIEF)}")
        if any(not isinstance(item, str) or not item.strip() for item in self.waypoints):
            raise FabricationSpecError("route.waypoints must contain nonempty strings")
        if len(set(self.waypoints)) != len(self.waypoints):
            raise FabricationSpecError("route.waypoints must be unique")
        object.__setattr__(self, "waypoints", tuple(self.waypoints))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RouteSpec":
        data = _object(value, "route")
        names = {"topology", "routed_length_m", "minimum_bend_radius_m", "service_loop_m", "strain_relief", "waypoints"}
        _keys(data, names, names - {"waypoints"}, "route")
        return cls(
            _text(data["topology"], "route.topology"),
            _number(data["routed_length_m"], "route.routed_length_m", positive=True),
            _number(data["minimum_bend_radius_m"], "route.minimum_bend_radius_m", nonnegative=True),
            _number(data["service_loop_m"], "route.service_loop_m", nonnegative=True),
            _text(data["strain_relief"], "route.strain_relief"),
            tuple(_text(item, f"route.waypoints[{index}]") for index, item in enumerate(_sequence(data.get("waypoints", []), "route.waypoints"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology,
            "routed_length_m": self.routed_length_m,
            "minimum_bend_radius_m": self.minimum_bend_radius_m,
            "service_loop_m": self.service_loop_m,
            "strain_relief": self.strain_relief,
            "waypoints": list(self.waypoints),
        }


@dataclass(frozen=True, slots=True)
class TravelSpec:
    kind: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"bounded", "continuous"}:
            raise FabricationSpecError("travel.kind must be bounded or continuous")
        if self.unit not in {"rad", "m"}:
            raise FabricationSpecError("travel.unit must be rad or m")
        minimum = _optional_number(self.minimum, "travel.minimum")
        maximum = _optional_number(self.maximum, "travel.maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        if self.kind == "bounded":
            if minimum is None or maximum is None or minimum >= maximum:
                raise FabricationSpecError("bounded travel requires minimum < maximum")
        elif minimum is not None or maximum is not None:
            raise FabricationSpecError("continuous travel cannot declare limits")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TravelSpec":
        data = _object(value, "travel")
        _keys(data, {"kind", "unit", "minimum", "maximum"}, {"kind", "unit"}, "travel")
        return cls(
            _text(data["kind"], "travel.kind"),
            _text(data["unit"], "travel.unit"),
            _optional_number(data.get("minimum"), "travel.minimum"),
            _optional_number(data.get("maximum"), "travel.maximum"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "unit": self.unit}
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        return result


@dataclass(frozen=True, slots=True)
class ConnectorFabricationSpec:
    """One connector constraint or one selected assembly implementation."""

    kind: str
    status: str
    missing: tuple[str, ...]
    standards: tuple[StandardReferenceSpec, ...] = ()
    retention: RetentionSpec | None = None
    bearing: BearingSpec | None = None
    conductor: ConductorSpec | None = None
    termination: TerminationSpec | None = None
    protection: ProtectionSpec | None = None
    route: RouteSpec | None = None
    travel: TravelSpec | None = None
    alignment_tolerance_m: float | None = None
    alignment_tolerance_rad: float | None = None
    evidence: tuple[FabricationEvidenceSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _FABRICATION_KINDS:
            raise FabricationSpecError(f"fabrication.kind must be one of {sorted(_FABRICATION_KINDS)}")
        if self.status not in _STATUSES:
            raise FabricationSpecError(f"fabrication.status must be one of {sorted(_STATUSES)}")
        if any(not isinstance(item, str) or not item.strip() for item in self.missing):
            raise FabricationSpecError("fabrication.missing must contain nonempty field paths")
        if len(set(self.missing)) != len(self.missing):
            raise FabricationSpecError("fabrication.missing field paths must be unique")
        if any(not isinstance(item, StandardReferenceSpec) for item in self.standards):
            raise FabricationSpecError("fabrication.standards must contain standard references")
        if any(not isinstance(item, FabricationEvidenceSpec) for item in self.evidence):
            raise FabricationSpecError("fabrication.evidence must contain evidence references")
        object.__setattr__(self, "missing", tuple(self.missing))
        object.__setattr__(self, "standards", tuple(self.standards))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "alignment_tolerance_m", _optional_number(self.alignment_tolerance_m, "fabrication.alignment_tolerance_m", nonnegative=True))
        object.__setattr__(self, "alignment_tolerance_rad", _optional_number(self.alignment_tolerance_rad, "fabrication.alignment_tolerance_rad", nonnegative=True))
        mechanical_payload = any(item is not None for item in (self.retention, self.bearing, self.travel))
        electrical_payload = any(item is not None for item in (self.conductor, self.termination, self.protection, self.route))
        if self.kind in {"fixed_mount", "rotary_support"} and electrical_payload:
            raise FabricationSpecError(f"{self.kind} cannot contain electrical fabrication fields")
        if self.kind == "electrical_termination" and mechanical_payload:
            raise FabricationSpecError("electrical_termination cannot contain mechanical fabrication fields")
        if self.kind == "optical_alignment" and (mechanical_payload or electrical_payload):
            raise FabricationSpecError("optical_alignment accepts only standards and alignment tolerances")
        payload_present = bool(self.standards or mechanical_payload or electrical_payload or self.alignment_tolerance_m is not None or self.alignment_tolerance_rad is not None)
        computed = self.connector_missing_fields()
        if self.status == "missing":
            if payload_present:
                raise FabricationSpecError("status missing cannot carry claimed fabrication values")
            if not self.missing:
                raise FabricationSpecError("status missing requires explicit missing field paths")
            omitted = sorted(set(computed) - set(self.missing))
            if omitted:
                raise FabricationSpecError(
                    "status missing must identify every absent required field: "
                    + ", ".join(omitted)
                )
        elif self.status == "partial":
            if not payload_present or not self.missing or not self.evidence:
                raise FabricationSpecError("status partial requires evidence, known values, and explicit missing fields")
            omitted = sorted(set(computed) - set(self.missing))
            if omitted:
                raise FabricationSpecError(
                    "status partial must identify every absent required field: "
                    + ", ".join(omitted)
                )
        else:
            if self.missing:
                raise FabricationSpecError("status specified cannot declare missing fields")
            if computed:
                raise FabricationSpecError(
                    "status specified is incomplete: " + ", ".join(computed)
                )
            if not self.evidence:
                raise FabricationSpecError("status specified requires evidence")

    def connector_missing_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.kind == "fixed_mount":
            if self.retention is None:
                missing.append("retention")
            else:
                missing.extend(f"retention.{name}" for name in self.retention.missing_fields)
        elif self.kind == "rotary_support":
            if self.retention is None:
                missing.append("retention")
            else:
                missing.extend(f"retention.{name}" for name in self.retention.missing_fields)
            if self.bearing is None:
                missing.append("bearing")
            else:
                missing.extend(f"bearing.{name}" for name in self.bearing.missing_fields)
            if self.travel is None:
                missing.append("travel")
        elif self.kind == "electrical_termination":
            if self.conductor is None:
                missing.append("conductor")
            else:
                missing.extend(f"conductor.{name}" for name in self.conductor.missing_fields)
            if self.termination is None:
                missing.append("termination")
            else:
                missing.extend(f"termination.{name}" for name in self.termination.missing_fields)
        elif self.kind == "optical_alignment":
            if not self.standards:
                missing.append("standards")
            if self.alignment_tolerance_m is None:
                missing.append("alignment_tolerance_m")
            if self.alignment_tolerance_rad is None:
                missing.append("alignment_tolerance_rad")
        elif not self.standards:
            missing.append("standards")
        return tuple(missing)

    def implementation_missing_fields(self, *, endpoint_count: int = 2) -> tuple[str, ...]:
        missing = list(self.connector_missing_fields())
        if self.kind == "electrical_termination":
            if self.protection is None:
                missing.append("protection")
            else:
                missing.extend(f"protection.{name}" for name in self.protection.missing_fields)
            if self.route is None:
                missing.append("route")
            elif endpoint_count > 2 and self.route.topology == "point_to_point":
                missing.append("route.topology_for_multi_endpoint_net")
        return tuple(dict.fromkeys((*missing, *self.missing)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConnectorFabricationSpec":
        data = _object(value, "fabrication")
        names = {
            "kind", "status", "missing", "standards", "retention", "bearing",
            "conductor", "termination", "protection", "route", "travel",
            "alignment_tolerance_m", "alignment_tolerance_rad", "evidence",
        }
        _keys(data, names, {"kind", "status", "missing"}, "fabrication")
        return cls(
            _text(data["kind"], "fabrication.kind"),
            _text(data["status"], "fabrication.status"),
            tuple(_text(item, f"fabrication.missing[{index}]") for index, item in enumerate(_sequence(data["missing"], "fabrication.missing"))),
            tuple(StandardReferenceSpec.from_dict(_object(item, f"fabrication.standards[{index}]")) for index, item in enumerate(_sequence(data.get("standards", []), "fabrication.standards"))),
            None if data.get("retention") is None else RetentionSpec.from_dict(_object(data["retention"], "fabrication.retention")),
            None if data.get("bearing") is None else BearingSpec.from_dict(_object(data["bearing"], "fabrication.bearing")),
            None if data.get("conductor") is None else ConductorSpec.from_dict(_object(data["conductor"], "fabrication.conductor")),
            None if data.get("termination") is None else TerminationSpec.from_dict(_object(data["termination"], "fabrication.termination")),
            None if data.get("protection") is None else ProtectionSpec.from_dict(_object(data["protection"], "fabrication.protection")),
            None if data.get("route") is None else RouteSpec.from_dict(_object(data["route"], "fabrication.route")),
            None if data.get("travel") is None else TravelSpec.from_dict(_object(data["travel"], "fabrication.travel")),
            _optional_number(data.get("alignment_tolerance_m"), "fabrication.alignment_tolerance_m", nonnegative=True),
            _optional_number(data.get("alignment_tolerance_rad"), "fabrication.alignment_tolerance_rad", nonnegative=True),
            tuple(FabricationEvidenceSpec.from_dict(_object(item, f"fabrication.evidence[{index}]")) for index, item in enumerate(_sequence(data.get("evidence", []), "fabrication.evidence"))),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "status": self.status,
            "missing": list(self.missing),
        }
        if self.standards:
            result["standards"] = [item.to_dict() for item in self.standards]
        for name in ("retention", "bearing", "conductor", "termination", "protection", "route", "travel"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value.to_dict()
        for name in ("alignment_tolerance_m", "alignment_tolerance_rad"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.evidence:
            result["evidence"] = [item.to_dict() for item in self.evidence]
        return result


def fabrication_compatibility_issues(
    left: ConnectorFabricationSpec | None,
    right: ConnectorFabricationSpec | None,
) -> tuple[str, ...]:
    """Return deterministic conflicts; missing facts are not conflicts."""

    if left is None or right is None or left.status == "missing" or right.status == "missing":
        return ()
    issues: list[str] = []
    if left.kind != right.kind:
        issues.append(f"kind differs ({left.kind!r} versus {right.kind!r})")
        return tuple(issues)

    def all_standards(value: ConnectorFabricationSpec) -> tuple[StandardReferenceSpec, ...]:
        result = list(value.standards)
        if value.retention is not None and value.retention.hardware is not None:
            result.append(value.retention.hardware)
        if value.bearing is not None:
            if value.bearing.standard is not None:
                result.append(value.bearing.standard)
            if (
                value.bearing.axial_retention is not None
                and value.bearing.axial_retention.hardware is not None
            ):
                result.append(value.bearing.axial_retention.hardware)
        if value.conductor is not None:
            result.extend(
                standard
                for standard in (
                    value.conductor.standard,
                    value.conductor.insulation_standard,
                )
                if standard is not None
            )
        if value.termination is not None and value.termination.hardware is not None:
            result.append(value.termination.hardware)
        if value.protection is not None and value.protection.standard is not None:
            result.append(value.protection.standard)
        return tuple(result)

    left_by_family: dict[str, list[StandardReferenceSpec]] = {}
    right_by_family: dict[str, list[StandardReferenceSpec]] = {}
    for item in all_standards(left):
        left_by_family.setdefault(item.family, []).append(item)
    for item in all_standards(right):
        right_by_family.setdefault(item.family, []).append(item)
    for family in sorted(set(left_by_family) & set(right_by_family)):
        for first in left_by_family[family]:
            for second in right_by_family[family]:
                for name in ("nominal_diameter_m", "pitch_m", "gauge_awg"):
                    a, b = getattr(first, name), getattr(second, name)
                    if a is not None and b is not None and a != b:
                        issues.append(
                            f"{family} {name} differs ({a!r} versus {b!r})"
                        )
                role_pair = {first.role, second.role}
                if (
                    first.role != "neutral"
                    and second.role != "neutral"
                    and role_pair
                    not in ({"internal", "external"}, {"plug", "receptacle"})
                ):
                    issues.append(
                        f"{family} roles are not complementary "
                        f"({first.role!r}, {second.role!r})"
                    )
    return tuple(dict.fromkeys(issues))


def missing_fabrication(kind: str, fields: Sequence[str]) -> ConnectorFabricationSpec:
    """Create an explicit evidence-free missing record without inventing values."""

    return ConnectorFabricationSpec(kind, "missing", tuple(fields))


__all__ = [
    "BearingSpec",
    "COMMON_STANDARD_FAMILIES",
    "ConductorSpec",
    "ConnectorFabricationSpec",
    "FabricationEvidenceSpec",
    "FabricationSpecError",
    "ProtectionSpec",
    "RetentionSpec",
    "RouteSpec",
    "StandardReferenceSpec",
    "TerminationSpec",
    "TravelSpec",
    "fabrication_compatibility_issues",
    "missing_fabrication",
]
