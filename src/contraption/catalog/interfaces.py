"""Abstract PMDL interfaces discovered from the physical model catalog.

The filesystem is authoritative.  Every top-level catalog directory is a
physical domain, every second-level directory is a category, and an optional
third level is a concrete device type.  Each layer declares its contract in an
``interface.pmdl`` document; there is no parallel JSON taxonomy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable

from ..physics.specs import (
    FrozenDict,
    ModelSpec,
    SpecError,
    StrictRecord,
    _boolean,
    _freeze,
    _identifier,
    _keys,
    _object,
    _sequence,
    _string,
    _strings,
)
from ..physics.units import UnitError, parse_unit


INTERFACE_FORMAT = "pmdl-interface-1"


def _semantic_key(value: str) -> str:
    """Normalize interface ids/names for case/punctuation-safe collisions."""

    return "-".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _interface_header(data: Mapping[str, Any], kind: str) -> None:
    if data.get("format") != INTERFACE_FORMAT:
        raise SpecError(
            f"interface.format must be {INTERFACE_FORMAT!r}, got {data.get('format')!r}"
        )
    if data.get("kind") != kind:
        raise SpecError(f"interface.kind must be {kind!r}, got {data.get('kind')!r}")
    if data.get("abstract") is not True:
        raise SpecError("PMDL interface declarations must set abstract: true")


@dataclass(frozen=True, slots=True)
class PowerPortInterface(StrictRecord):
    name: str
    domain: str
    effort_unit: str
    flow_unit: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PowerPortInterface":
        data = _object(data, "power_port_interface")
        names = ("name", "domain", "effort_unit", "flow_unit")
        _keys(data, names, "power_port_interface", names)
        return cls(
            _identifier(data["name"], "power_port_interface.name", symbol=True),
            _identifier(data["domain"], "power_port_interface.domain"),
            _string(data["effort_unit"], "power_port_interface.effort_unit"),
            _string(data["flow_unit"], "power_port_interface.flow_unit"),
        )


@dataclass(frozen=True, slots=True)
class SignalPortInterface(StrictRecord):
    name: str
    direction: str
    unit: str = "1"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalPortInterface":
        data = _object(data, "signal_port_interface")
        _keys(
            data,
            ("name", "direction", "unit"),
            "signal_port_interface",
            ("name", "direction"),
        )
        direction = _string(data["direction"], "signal_port_interface.direction")
        if direction not in {"input", "output"}:
            raise SpecError("signal port interface direction must be input or output")
        return cls(
            _identifier(data["name"], "signal_port_interface.name", symbol=True),
            direction,
            _string(data.get("unit", "1"), "signal_port_interface.unit"),
        )


@dataclass(frozen=True, slots=True)
class DomainInterface(StrictRecord):
    format: str
    kind: str
    abstract: bool
    id: str
    name: str
    version: str
    requires_physics: tuple[str, ...]
    allowed_port_domains: tuple[str, ...]
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainInterface":
        data = _object(data, "domain interface")
        names = (
            "format",
            "kind",
            "abstract",
            "id",
            "name",
            "version",
            "requires_physics",
            "allowed_port_domains",
            "description",
        )
        _keys(data, names, "domain interface", names[:8])
        _interface_header(data, "domain")
        return cls(
            INTERFACE_FORMAT,
            "domain",
            _boolean(data["abstract"], "domain_interface.abstract"),
            _identifier(data["id"], "domain_interface.id"),
            _string(data["name"], "domain_interface.name"),
            _string(data["version"], "domain_interface.version"),
            tuple(
                _identifier(value, "domain_interface.requires_physics[]")
                for value in _sequence(
                    data["requires_physics"], "domain_interface.requires_physics"
                )
            ),
            tuple(
                _identifier(value, "domain_interface.allowed_port_domains[]")
                for value in _sequence(
                    data["allowed_port_domains"],
                    "domain_interface.allowed_port_domains",
                )
            ),
            _string(data.get("description", ""), "domain_interface.description"),
        )


@dataclass(frozen=True, slots=True)
class CategoryInterface(StrictRecord):
    format: str
    kind: str
    abstract: bool
    id: str
    name: str
    version: str
    domains: tuple[str, ...]
    required_power_ports: tuple[PowerPortInterface, ...] = ()
    required_signal_ports: tuple[SignalPortInterface, ...] = ()
    ideal_models: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CategoryInterface":
        data = _object(data, "category interface")
        names = (
            "format",
            "kind",
            "abstract",
            "id",
            "name",
            "version",
            "domains",
            "required_power_ports",
            "required_signal_ports",
            "ideal_models",
            "constraints",
            "description",
        )
        _keys(data, names, "category interface", names[:7])
        _interface_header(data, "category")
        return cls(
            INTERFACE_FORMAT,
            "category",
            _boolean(data["abstract"], "category_interface.abstract"),
            _identifier(data["id"], "category_interface.id"),
            _string(data["name"], "category_interface.name"),
            _string(data["version"], "category_interface.version"),
            tuple(
                _identifier(value, "category_interface.domains[]")
                for value in _sequence(data["domains"], "category_interface.domains")
            ),
            tuple(
                PowerPortInterface.from_dict(value)
                for value in _sequence(
                    data.get("required_power_ports", []),
                    "category_interface.required_power_ports",
                )
            ),
            tuple(
                SignalPortInterface.from_dict(value)
                for value in _sequence(
                    data.get("required_signal_ports", []),
                    "category_interface.required_signal_ports",
                )
            ),
            tuple(
                _identifier(value, "category_interface.ideal_models[]")
                for value in _sequence(
                    data.get("ideal_models", []), "category_interface.ideal_models"
                )
            ),
            _strings(data.get("constraints", []), "category_interface.constraints"),
            _string(data.get("description", ""), "category_interface.description"),
        )


@dataclass(frozen=True, slots=True)
class DeviceInterface(StrictRecord):
    format: str
    kind: str
    abstract: bool
    id: str
    name: str
    version: str
    parent: str
    model_specificity: str
    changes_contract: bool = False
    required_power_ports: tuple[PowerPortInterface, ...] = ()
    required_signal_ports: tuple[SignalPortInterface, ...] = ()
    models: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeviceInterface":
        data = _object(data, "device interface")
        names = (
            "format",
            "kind",
            "abstract",
            "id",
            "name",
            "version",
            "parent",
            "model_specificity",
            "changes_contract",
            "required_power_ports",
            "required_signal_ports",
            "models",
            "constraints",
            "description",
        )
        _keys(data, names, "device interface", names[:8])
        _interface_header(data, "device")
        specificity = _string(
            data["model_specificity"], "device_interface.model_specificity"
        )
        if len(specificity.strip()) < 12:
            raise SpecError(
                "device_interface.model_specificity must explain its more-specific physical model"
            )
        return cls(
            INTERFACE_FORMAT,
            "device",
            _boolean(data["abstract"], "device_interface.abstract"),
            _identifier(data["id"], "device_interface.id"),
            _string(data["name"], "device_interface.name"),
            _string(data["version"], "device_interface.version"),
            _identifier(data["parent"], "device_interface.parent"),
            specificity,
            _boolean(data.get("changes_contract", False), "device_interface.changes_contract"),
            tuple(
                PowerPortInterface.from_dict(value)
                for value in _sequence(
                    data.get("required_power_ports", []),
                    "device_interface.required_power_ports",
                )
            ),
            tuple(
                SignalPortInterface.from_dict(value)
                for value in _sequence(
                    data.get("required_signal_ports", []),
                    "device_interface.required_signal_ports",
                )
            ),
            tuple(
                _identifier(value, "device_interface.models[]")
                for value in _sequence(data.get("models", []), "device_interface.models")
            ),
            _strings(data.get("constraints", []), "device_interface.constraints"),
            _string(data.get("description", ""), "device_interface.description"),
        )


ModelInterface = DomainInterface | CategoryInterface | DeviceInterface


def _strict_json(source: str, label: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SpecError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(source, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"{label}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return _object(value, label)


def parse_interface(source: str | Mapping[str, Any], *, source_name: str = "<interface>") -> ModelInterface:
    data = _strict_json(source, source_name) if isinstance(source, str) else _object(source, source_name)
    kind = data.get("kind")
    if kind == "domain":
        return DomainInterface.from_dict(data)
    if kind == "category":
        return CategoryInterface.from_dict(data)
    if kind == "device":
        return DeviceInterface.from_dict(data)
    raise SpecError(f"{source_name}: interface.kind must be domain, category, or device")


def load_interface(path: str | Path) -> ModelInterface:
    source = Path(path)
    try:
        return parse_interface(source.read_text(encoding="utf-8"), source_name=str(source))
    except OSError as exc:
        raise SpecError(f"could not read PMDL interface {source}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ModelInterfaceCatalog:
    domains: tuple[DomainInterface, ...]
    categories: tuple[CategoryInterface, ...]
    devices: tuple[DeviceInterface, ...] = ()
    version: str = "1.0.0"
    source_paths: FrozenDict[str] = field(
        default_factory=FrozenDict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_paths", _freeze(self.source_paths, "catalog.source_paths"))
        self._require_consistent()

    @property
    def domain_map(self) -> dict[str, DomainInterface]:
        return {item.id: item for item in self.domains}

    @property
    def category_map(self) -> dict[str, CategoryInterface]:
        return {item.id: item for item in self.categories}

    @property
    def device_map(self) -> dict[str, DeviceInterface]:
        return {item.id: item for item in self.devices}

    @property
    def interface_map(self) -> dict[str, ModelInterface]:
        return {**self.domain_map, **self.category_map, **self.device_map}

    def _require_consistent(self) -> None:
        groups = {
            "domain": [item.id for item in self.domains],
            "category": [item.id for item in self.categories],
            "device": [item.id for item in self.devices],
        }
        for label, values in groups.items():
            duplicates = sorted(
                value for value, count in Counter(values).items() if count > 1
            )
            if duplicates:
                raise SpecError(f"duplicate {label} interface id(s): {', '.join(duplicates)}")
        overlap = (
            (set(groups["domain"]) & set(groups["category"]))
            | (set(groups["domain"]) & set(groups["device"]))
            | (set(groups["category"]) & set(groups["device"]))
        )
        if overlap:
            raise SpecError(
                "interface ids must be globally unique: " + ", ".join(sorted(overlap))
            )
        semantic_owners: dict[str, str] = {}
        for interface in (*self.categories, *self.devices):
            for identity in {interface.id, interface.name}:
                key = _semantic_key(identity)
                owner = semantic_owners.get(key)
                if owner is not None and owner != interface.id:
                    raise SpecError(
                        "category/device interfaces cannot define parallel semantic "
                        f"identities: {interface.id!r} collides with {owner!r} as {key!r}"
                    )
                semantic_owners[key] = interface.id
        domains = self.domain_map
        for domain in self.domains:
            unknown = set(domain.allowed_port_domains) - set(domains)
            if unknown:
                raise SpecError(
                    f"domain {domain.id!r} allows unknown port domains: {', '.join(sorted(unknown))}"
                )
        for category in self.categories:
            unknown = set(category.domains) - set(domains)
            if unknown:
                raise SpecError(
                    f"category {category.id!r} has unknown domains: {', '.join(sorted(unknown))}"
                )
        for device in self.devices:
            if device.parent not in self.category_map:
                raise SpecError(
                    f"device {device.id!r} has unknown category parent {device.parent!r}"
                )
            if (
                device.required_power_ports or device.required_signal_ports
            ) and not device.changes_contract:
                raise SpecError(
                    f"device {device.id!r} adds ports but changes_contract is false"
                )

    def ancestry(self, interface_id: str) -> tuple[str, ...]:
        if interface_id in self.category_map:
            return (interface_id,)
        try:
            device = self.device_map[interface_id]
        except KeyError as exc:
            raise KeyError(interface_id) from exc
        return (device.parent, device.id)

    def category_for(self, interface_id: str) -> CategoryInterface:
        return self.category_map[self.ancestry(interface_id)[0]]

    def port_contract(
        self, interface_id: str
    ) -> tuple[tuple[PowerPortInterface, ...], tuple[SignalPortInterface, ...]]:
        category = self.category_for(interface_id)
        power = list(category.required_power_ports)
        signal = list(category.required_signal_ports)
        if interface_id in self.device_map:
            device = self.device_map[interface_id]
            power.extend(device.required_power_ports)
            signal.extend(device.required_signal_ports)
        return tuple(power), tuple(signal)

    def validate_model_contract(self, model: ModelSpec) -> tuple[Any, ...]:
        from ..physics.validation import ValidationIssue

        issues: list[ValidationIssue] = []
        try:
            category = self.category_for(model.implements)
            required_power, required_signal = self.port_contract(model.implements)
        except KeyError:
            return (
                ValidationIssue(
                    "error",
                    "interface.missing",
                    "model.implements",
                    f"unknown model interface {model.implements!r}",
                ),
            )
        if not set(category.domains).issubset(model.domains):
            missing = sorted(set(category.domains) - set(model.domains))
            issues.append(
                ValidationIssue(
                    "error",
                    "interface.domain",
                    "model.domains",
                    f"missing category domain(s): {', '.join(missing)}",
                )
            )
        for domain_name in model.domains:
            domain = self.domain_map.get(domain_name)
            if domain is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "interface.domain_unknown",
                        "model.domains",
                        f"unknown physical domain {domain_name!r}",
                    )
                )
                continue
            missing_physics = set(domain.requires_physics) - set(model.domains)
            if missing_physics:
                issues.append(
                    ValidationIssue(
                        "error",
                        "interface.physics",
                        "model.domains",
                        f"domain {domain_name!r} requires physics {', '.join(sorted(missing_physics))}",
                    )
                )
        model_power = {port.name: port for port in model.power_ports}
        for contract in required_power:
            port = model_power.get(contract.name)
            path = f"model.power_ports.{contract.name}"
            if port is None:
                issues.append(
                    ValidationIssue("error", "interface.port_missing", path, "required power port is absent")
                )
                continue
            if port.domain != contract.domain:
                issues.append(
                    ValidationIssue(
                        "error",
                        "interface.port_domain",
                        path,
                        f"expected {contract.domain!r}, got {port.domain!r}",
                    )
                )
            try:
                if (
                    parse_unit(port.effort_unit).dimension
                    != parse_unit(contract.effort_unit).dimension
                    or parse_unit(port.flow_unit).dimension
                    != parse_unit(contract.flow_unit).dimension
                ):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "interface.port_units",
                            path,
                            "port effort/flow dimensions violate the interface",
                        )
                    )
            except UnitError as exc:
                issues.append(
                    ValidationIssue("error", "interface.port_units", path, str(exc))
                )
        model_signal = {port.name: port for port in model.signal_ports}
        for contract in required_signal:
            port = model_signal.get(contract.name)
            path = f"model.signal_ports.{contract.name}"
            if port is None:
                issues.append(
                    ValidationIssue("error", "interface.signal_missing", path, "required signal port is absent")
                )
            elif port.direction != contract.direction:
                issues.append(
                    ValidationIssue(
                        "error",
                        "interface.signal_direction",
                        path,
                        f"expected {contract.direction!r}, got {port.direction!r}",
                    )
                )
        return tuple(issues)

    def validate_model_path(
        self, model: ModelSpec, path: str | Path, root: str | Path
    ) -> None:
        catalog_root = Path(root).resolve()
        source = Path(path).resolve()
        try:
            relative = source.relative_to(catalog_root)
        except ValueError as exc:
            raise SpecError(f"model path {source} is outside catalog root {catalog_root}") from exc
        if source.name == "interface.pmdl" or "instantiations" in relative.parts:
            raise SpecError(f"{relative}: concrete models cannot be interfaces or instance files")
        if len(relative.parts) not in {3, 4}:
            raise SpecError(
                f"{relative}: concrete PMDL must live at category or device layer"
            )
        interface_directory = relative.parent.as_posix()
        expected = next(
            (
                interface_id
                for interface_id, interface_path in self.source_paths.items()
                if Path(interface_path).parent.as_posix() == interface_directory
            ),
            None,
        )
        if expected is None:
            raise SpecError(f"{relative}: parent directory has no interface.pmdl")
        if model.implements != expected:
            raise SpecError(
                f"{relative}: model implements {model.implements!r}, expected {expected!r} from its directory"
            )
        domain_id = relative.parts[0]
        if domain_id not in model.domains:
            raise SpecError(
                f"{relative}: top-level physical domain {domain_id!r} is absent from model.domains"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "pmdl-interface-catalog-1",
            "version": self.version,
            "domains": [item.to_dict() for item in self.domains],
            "categories": [item.to_dict() for item in self.categories],
            "devices": [item.to_dict() for item in self.devices],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInterfaceCatalog":
        data = _object(data, "interface catalog")
        names = ("format", "version", "domains", "categories", "devices")
        _keys(data, names, "interface catalog", names[:4])
        if data["format"] != "pmdl-interface-catalog-1":
            raise SpecError("interface catalog format must be 'pmdl-interface-catalog-1'")
        return cls(
            tuple(
                DomainInterface.from_dict(item)
                for item in _sequence(data["domains"], "interface_catalog.domains")
            ),
            tuple(
                CategoryInterface.from_dict(item)
                for item in _sequence(data["categories"], "interface_catalog.categories")
            ),
            tuple(
                DeviceInterface.from_dict(item)
                for item in _sequence(data.get("devices", []), "interface_catalog.devices")
            ),
            _string(data["version"], "interface_catalog.version"),
        )

    @classmethod
    def from_json(cls, source: str) -> "ModelInterfaceCatalog":
        return cls.from_dict(_strict_json(source, "interface catalog"))


def interface_paths(root: str | Path) -> tuple[Path, ...]:
    catalog_root = Path(root)
    if not catalog_root.is_dir():
        raise SpecError(f"model catalog directory does not exist: {catalog_root}")
    return tuple(sorted(catalog_root.rglob("interface.pmdl")))


def concrete_model_paths(root: str | Path) -> tuple[Path, ...]:
    catalog_root = Path(root)
    return tuple(
        path
        for path in sorted(catalog_root.rglob("*.pmdl"))
        if path.name != "interface.pmdl" and "instantiations" not in path.parts
    )


def load_interface_catalog(root: str | Path) -> ModelInterfaceCatalog:
    catalog_root = Path(root).resolve()
    paths = interface_paths(catalog_root)
    if not paths:
        raise SpecError(f"model catalog {catalog_root} contains no interface.pmdl files")
    domains: list[DomainInterface] = []
    categories: list[CategoryInterface] = []
    devices: list[DeviceInterface] = []
    source_paths: dict[str, str] = {}
    directory_ids: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(catalog_root)
        depth = len(relative.parts)
        expected_kind = {2: "domain", 3: "category", 4: "device"}.get(depth)
        if expected_kind is None or "instantiations" in relative.parts:
            raise SpecError(
                f"{relative}: interface.pmdl must be at domain, category, or device layer"
            )
        interface = load_interface(path)
        if interface.kind != expected_kind:
            raise SpecError(
                f"{relative}: filesystem layer requires kind {expected_kind!r}, got {interface.kind!r}"
            )
        if interface.id in source_paths:
            raise SpecError(f"duplicate interface id {interface.id!r}")
        source_paths[interface.id] = relative.as_posix()
        directory_ids[relative.parent.as_posix()] = interface.id
        if isinstance(interface, DomainInterface):
            if relative.parts[0] != interface.id:
                raise SpecError(
                    f"{relative}: domain directory must equal interface id {interface.id!r}"
                )
            domains.append(interface)
        elif isinstance(interface, CategoryInterface):
            categories.append(interface)
        else:
            devices.append(interface)

    catalog = ModelInterfaceCatalog(
        tuple(domains),
        tuple(categories),
        tuple(devices),
        source_paths=FrozenDict(source_paths),
    )
    for category in catalog.categories:
        relative = Path(source_paths[category.id])
        domain_id = relative.parts[0]
        if domain_id not in category.domains:
            raise SpecError(
                f"{relative}: containing domain {domain_id!r} is absent from category.domains"
            )
    for device in catalog.devices:
        relative = Path(source_paths[device.id])
        parent_directory = relative.parent.parent.as_posix()
        actual_parent = directory_ids.get(parent_directory)
        if actual_parent != device.parent:
            raise SpecError(
                f"{relative}: device parent {device.parent!r} does not match directory category {actual_parent!r}"
            )
    auxiliary_directories = {catalog_root / "procurement"}
    top_level_directories = sorted(
        path
        for path in catalog_root.iterdir()
        if path.is_dir() and path not in auxiliary_directories
    )
    declared_domain_directories = {catalog_root / item.id for item in catalog.domains}
    undeclared = [path.name for path in top_level_directories if path not in declared_domain_directories]
    if undeclared:
        raise SpecError(
            "every top-level model_catalog directory must be a physical domain; missing interface.pmdl for: "
            + ", ".join(undeclared)
        )
    return catalog


def default_interface_root() -> Path:
    from ..paths import asset_root

    return asset_root() / "model_catalog"


def load_default_interface_catalog() -> ModelInterfaceCatalog:
    return load_interface_catalog(default_interface_root())


__all__ = [
    "CategoryInterface",
    "DeviceInterface",
    "DomainInterface",
    "INTERFACE_FORMAT",
    "ModelInterfaceCatalog",
    "PowerPortInterface",
    "SignalPortInterface",
    "concrete_model_paths",
    "default_interface_root",
    "interface_paths",
    "load_default_interface_catalog",
    "load_interface",
    "load_interface_catalog",
    "parse_interface",
]
