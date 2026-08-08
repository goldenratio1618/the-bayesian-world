"""Strict component taxonomy and physical-instantiation contracts.

Subcategories are allowed only when they document a materially more specific
model.  Whether a node changes its port contract is explicit, preventing a
classifier from silently treating incompatible parts as synonyms.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .specs import (
    FrozenDict, GeometrySpec, ModelSpec, SpecError, StrictRecord, _boolean, _freeze,
    _identifier, _keys, _object, _sequence, _string, _strings,
)
from .units import UnitError, parse_unit


@dataclass(frozen=True, slots=True)
class DomainContract(StrictRecord):
    id: str
    name: str
    requires_physics: tuple[str, ...]
    allowed_port_domains: tuple[str, ...]
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainContract":
        data = _object(data, "domain")
        names = ("id", "name", "requires_physics", "allowed_port_domains", "description")
        _keys(data, names, "domain", names[:4])
        return cls(_identifier(data["id"], "domain.id"), _string(data["name"], "domain.name"),
                   tuple(_identifier(value, "domain.requires_physics[]") for value in _sequence(data["requires_physics"], "domain.requires_physics")),
                   tuple(_identifier(value, "domain.allowed_port_domains[]") for value in _sequence(data["allowed_port_domains"], "domain.allowed_port_domains")),
                   _string(data.get("description", ""), "domain.description"))


@dataclass(frozen=True, slots=True)
class PowerPortContract(StrictRecord):
    name: str
    domain: str
    effort_unit: str
    flow_unit: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PowerPortContract":
        data = _object(data, "power_port_contract")
        _keys(data, ("name", "domain", "effort_unit", "flow_unit"), "power_port_contract", ("name", "domain", "effort_unit", "flow_unit"))
        return cls(_identifier(data["name"], "power_port_contract.name", symbol=True), _identifier(data["domain"], "power_port_contract.domain"),
                   _string(data["effort_unit"], "power_port_contract.effort_unit"), _string(data["flow_unit"], "power_port_contract.flow_unit"))


@dataclass(frozen=True, slots=True)
class SignalPortContract(StrictRecord):
    name: str
    direction: str
    unit: str = "1"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalPortContract":
        data = _object(data, "signal_port_contract")
        _keys(data, ("name", "direction", "unit"), "signal_port_contract", ("name", "direction"))
        direction = _string(data["direction"], "signal_port_contract.direction")
        if direction not in {"input", "output"}:
            raise SpecError("signal port contract direction must be input or output")
        return cls(_identifier(data["name"], "signal_port_contract.name", symbol=True), direction,
                   _string(data.get("unit", "1"), "signal_port_contract.unit"))


@dataclass(frozen=True, slots=True)
class CategoryContract(StrictRecord):
    id: str
    name: str
    domains: tuple[str, ...]
    required_power_ports: tuple[PowerPortContract, ...] = ()
    required_signal_ports: tuple[SignalPortContract, ...] = ()
    ideal_models: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CategoryContract":
        data = _object(data, "category")
        names = ("id", "name", "domains", "required_power_ports", "required_signal_ports", "ideal_models", "constraints", "description")
        _keys(data, names, "category", names[:3])
        return cls(_identifier(data["id"], "category.id"), _string(data["name"], "category.name"),
                   tuple(_identifier(value, "category.domains[]") for value in _sequence(data["domains"], "category.domains")),
                   tuple(PowerPortContract.from_dict(value) for value in _sequence(data.get("required_power_ports", []), "category.required_power_ports")),
                   tuple(SignalPortContract.from_dict(value) for value in _sequence(data.get("required_signal_ports", []), "category.required_signal_ports")),
                   tuple(_identifier(value, "category.ideal_models[]") for value in _sequence(data.get("ideal_models", []), "category.ideal_models")),
                   _strings(data.get("constraints", []), "category.constraints"), _string(data.get("description", ""), "category.description"))


@dataclass(frozen=True, slots=True)
class SubcategoryContract(StrictRecord):
    id: str
    name: str
    parent: str
    model_specificity: str
    changes_contract: bool = False
    required_power_ports: tuple[PowerPortContract, ...] = ()
    required_signal_ports: tuple[SignalPortContract, ...] = ()
    models: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubcategoryContract":
        data = _object(data, "subcategory")
        names = ("id", "name", "parent", "model_specificity", "changes_contract", "required_power_ports", "required_signal_ports", "models", "constraints", "description")
        _keys(data, names, "subcategory", names[:4])
        specificity = _string(data["model_specificity"], "subcategory.model_specificity")
        if len(specificity.strip()) < 12:
            raise SpecError("subcategory.model_specificity must explain its more-specific physical model")
        return cls(_identifier(data["id"], "subcategory.id"), _string(data["name"], "subcategory.name"),
                   _identifier(data["parent"], "subcategory.parent"), specificity,
                   _boolean(data.get("changes_contract", False), "subcategory.changes_contract"),
                   tuple(PowerPortContract.from_dict(value) for value in _sequence(data.get("required_power_ports", []), "subcategory.required_power_ports")),
                   tuple(SignalPortContract.from_dict(value) for value in _sequence(data.get("required_signal_ports", []), "subcategory.required_signal_ports")),
                   tuple(_identifier(value, "subcategory.models[]") for value in _sequence(data.get("models", []), "subcategory.models")),
                   _strings(data.get("constraints", []), "subcategory.constraints"), _string(data.get("description", ""), "subcategory.description"))


@dataclass(frozen=True, slots=True)
class PurchasingSpec(StrictRecord):
    manufacturer: str | None = None
    part_number: str | None = None
    supplier: str | None = None
    url: str | None = None
    price: float | None = None
    currency: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PurchasingSpec":
        data = _object(data, "purchasing")
        names = ("manufacturer", "part_number", "supplier", "url", "price", "currency")
        _keys(data, names, "purchasing")
        price = data.get("price")
        if price is not None and (isinstance(price, bool) or not isinstance(price, (int, float)) or price < 0):
            raise SpecError("purchasing.price must be a nonnegative number")
        strings = [None if data.get(name) is None else _string(data[name], f"purchasing.{name}") for name in names[:4]]
        return cls(*strings, None if price is None else float(price),
                   None if data.get("currency") is None else _string(data["currency"], "purchasing.currency"))


@dataclass(frozen=True, slots=True)
class TaxonomyInstantiation(StrictRecord):
    id: str
    name: str
    taxonomy_node: str
    model: str
    version: str
    geometry: GeometrySpec
    condition: str = "unverified"
    parameters: FrozenDict[Any] = FrozenDict()
    parameter_uncertainty: FrozenDict[Any] = FrozenDict()
    purchasing: PurchasingSpec = PurchasingSpec()
    connection_info: FrozenDict[Any] = FrozenDict()
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        for name in ("id", "taxonomy_node", "model"):
            _identifier(getattr(self, name), f"instantiation.{name}")
        if self.condition not in {"unverified", "inspected", "calibrated", "degraded", "retired"}:
            raise SpecError(f"unsupported instantiation condition {self.condition!r}")
        for name in ("parameters", "parameter_uncertainty", "connection_info", "metadata"):
            object.__setattr__(self, name, _freeze(getattr(self, name), f"instantiation.{name}"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaxonomyInstantiation":
        data = _object(data, "instantiation")
        names = ("id", "name", "taxonomy_node", "model", "version", "condition", "parameters", "parameter_uncertainty", "geometry", "purchasing", "connection_info", "metadata")
        _keys(data, names, "instantiation", (*names[:5], "geometry"))
        return cls(
            id=_identifier(data["id"], "instantiation.id"),
            name=_string(data["name"], "instantiation.name"),
            taxonomy_node=_identifier(data["taxonomy_node"], "instantiation.taxonomy_node"),
            model=_identifier(data["model"], "instantiation.model"),
            version=_string(data["version"], "instantiation.version"),
            geometry=GeometrySpec.from_dict(data["geometry"]),
            condition=_string(data.get("condition", "unverified"), "instantiation.condition"),
            parameters=_freeze(_object(data.get("parameters", {}), "instantiation.parameters")),
            parameter_uncertainty=_freeze(_object(data.get("parameter_uncertainty", {}), "instantiation.parameter_uncertainty")),
            purchasing=PurchasingSpec.from_dict(data.get("purchasing", {})),
            connection_info=_freeze(_object(data.get("connection_info", {}), "instantiation.connection_info")),
            metadata=_freeze(_object(data.get("metadata", {}), "instantiation.metadata")),
        )


InstantiationSpec = TaxonomyInstantiation


@dataclass(frozen=True, slots=True)
class Taxonomy(StrictRecord):
    format: str
    version: str
    domains: tuple[DomainContract, ...]
    categories: tuple[CategoryContract, ...]
    subcategories: tuple[SubcategoryContract, ...] = ()
    instantiations: tuple[TaxonomyInstantiation, ...] = ()
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        if self.format != "taxonomy-1":
            raise SpecError(f"unsupported taxonomy format {self.format!r}")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "taxonomy.metadata"))
        self._require_consistent()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Taxonomy":
        data = _object(data, "taxonomy")
        names = ("format", "version", "domains", "categories", "subcategories", "instantiations", "metadata")
        _keys(data, names, "taxonomy", names[:4])
        return cls(_string(data["format"], "taxonomy.format"), _string(data["version"], "taxonomy.version"),
                   tuple(DomainContract.from_dict(item) for item in _sequence(data["domains"], "taxonomy.domains")),
                   tuple(CategoryContract.from_dict(item) for item in _sequence(data["categories"], "taxonomy.categories")),
                   tuple(SubcategoryContract.from_dict(item) for item in _sequence(data.get("subcategories", []), "taxonomy.subcategories")),
                   tuple(TaxonomyInstantiation.from_dict(item) for item in _sequence(data.get("instantiations", []), "taxonomy.instantiations")),
                   _freeze(_object(data.get("metadata", {}), "taxonomy.metadata")))

    @property
    def domain_map(self) -> dict[str, DomainContract]:
        return {item.id: item for item in self.domains}

    @property
    def category_map(self) -> dict[str, CategoryContract]:
        return {item.id: item for item in self.categories}

    @property
    def subcategory_map(self) -> dict[str, SubcategoryContract]:
        return {item.id: item for item in self.subcategories}

    @property
    def instantiation_map(self) -> dict[str, TaxonomyInstantiation]:
        return {item.id: item for item in self.instantiations}

    def _require_consistent(self) -> None:
        groups = {
            "domain": [item.id for item in self.domains], "category": [item.id for item in self.categories],
            "subcategory": [item.id for item in self.subcategories], "instantiation": [item.id for item in self.instantiations],
        }
        for label, values in groups.items():
            duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
            if duplicates:
                raise SpecError(f"duplicate taxonomy {label} id(s): {', '.join(duplicates)}")
        overlap = set(groups["category"]) & set(groups["subcategory"])
        if overlap:
            raise SpecError(f"category/subcategory ids must be globally unique: {', '.join(sorted(overlap))}")
        domains = self.domain_map
        for domain in self.domains:
            unknown = set(domain.allowed_port_domains) - set(domains)
            if unknown:
                raise SpecError(f"domain {domain.id!r} allows unknown port domains: {', '.join(sorted(unknown))}")
        for category in self.categories:
            unknown = set(category.domains) - set(domains)
            if unknown:
                raise SpecError(f"category {category.id!r} has unknown domains: {', '.join(sorted(unknown))}")
        nodes = set(groups["category"]) | set(groups["subcategory"])
        for node in self.subcategories:
            if node.parent not in nodes:
                raise SpecError(f"subcategory {node.id!r} has unknown parent {node.parent!r}")
            if (node.required_power_ports or node.required_signal_ports) and not node.changes_contract:
                raise SpecError(f"subcategory {node.id!r} adds ports but changes_contract is false")
            visited = {node.id}
            parent = node.parent
            while parent in self.subcategory_map:
                if parent in visited:
                    raise SpecError(f"taxonomy cycle involving {node.id!r}")
                visited.add(parent)
                parent = self.subcategory_map[parent].parent
        for instance in self.instantiations:
            if instance.taxonomy_node not in nodes:
                raise SpecError(f"instantiation {instance.id!r} references unknown taxonomy node {instance.taxonomy_node!r}")

    def ancestry(self, node_id: str) -> tuple[str, ...]:
        if node_id in self.category_map:
            return (node_id,)
        if node_id not in self.subcategory_map:
            raise KeyError(node_id)
        result = [node_id]
        parent = self.subcategory_map[node_id].parent
        while parent in self.subcategory_map:
            result.append(parent)
            parent = self.subcategory_map[parent].parent
        if parent not in self.category_map:
            raise SpecError(f"taxonomy path for {node_id!r} does not terminate at a category")
        result.append(parent)
        return tuple(reversed(result))

    def category_for(self, node_id: str) -> CategoryContract:
        return self.category_map[self.ancestry(node_id)[0]]

    def port_contract(self, node_id: str) -> tuple[tuple[PowerPortContract, ...], tuple[SignalPortContract, ...]]:
        ancestry = self.ancestry(node_id)
        category = self.category_map[ancestry[0]]
        power, signal = list(category.required_power_ports), list(category.required_signal_ports)
        for descendant in ancestry[1:]:
            node = self.subcategory_map[descendant]
            power.extend(node.required_power_ports)
            signal.extend(node.required_signal_ports)
        return tuple(power), tuple(signal)

    def validate_model_contract(self, model: ModelSpec) -> tuple[Any, ...]:
        from .validation import ValidationIssue
        issues: list[ValidationIssue] = []
        try:
            category = self.category_for(model.category)
            required_power, required_signal = self.port_contract(model.category)
        except KeyError:
            return (ValidationIssue("error", "taxonomy.category", "model.category", f"unknown taxonomy category {model.category!r}"),)
        if not set(category.domains).issubset(model.domains):
            missing = sorted(set(category.domains) - set(model.domains))
            issues.append(ValidationIssue("error", "taxonomy.domain", "model.domains", f"missing category domain(s): {', '.join(missing)}"))
        for domain_name in model.domains:
            domain = self.domain_map.get(domain_name)
            if domain is None:
                issues.append(ValidationIssue("error", "taxonomy.domain_unknown", "model.domains", f"unknown domain {domain_name!r}"))
                continue
            missing_physics = set(domain.requires_physics) - set(model.domains)
            if missing_physics:
                issues.append(ValidationIssue("error", "taxonomy.physics", "model.domains", f"domain {domain_name!r} requires physics {', '.join(sorted(missing_physics))}"))
        model_power = {port.name: port for port in model.power_ports}
        for contract in required_power:
            port = model_power.get(contract.name)
            path = f"model.power_ports.{contract.name}"
            if port is None:
                issues.append(ValidationIssue("error", "taxonomy.port_missing", path, "required power port is absent"))
                continue
            if port.domain != contract.domain:
                issues.append(ValidationIssue("error", "taxonomy.port_domain", path, f"expected {contract.domain!r}, got {port.domain!r}"))
            try:
                if parse_unit(port.effort_unit).dimension != parse_unit(contract.effort_unit).dimension or parse_unit(port.flow_unit).dimension != parse_unit(contract.flow_unit).dimension:
                    issues.append(ValidationIssue("error", "taxonomy.port_units", path, "port effort/flow dimensions violate the category contract"))
            except UnitError as exc:
                issues.append(ValidationIssue("error", "taxonomy.port_units", path, str(exc)))
        model_signal = {port.name: port for port in model.signal_ports}
        for contract in required_signal:
            port = model_signal.get(contract.name)
            path = f"model.signal_ports.{contract.name}"
            if port is None:
                issues.append(ValidationIssue("error", "taxonomy.signal_missing", path, "required signal port is absent"))
            elif port.direction != contract.direction:
                issues.append(ValidationIssue("error", "taxonomy.signal_direction", path, f"expected {contract.direction!r}, got {port.direction!r}"))
        return tuple(issues)

    def instantiate(
        self, *, id: str, name: str, taxonomy_node: str, model: str, geometry: GeometrySpec,
        version: str = "1.0.0",
        parameters: Mapping[str, Any] | None = None, parameter_uncertainty: Mapping[str, Any] | None = None,
        purchasing: PurchasingSpec | None = None,
        connection_info: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None,
    ) -> TaxonomyInstantiation:
        if taxonomy_node not in self.category_map and taxonomy_node not in self.subcategory_map:
            raise SpecError(f"unknown taxonomy node {taxonomy_node!r}")
        return TaxonomyInstantiation(
            id=id, name=name, taxonomy_node=taxonomy_node, model=model, version=version,
            condition="unverified", parameters=_freeze(parameters or {}), parameter_uncertainty=_freeze(parameter_uncertainty or {}),
            geometry=geometry, purchasing=purchasing or PurchasingSpec(),
            connection_info=_freeze(connection_info or {}), metadata=_freeze(metadata or {}),
        )


TaxonomySpec = Taxonomy


def load_taxonomy(path: str | Path) -> Taxonomy:
    taxonomy_path = Path(path)
    try:
        return Taxonomy.from_json(taxonomy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"could not read taxonomy {taxonomy_path}: {exc}") from exc


def default_taxonomy_path() -> Path:
    from .paths import asset_root

    return asset_root() / "data" / "taxonomy.json"


def load_default_taxonomy() -> Taxonomy:
    return load_taxonomy(default_taxonomy_path())
