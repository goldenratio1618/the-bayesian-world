"""Static physical parts and initialized PMDL model instances.

An instantiation directory contains one ``static.part`` plus one or more
``vN.model`` files.  Static files own model-invariant geometry, connectors,
provenance, purchasing data, and metadata.  Model files select an exact PMDL
class and initialize every one of its parameters.  Contraptions reference the
model-instance ID and never repeat those parameters.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from ..physics.physical import (
    BodySpec,
    ResolvedPartSpec,
    ResolvedPartRegistry,
    ModelReferenceSpec,
    PhysicalConnectorSpec,
    PhysicalParameterBindingSpec,
    PhysicalSpecError,
    ProvenanceSpec,
    _canonical_json_value,
    _duplicates,
    _identifier,
    _json_loads,
    _keys,
    _mapping,
    _number,
    _sequence,
    _text,
)
from ..physics.specs import FrozenDict, ModelSpec


_VARIANT_FILE = re.compile(r"^v[1-9][0-9]*\.model$")
_CONDITIONS = frozenset(
    {"unverified", "inspected", "calibrated", "degraded", "retired"}
)


def _freeze_json(value: Any, context: str) -> Any:
    canonical = _canonical_json_value(value, context)
    if isinstance(canonical, dict):
        return FrozenDict(
            (key, _freeze_json(item, f"{context}.{key}"))
            for key, item in canonical.items()
        )
    if isinstance(canonical, list):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(canonical)
        )
    return canonical


@dataclass(frozen=True, slots=True)
class StaticPartSpec:
    format: str
    id: str
    name: str
    version: str
    physical_role: str
    bodies: tuple[BodySpec, ...]
    connectors: tuple[PhysicalConnectorSpec, ...]
    parameter_bindings: tuple[PhysicalParameterBindingSpec, ...]
    provenance: ProvenanceSpec
    purchasing: FrozenDict[Any] = FrozenDict()
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        if self.format != "static-part-1":
            raise PhysicalSpecError(
                f"static part format must be 'static-part-1', got {self.format!r}"
            )
        _identifier(self.id, "static_part.id")
        _text(self.name, "static_part.name")
        _text(self.version, "static_part.version")
        # Reuse the resolved part's mature physical invariants with a
        # deterministic placeholder model reference.  The real reference is
        # supplied only by a vN.model file.
        ResolvedPartSpec(
            "resolved-part-1",
            self.id,
            self.version,
            self.physical_role,
            ModelReferenceSpec("catalog.placeholder", "0", "sha256:" + "0" * 64),
            tuple(self.bodies),
            tuple(self.connectors),
            tuple(self.parameter_bindings),
            self.provenance,
        )
        object.__setattr__(self, "bodies", tuple(self.bodies))
        object.__setattr__(self, "connectors", tuple(self.connectors))
        object.__setattr__(self, "parameter_bindings", tuple(self.parameter_bindings))
        object.__setattr__(self, "purchasing", _freeze_json(self.purchasing, "static_part.purchasing"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "static_part.metadata"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StaticPartSpec":
        data = _mapping(value, "static part")
        names = (
            "format",
            "id",
            "name",
            "version",
            "physical_role",
            "bodies",
            "connectors",
            "parameter_bindings",
            "provenance",
            "purchasing",
            "metadata",
        )
        _keys(data, names, "static part", names[:9])
        return cls(
            _text(data["format"], "static_part.format"),
            _identifier(data["id"], "static_part.id"),
            _text(data["name"], "static_part.name"),
            _text(data["version"], "static_part.version"),
            _text(data["physical_role"], "static_part.physical_role"),
            tuple(
                BodySpec.from_dict(_mapping(item, f"static_part.bodies[{index}]"))
                for index, item in enumerate(_sequence(data["bodies"], "static_part.bodies"))
            ),
            tuple(
                PhysicalConnectorSpec.from_dict(
                    _mapping(item, f"static_part.connectors[{index}]")
                )
                for index, item in enumerate(
                    _sequence(data["connectors"], "static_part.connectors")
                )
            ),
            tuple(
                PhysicalParameterBindingSpec.from_dict(
                    _mapping(item, f"static_part.parameter_bindings[{index}]")
                )
                for index, item in enumerate(
                    _sequence(
                        data["parameter_bindings"], "static_part.parameter_bindings"
                    )
                )
            ),
            ProvenanceSpec.from_dict(_mapping(data["provenance"], "static_part.provenance")),
            _freeze_json(_mapping(data.get("purchasing", {}), "static_part.purchasing"), "static_part.purchasing"),
            _freeze_json(_mapping(data.get("metadata", {}), "static_part.metadata"), "static_part.metadata"),
        )

    @classmethod
    def from_json(cls, source: str) -> "StaticPartSpec":
        return cls.from_dict(_json_loads(source, "static part"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "physical_role": self.physical_role,
            "bodies": [item.to_dict() for item in self.bodies],
            "connectors": [item.to_dict() for item in self.connectors],
            "parameter_bindings": [item.to_dict() for item in self.parameter_bindings],
            "provenance": self.provenance.to_dict(),
            "purchasing": _canonical_json_value(self.purchasing),
            "metadata": _canonical_json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ComputeCostSpec:
    relative_cost: float
    notes: str = ""

    def __post_init__(self) -> None:
        value = _number(self.relative_cost, "model_instance.compute.relative_cost")
        if value <= 0:
            raise PhysicalSpecError("model_instance.compute.relative_cost must be positive")
        object.__setattr__(self, "relative_cost", value)
        if self.notes:
            _text(self.notes, "model_instance.compute.notes")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComputeCostSpec":
        data = _mapping(value, "model instance compute cost")
        _keys(data, ("relative_cost", "notes"), "model instance compute cost", ("relative_cost",))
        return cls(
            _number(data["relative_cost"], "model_instance.compute.relative_cost"),
            "" if data.get("notes") is None else _text(data["notes"], "model_instance.compute.notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"relative_cost": self.relative_cost, "notes": self.notes}


@dataclass(frozen=True, slots=True)
class ModelInstantiationSpec:
    format: str
    id: str
    variant: str
    part: str
    version: str
    model: ModelReferenceSpec
    parameters: FrozenDict[Any]
    parameter_uncertainty: FrozenDict[Any]
    condition: str
    compute: ComputeCostSpec
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        if self.format != "model-instance-1":
            raise PhysicalSpecError(
                f"model instance format must be 'model-instance-1', got {self.format!r}"
            )
        _identifier(self.id, "model_instance.id")
        _identifier(self.variant, "model_instance.variant")
        _identifier(self.part, "model_instance.part")
        _text(self.version, "model_instance.version")
        if self.condition not in _CONDITIONS:
            raise PhysicalSpecError(
                f"unsupported model instance condition {self.condition!r}"
            )
        if self.id != f"{self.part}.{self.variant}":
            raise PhysicalSpecError(
                "model_instance.id must equal '<part>.<variant>'; "
                f"expected {self.part}.{self.variant!s}, got {self.id!r}"
            )
        object.__setattr__(self, "parameters", _freeze_json(self.parameters, "model_instance.parameters"))
        object.__setattr__(
            self,
            "parameter_uncertainty",
            _freeze_json(self.parameter_uncertainty, "model_instance.parameter_uncertainty"),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "model_instance.metadata"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelInstantiationSpec":
        data = _mapping(value, "model instance")
        names = (
            "format",
            "id",
            "variant",
            "part",
            "version",
            "model",
            "parameters",
            "parameter_uncertainty",
            "condition",
            "compute",
            "metadata",
        )
        _keys(data, names, "model instance", names[:10])
        return cls(
            _text(data["format"], "model_instance.format"),
            _identifier(data["id"], "model_instance.id"),
            _identifier(data["variant"], "model_instance.variant"),
            _identifier(data["part"], "model_instance.part"),
            _text(data["version"], "model_instance.version"),
            ModelReferenceSpec.from_dict(_mapping(data["model"], "model_instance.model")),
            _freeze_json(_mapping(data["parameters"], "model_instance.parameters"), "model_instance.parameters"),
            _freeze_json(
                _mapping(data["parameter_uncertainty"], "model_instance.parameter_uncertainty"),
                "model_instance.parameter_uncertainty",
            ),
            _text(data["condition"], "model_instance.condition"),
            ComputeCostSpec.from_dict(_mapping(data["compute"], "model_instance.compute")),
            _freeze_json(_mapping(data.get("metadata", {}), "model_instance.metadata"), "model_instance.metadata"),
        )

    @classmethod
    def from_json(cls, source: str) -> "ModelInstantiationSpec":
        return cls.from_dict(_json_loads(source, "model instance"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "id": self.id,
            "variant": self.variant,
            "part": self.part,
            "version": self.version,
            "model": self.model.to_dict(),
            "parameters": _canonical_json_value(self.parameters),
            "parameter_uncertainty": _canonical_json_value(self.parameter_uncertainty),
            "condition": self.condition,
            "compute": self.compute.to_dict(),
            "metadata": _canonical_json_value(self.metadata),
        }

    def initialized_parameters(self) -> FrozenDict[Any]:
        values: dict[str, Any] = {}
        for name, raw in self.parameters.items():
            uncertainty = self.parameter_uncertainty.get(name)
            if uncertainty is None:
                values[name] = raw
            elif isinstance(raw, Mapping):
                combined = dict(raw)
                combined["uncertainty"] = _canonical_json_value(uncertainty)
                values[name] = combined
            else:
                values[name] = {
                    "value": raw,
                    "uncertainty": _canonical_json_value(uncertainty),
                }
        return _freeze_json(values, "initialized_parameters")


@dataclass(frozen=True, slots=True)
class PartInstantiation:
    static: StaticPartSpec
    model_instance: ModelInstantiationSpec
    directory: Path

    def __post_init__(self) -> None:
        if self.model_instance.part != self.static.id:
            raise PhysicalSpecError(
                f"model instance {self.model_instance.id!r} references part "
                f"{self.model_instance.part!r}, but static.part declares {self.static.id!r}"
            )

    @property
    def id(self) -> str:
        return self.model_instance.id

    @property
    def parameters(self) -> FrozenDict[Any]:
        return self.model_instance.initialized_parameters()

    def resolved_part(self) -> ResolvedPartSpec:
        return ResolvedPartSpec(
            "resolved-part-1",
            self.id,
            self.model_instance.version,
            self.static.physical_role,
            self.model_instance.model,
            self.static.bodies,
            self.static.connectors,
            self.static.parameter_bindings,
            self.static.provenance,
        )


class PartInstantiationRegistry(Mapping[str, PartInstantiation]):
    def __init__(self, values: Iterable[PartInstantiation] = ()) -> None:
        items: dict[str, PartInstantiation] = {}
        for value in values:
            if value.id in items:
                raise PhysicalSpecError(f"duplicate model instance id {value.id!r}")
            items[value.id] = value
        self._items = FrozenDict(items)

    def __getitem__(self, key: str) -> PartInstantiation:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def resolved_parts(self) -> ResolvedPartRegistry:
        return ResolvedPartRegistry(item.resolved_part() for item in self.values())

    def validate_models(self, models: Mapping[str, ModelSpec]) -> None:
        for item in self.values():
            reference = item.model_instance.model
            try:
                model = models[reference.id]
            except KeyError as exc:
                raise PhysicalSpecError(
                    f"model instance {item.id!r} references missing PMDL {reference.id!r}"
                ) from exc
            digest = "sha256:" + hashlib.sha256(model.to_json().encode("utf-8")).hexdigest()
            if model.version != reference.version or digest != reference.sha256:
                raise PhysicalSpecError(
                    f"model instance {item.id!r} PMDL identity/hash mismatch: "
                    f"expected {reference.id}@{reference.version} {reference.sha256}, "
                    f"got {model.id}@{model.version} {digest}"
                )
            declared = {parameter.name: parameter for parameter in model.parameters}
            supplied = set(item.model_instance.parameters)
            missing = sorted(set(declared) - supplied)
            unknown = sorted(supplied - set(declared))
            if missing or unknown:
                raise PhysicalSpecError(
                    f"model instance {item.id!r} must initialize every PMDL parameter; "
                    f"missing={missing}, unknown={unknown}"
                )
            unknown_uncertainty = sorted(
                set(item.model_instance.parameter_uncertainty) - set(declared)
            )
            if unknown_uncertainty:
                raise PhysicalSpecError(
                    f"model instance {item.id!r} has uncertainty for unknown parameters: "
                    + ", ".join(unknown_uncertainty)
                )
            for name, raw in item.model_instance.parameters.items():
                scalar = raw.get("value") if isinstance(raw, Mapping) else raw
                value = _number(scalar, f"model_instance {item.id}.{name}")
                if not declared[name].bounds.contains(value):
                    raise PhysicalSpecError(
                        f"model instance {item.id!r} parameter {name!r}={value} is outside "
                        f"PMDL bounds [{declared[name].bounds.lower}, {declared[name].bounds.upper}]"
                    )

    @classmethod
    def load_catalog(
        cls, root: str | Path, *, models: Mapping[str, ModelSpec] | None = None
    ) -> "PartInstantiationRegistry":
        catalog_root = Path(root).resolve()
        static_paths = sorted(catalog_root.rglob("static.part"))
        if not static_paths:
            raise PhysicalSpecError(
                f"model catalog {catalog_root} contains no static.part instantiations"
            )
        values: list[PartInstantiation] = []
        for static_path in static_paths:
            directory = static_path.parent
            if directory.parent.name != "instantiations":
                raise PhysicalSpecError(
                    f"{static_path.relative_to(catalog_root)} must be immediately below an instantiations directory"
                )
            contract_directory = directory.parent.parent
            try:
                contract_relative = contract_directory.relative_to(catalog_root)
            except ValueError as exc:
                raise PhysicalSpecError(
                    f"{static_path}: instantiation contract directory escapes the catalog"
                ) from exc
            if len(contract_relative.parts) not in {2, 3}:
                raise PhysicalSpecError(
                    f"{static_path.relative_to(catalog_root)} must belong to a category or device layer"
                )
            contract_path = contract_directory / "interface.pmdl"
            if contract_path.is_symlink() or not contract_path.is_file():
                raise PhysicalSpecError(
                    f"{static_path.relative_to(catalog_root)} has no category/device interface.pmdl at "
                    f"{contract_relative.as_posix()}"
                )
            static = StaticPartSpec.from_json(static_path.read_text(encoding="utf-8"))
            from .interfaces import load_interface

            contract = load_interface(contract_path)
            model_paths = sorted(directory.glob("*.model"))
            if not model_paths:
                raise PhysicalSpecError(f"{directory} must contain at least v1.model")
            for model_path in model_paths:
                if _VARIANT_FILE.fullmatch(model_path.name) is None:
                    raise PhysicalSpecError(
                        f"unsupported model-instance filename {model_path.name!r}; expected vN.model"
                    )
                model_instance = ModelInstantiationSpec.from_json(
                    model_path.read_text(encoding="utf-8")
                )
                if model_instance.variant != model_path.stem:
                    raise PhysicalSpecError(
                        f"{model_path}: variant {model_instance.variant!r} must match filename stem"
                    )
                if models is not None:
                    referenced_model = models.get(model_instance.model.id)
                    if referenced_model is not None and referenced_model.implements != contract.id:
                        raise PhysicalSpecError(
                            f"{model_path.relative_to(catalog_root)} references PMDL implementing "
                            f"{referenced_model.implements!r}, but its instantiation directory belongs "
                            f"to interface {contract.id!r}"
                        )
                values.append(PartInstantiation(static, model_instance, directory))
            extra = sorted(
                path.name
                for path in directory.iterdir()
                if path.is_file() and path.name != "static.part" and path.suffix != ".model"
            )
            if extra:
                raise PhysicalSpecError(
                    f"{directory}: unsupported instantiation files: {', '.join(extra)}"
                )
        registry = cls(values)
        if models is not None:
            registry.validate_models(models)
        return registry


__all__ = [
    "ComputeCostSpec",
    "ModelInstantiationSpec",
    "PartInstantiation",
    "PartInstantiationRegistry",
    "StaticPartSpec",
]
