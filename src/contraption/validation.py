"""Machine-checkable structural, dimensional, and composition validation.

Validation deliberately distinguishes errors (unsafe to compile) from warnings
(an evidence or numerical-quality gap).  A valid report proves schema and
dimension consistency; it is not a claim that a model has been empirically
certified.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dsl import Call, Conditional, DSLParseError, Expression, ExpressionType, parse_expression
from .specs import (
    ComponentInstanceSpec, ContraptionSpec, ModelSpec, PortRef, SpecError,
)
from .units import DIMENSIONLESS, TIME, Dimension, UnitError, parse_unit


@dataclass(frozen=True, slots=True, order=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning", "info"}:
            raise ValueError(f"invalid issue severity {self.severity!r}")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def is_valid(self) -> bool:
        return self.valid

    def require_valid(self) -> "ValidationReport":
        if self.errors:
            details = "\n".join(f"- [{item.code}] {item.path}: {item.message}" for item in self.errors)
            raise ModelValidationError(f"validation failed with {len(self.errors)} error(s):\n{details}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [
                {"severity": issue.severity, "code": issue.code, "path": issue.path, "message": issue.message}
                for issue in self.issues
            ],
        }


class ModelValidationError(SpecError):
    """Raised by ``require_valid`` when strict validation fails."""


class _Issues:
    def __init__(self) -> None:
        self.values: list[ValidationIssue] = []

    def add(self, severity: str, code: str, path: str, message: str) -> None:
        self.values.append(ValidationIssue(severity, code, path, message))

    def error(self, code: str, path: str, message: str) -> None:
        self.add("error", code, path, message)

    def warning(self, code: str, path: str, message: str) -> None:
        self.add("warning", code, path, message)

    def report(self) -> ValidationReport:
        return ValidationReport(tuple(sorted(self.values, key=lambda item: (item.path, item.severity, item.code, item.message))))


def _duplicates(names: Iterable[str]) -> tuple[str, ...]:
    counts = Counter(names)
    return tuple(sorted(name for name, count in counts.items() if count > 1))


def _unit_dimension(unit: str, path: str, issues: _Issues) -> Dimension | None:
    try:
        return parse_unit(unit).dimension
    except UnitError as exc:
        issues.error("unit.invalid", path, str(exc))
        return None


def model_symbol_table(model: ModelSpec) -> dict[str, ExpressionType]:
    """Build the strongly typed scalar symbol table for a model."""

    symbols: dict[str, ExpressionType] = {"t": ExpressionType("real", TIME)}
    for state in model.states:
        dimension = parse_unit(state.unit).dimension
        symbols[state.name] = ExpressionType("real", dimension)
        symbols[state.derivative or f"{state.name}_dot"] = ExpressionType("real", dimension / TIME)
    for variable in model.algebraics:
        symbols[variable.name] = ExpressionType("real", parse_unit(variable.unit).dimension)
    for parameter in model.parameters:
        symbols[parameter.name] = ExpressionType("real", parse_unit(parameter.unit).dimension)
    for port in model.power_ports:
        symbols[port.effort] = ExpressionType("real", parse_unit(port.effort_unit).dimension)
        symbols[port.flow] = ExpressionType("real", parse_unit(port.flow_unit).dimension)
    for port in model.signal_ports:
        symbols[port.name] = ExpressionType("real", parse_unit(port.unit).dimension) if port.dtype != "bool" else ExpressionType("boolean")
    return symbols


def _typed(expression: str, symbols: Mapping[str, ExpressionType], path: str, issues: _Issues, expected: str | None = None) -> Expression | None:
    try:
        parsed = parse_expression(expression)
        result = parsed.infer_type(symbols)
        if expected is not None and result.kind != expected:
            issues.error("expression.type", path, f"expected {expected}, got {result.kind}")
        return parsed
    except (DSLParseError, UnitError) as exc:
        issues.error("expression.invalid", path, str(exc))
        return None


def _walk(expression: Expression) -> Iterable[Expression]:
    yield expression
    for attribute in ("operand", "left", "right", "condition", "when_true", "when_false"):
        child = getattr(expression, attribute, None)
        if isinstance(child, Expression):
            yield from _walk(child)
    for argument in getattr(expression, "arguments", ()):
        yield from _walk(argument)


def validate_model(model: ModelSpec, taxonomy: Any = None) -> ValidationReport:
    """Validate a parsed model without executing model-authored code."""

    issues = _Issues()
    named_groups = {
        "states": [item.name for item in model.states], "algebraics": [item.name for item in model.algebraics],
        "parameters": [item.name for item in model.parameters], "power_ports": [item.name for item in model.power_ports],
        "signal_ports": [item.name for item in model.signal_ports], "relations": [item.name for item in model.relations],
        "stored_energy": [item.name for item in model.stored_energy], "dissipation": [item.name for item in model.dissipation],
        "sources": [item.name for item in model.sources], "modes": [item.name for item in model.modes],
        "fidelity_levels": [item.name for item in model.fidelity_levels], "properties": [item.name for item in model.properties],
    }
    for group, names in named_groups.items():
        for duplicate in _duplicates(names):
            issues.error("name.duplicate", f"model.{group}", f"duplicate name {duplicate!r}")

    scalar_names = (
        named_groups["states"] + named_groups["algebraics"] + named_groups["parameters"] +
        [port.effort for port in model.power_ports] + [port.flow for port in model.power_ports] + named_groups["signal_ports"]
    )
    for duplicate in _duplicates(scalar_names):
        issues.error("symbol.duplicate", "model", f"scalar symbol {duplicate!r} is declared more than once")

    for index, state in enumerate(model.states):
        _unit_dimension(state.unit, f"model.states[{index}].unit", issues)
    for index, variable in enumerate(model.algebraics):
        _unit_dimension(variable.unit, f"model.algebraics[{index}].unit", issues)
    for index, parameter in enumerate(model.parameters):
        _unit_dimension(parameter.unit, f"model.parameters[{index}].unit", issues)
        uncertainty = parameter.uncertainty
        values = uncertainty.parameters
        if uncertainty.distribution in {"normal", "lognormal"}:
            standard_deviation = values.get("std", values.get("sigma"))
            if standard_deviation is None or not isinstance(standard_deviation, (int, float)) or standard_deviation <= 0:
                issues.error("uncertainty.parameters", f"model.parameters[{index}].uncertainty", "normal distributions require positive std")
        if uncertainty.distribution == "uniform":
            lower, upper = values.get("lower"), values.get("upper")
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)) or lower >= upper:
                issues.error("uncertainty.parameters", f"model.parameters[{index}].uncertainty", "uniform distributions require lower < upper")

    power_dimension = parse_unit("W").dimension
    for index, port in enumerate(model.power_ports):
        effort = _unit_dimension(port.effort_unit, f"model.power_ports[{index}].effort_unit", issues)
        flow = _unit_dimension(port.flow_unit, f"model.power_ports[{index}].flow_unit", issues)
        if effort is not None and flow is not None and effort * flow != power_dimension:
            issues.error("port.not_power_conjugate", f"model.power_ports[{index}]", "effort × flow must have units of power")
    for index, port in enumerate(model.signal_ports):
        _unit_dimension(port.unit, f"model.signal_ports[{index}].unit", issues)

    try:
        symbols = model_symbol_table(model)
    except UnitError:
        symbols = {}

    parsed_relations: list[Expression] = []
    derivative_symbols = {state.derivative or f"{state.name}_dot" for state in model.states}
    seen_derivatives: set[str] = set()
    nonsmooth = {"abs", "min", "max", "clip", "sign", "where"}
    for index, relation in enumerate(model.relations):
        path = f"model.relations[{index}].expression"
        expression = _typed(relation.expression, symbols, path, issues, "real")
        if expression is not None:
            parsed_relations.append(expression)
            seen_derivatives.update(expression.variables() & derivative_symbols)
            used_nonsmooth = sorted({node.function for node in _walk(expression) if isinstance(node, Call) and node.function in nonsmooth})
            if used_nonsmooth:
                issues.error("differentiability.nonsmooth", path, f"residual uses nonsmooth primitive(s): {', '.join(used_nonsmooth)}; use modes or smooth_abs")
            if any(isinstance(node, Conditional) for node in _walk(expression)):
                issues.error("differentiability.conditional", path, "residual conditionals must be represented as discrete modes")
    for derivative in sorted(derivative_symbols - seen_derivatives):
        issues.error("state.unconstrained_derivative", "model.relations", f"state derivative {derivative!r} does not occur in a residual")

    def validate_named(collection: Any, collection_name: str, *, must_nonnegative: bool = False) -> None:
        for index, item in enumerate(collection):
            path = f"model.{collection_name}[{index}]"
            expression = _typed(item.expression, symbols, f"{path}.expression", issues, "real")
            declared = _unit_dimension(item.unit, f"{path}.unit", issues)
            if expression is not None and declared is not None:
                try:
                    inferred = expression.infer_type(symbols).dimension
                    if inferred != declared:
                        issues.error("expression.unit", path, f"expression is {inferred.describe()}, declared unit is {declared.describe()}")
                except DSLParseError:
                    pass
            if must_nonnegative and not item.nonnegative:
                issues.error("dissipation.nonnegative", path, "dissipation must be declared nonnegative and backed by a property test")

    validate_named(model.stored_energy, "stored_energy")
    validate_named(model.dissipation, "dissipation", must_nonnegative=True)
    validate_named(model.sources, "sources")

    relation_names = set(named_groups["relations"])
    mode_names = set(named_groups["modes"])
    if model.modes and sum(mode.initial for mode in model.modes) != 1:
        issues.error("mode.initial", "model.modes", "a model with modes must declare exactly one initial mode")
    for mode_index, mode in enumerate(model.modes):
        for name in mode.active_relations:
            if name not in relation_names:
                issues.error("mode.relation", f"model.modes[{mode_index}]", f"unknown active relation {name!r}")
        for transition_index, transition in enumerate(mode.transitions):
            path = f"model.modes[{mode_index}].transitions[{transition_index}]"
            if transition.target not in mode_names:
                issues.error("mode.target", path, f"unknown target mode {transition.target!r}")
            _typed(transition.guard, symbols, f"{path}.guard", issues, "boolean")
            for target, expression in transition.resets.items():
                if target not in named_groups["states"]:
                    issues.error("mode.reset", path, f"reset target {target!r} is not a state")
                _typed(expression, symbols, f"{path}.resets.{target}", issues, "real")

    for index, constraint in enumerate(model.initialization.constraints):
        _typed(constraint.expression, symbols, f"model.initialization.constraints[{index}].expression", issues, "real")
    known_initial = set(named_groups["states"] + named_groups["algebraics"] + named_groups["parameters"])
    for name in model.initialization.required:
        if name not in known_initial:
            issues.error("initialization.unknown", "model.initialization.required", f"unknown symbol {name!r}")
    for name in model.validity.ranges:
        if name not in symbols:
            issues.error("validity.unknown", f"model.validity.ranges.{name}", "validity range references an unknown symbol")

    for index, fidelity in enumerate(model.fidelity_levels):
        for relation in fidelity.active_relations:
            if relation not in relation_names:
                issues.error("fidelity.relation", f"model.fidelity_levels[{index}]", f"unknown active relation {relation!r}")
        for parameter in fidelity.parameter_overrides:
            if parameter not in named_groups["parameters"]:
                issues.error("fidelity.parameter", f"model.fidelity_levels[{index}]", f"unknown parameter override {parameter!r}")
    for index, property_spec in enumerate(model.properties):
        _typed(property_spec.expression, symbols, f"model.properties[{index}].expression", issues, "boolean")

    if model.dissipation and not any(item.kind == "nonnegative_dissipation" for item in model.properties):
        issues.error("property.dissipation_missing", "model.properties", "a dissipative model requires a nonnegative_dissipation property")
    if not model.validity.assumptions and not model.validity.ranges:
        issues.warning("validity.empty", "model.validity", "model has no explicit validity envelope")
    if all(getattr(model.trust, aspect) == "unverified" for aspect in ("structural", "physical", "numerical", "empirical")):
        issues.warning("trust.unverified", "model.trust", "all trust dimensions are unverified")
    if taxonomy is not None:
        try:
            for item in taxonomy.validate_model_contract(model):
                issues.add(item.severity, item.code, item.path, item.message)
        except (AttributeError, SpecError) as exc:
            issues.error("taxonomy.contract", "model", str(exc))
    return issues.report()


def _component_model(component: ComponentInstanceSpec, registry: Mapping[str, ModelSpec], issues: _Issues) -> ModelSpec | None:
    try:
        return registry[component.model]
    except KeyError:
        issues.error("component.model_missing", f"components.{component.id}.model", f"model {component.model!r} is not registered")
        return None


def _validate_contraption_structure(spec: ContraptionSpec, issues: _Issues) -> dict[str, ComponentInstanceSpec]:
    """Populate checks that do not require component model definitions."""

    components = {component.id: component for component in spec.components}
    for duplicate in _duplicates(component.id for component in spec.components):
        issues.error("component.duplicate", "contraption.components", f"duplicate component id {duplicate!r}")
    for duplicate in _duplicates(connection.id for connection in spec.connections):
        issues.error("connection.duplicate", "contraption.connections", f"duplicate connection id {duplicate!r}")
    for duplicate in _duplicates(control.id for control in spec.controls):
        issues.error("control.duplicate", "contraption.controls", f"duplicate control id {duplicate!r}")

    for index, connection in enumerate(spec.connections):
        path = f"contraption.connections[{index}]"
        if len(set((endpoint.component, endpoint.port) for endpoint in connection.endpoints)) != len(connection.endpoints):
            issues.error("connection.self_duplicate", path, "an endpoint may occur only once in a connection")
        for endpoint_index, endpoint in enumerate(connection.endpoints):
            if endpoint.component not in components:
                issues.error(
                    "reference.component",
                    f"{path}.endpoints[{endpoint_index}]",
                    f"unknown component {endpoint.component!r}",
                )

    for index, control in enumerate(spec.controls):
        if control.target.component not in components:
            issues.error(
                "reference.component",
                f"contraption.controls[{index}].target",
                f"unknown component {control.target.component!r}",
            )
        if control.external and not control.source.startswith("external"):
            issues.warning("control.external_name", f"contraption.controls[{index}].source", "external controls should use an 'external...' source namespace")
    return components


def validate_contraption_structure(spec: ContraptionSpec) -> ValidationReport:
    """Validate IDs and component references without validating models or ports.

    This deliberately limited entry point is for manifests whose complete PMDL
    registry is unavailable.  A successful result does *not* establish model,
    parameter, port, domain, or unit compatibility.
    """

    issues = _Issues()
    _validate_contraption_structure(spec, issues)
    return issues.report()


def validate_contraption(spec: ContraptionSpec, registry: Mapping[str, ModelSpec] | None = None) -> ValidationReport:
    """Fully validate references, models, ports, parameters, units, and controls.

    Full validation requires an explicit model registry.  Call
    :func:`validate_contraption_structure` only when a manifest intentionally
    lacks the component PMDLs and the reduced validation scope is acceptable.
    """

    issues = _Issues()
    components = _validate_contraption_structure(spec, issues)
    if registry is None:
        issues.error(
            "registry.required",
            "contraption.components",
            "full contraption validation requires a model registry; use "
            "validate_contraption_structure() only for structural/component-reference checks",
        )
        return issues.report()

    models: dict[str, ModelSpec] = {}
    for component in spec.components:
        model = _component_model(component, registry, issues)
        if model is None:
            continue
        models[component.id] = model
        parameter_specs = {parameter.name: parameter for parameter in model.parameters}
        for name, value in component.parameters.items():
            if name not in parameter_specs:
                issues.error("component.parameter_unknown", f"components.{component.id}.parameters.{name}", "parameter is not declared by the model")
                continue
            scalar = value.get("value") if isinstance(value, Mapping) else value
            if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
                issues.error("component.parameter_type", f"components.{component.id}.parameters.{name}", "override must be numeric or an object containing numeric value")
            elif not parameter_specs[name].bounds.contains(float(scalar)):
                issues.error("component.parameter_bounds", f"components.{component.id}.parameters.{name}", "override is outside model bounds")

    def resolve(reference: PortRef, kind: str, path: str) -> Any:
        if reference.component not in components:
            return None
        model = models.get(reference.component)
        if model is None:
            return None
        ports = model.power_ports if kind == "power" else model.signal_ports
        matches = [port for port in ports if port.name == reference.port]
        if not matches:
            issues.error("reference.port", path, f"component {reference.component!r} has no {kind} port {reference.port!r}")
            return None
        return matches[0]

    for index, connection in enumerate(spec.connections):
        path = f"contraption.connections[{index}]"
        if connection.kind == "power":
            ports = [resolve(endpoint, "power", f"{path}.endpoints[{i}]") for i, endpoint in enumerate(connection.endpoints)]
            ports = [port for port in ports if port is not None]
            domains = {port.domain for port in ports}
            if len(domains) > 1:
                issues.error("connection.domain", path, f"power ports have incompatible domains: {', '.join(sorted(domains))}")
            if connection.domain is not None and domains and connection.domain not in domains:
                issues.error("connection.domain_declared", path, "declared connection domain does not match endpoint ports")
            if ports:
                effort_dimensions = {parse_unit(port.effort_unit).dimension for port in ports}
                flow_dimensions = {parse_unit(port.flow_unit).dimension for port in ports}
                if len(effort_dimensions) > 1 or len(flow_dimensions) > 1:
                    issues.error("connection.units", path, "connected power ports are not effort/flow compatible")
        elif connection.kind == "signal":
            ports = [resolve(endpoint, "signal", f"{path}.endpoints[{i}]") for i, endpoint in enumerate(connection.endpoints)]
            ports = [port for port in ports if port is not None]
            if ports and len(ports) == len(connection.endpoints) and sum(port.direction == "output" for port in ports) != 1:
                issues.error("connection.signal_direction", path, "a signal connection requires exactly one output source")
            dimensions = {parse_unit(port.unit).dimension for port in ports}
            if len(dimensions) > 1:
                issues.error("connection.signal_units", path, "connected signal ports have incompatible units")
        elif connection.kind == "attachment":
            ports = [resolve(endpoint, "power", f"{path}.endpoints[{i}]") for i, endpoint in enumerate(connection.endpoints)]
            ports = [port for port in ports if port is not None]
            domains = {port.domain for port in ports}
            if any(domain != "mechanical" for domain in domains):
                issues.error(
                    "connection.attachment_domain",
                    path,
                    "attachment endpoints must be mechanical power ports; found domains: "
                    + ", ".join(sorted(domains)),
                )
            if connection.domain is not None and connection.domain not in {"mechanical", "rigid_mechanical"}:
                issues.error(
                    "connection.attachment_domain_declared",
                    path,
                    "an attachment connection domain must be 'mechanical' or 'rigid_mechanical'",
                )
            if ports:
                effort_dimensions = {parse_unit(port.effort_unit).dimension for port in ports}
                flow_dimensions = {parse_unit(port.flow_unit).dimension for port in ports}
                if len(effort_dimensions) > 1 or len(flow_dimensions) > 1:
                    issues.error(
                        "connection.attachment_units",
                        path,
                        "attachment endpoint mechanical effort/flow units are incompatible",
                    )

    for index, control in enumerate(spec.controls):
        port = resolve(control.target, "signal", f"contraption.controls[{index}].target")
        if port is not None and port.direction != "input":
            issues.error("control.direction", f"contraption.controls[{index}].target", "control target must be an input signal port")
    return issues.report()


def validate_model_file(path: str | Path, taxonomy: Any = None) -> ValidationReport:
    from .dsl import load_model
    try:
        return validate_model(load_model(path), taxonomy)
    except (SpecError, OSError) as exc:
        return ValidationReport((ValidationIssue("error", "model.parse", str(path), str(exc)),))


def assert_valid_model(model: ModelSpec, taxonomy: Any = None) -> ModelSpec:
    validate_model(model, taxonomy).require_valid()
    return model


def assert_valid_contraption(spec: ContraptionSpec, registry: Mapping[str, ModelSpec] | None = None) -> ContraptionSpec:
    validate_contraption(spec, registry).require_valid()
    return spec
