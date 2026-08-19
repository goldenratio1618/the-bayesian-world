"""Deterministic, standalone Markdown documentation for validated catalog parts.

The renderer is deliberately downstream of the strict catalog parsers. It does
not interpret free text as physics and never changes an authoritative record.
Given the same validated catalog bytes it emits the same UTF-8 Markdown bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from io import StringIO
import json
from pathlib import Path
import tokenize
from typing import Any

from ..catalog.instantiations import PartInstantiation, PartInstantiationRegistry
from ..catalog.interfaces import (
    CategoryInterface,
    DeviceInterface,
    DomainInterface,
    ModelInterface,
    concrete_model_paths,
    load_interface,
    load_interface_catalog,
)
from ..physics.dsl import (
    Binary,
    Call,
    Comparison,
    Conditional,
    Expression,
    Literal,
    ModelRegistry,
    Symbol,
    Unary,
    load_model,
    parse_expression,
)
from ..physics.specs import ModelSpec


PART_README_FILENAME = "README.md"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(
        _plain(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _code(value: Any) -> str:
    text = str(value)
    fence = "`" if "`" not in text else "``"
    return f"{fence}{text}{fence}"


def _number(value: float | int | bool) -> str:
    if isinstance(value, bool):
        return r"\mathrm{true}" if value else r"\mathrm{false}"
    return format(float(value), ".17g")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\backslash ",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_symbol(name: str) -> str:
    if name in {"pi", "e"}:
        return r"\pi" if name == "pi" else "e"
    if "." in name:
        return r"\mathrm{" + _latex_escape(name) + "}"
    head, *tail = name.split("_")
    rendered = _latex_escape(head)
    if tail:
        rendered += r"_{\mathrm{" + _latex_escape("_".join(tail)) + "}}"
    return rendered


def _latex_expression(expression: Expression) -> str:
    if isinstance(expression, Literal):
        return _number(expression.value)
    if isinstance(expression, Symbol):
        return _latex_symbol(expression.name)
    if isinstance(expression, Unary):
        operand = _latex_expression(expression.operand)
        if expression.operator == "not":
            return r"\neg\left(" + operand + r"\right)"
        return expression.operator + r"\left(" + operand + r"\right)"
    if isinstance(expression, Binary):
        left = _latex_expression(expression.left)
        right = _latex_expression(expression.right)
        if expression.operator == "/":
            return r"\frac{" + left + "}{" + right + "}"
        if expression.operator == "**":
            return r"\left(" + left + r"\right)^{" + right + "}"
        operator = {
            "*": r"\,",
            "+": "+",
            "-": "-",
            "and": r"\land",
            "or": r"\lor",
        }[expression.operator]
        return r"\left(" + left + f" {operator} " + right + r"\right)"
    if isinstance(expression, Comparison):
        operator = {
            "<": "<",
            "<=": r"\le",
            ">": ">",
            ">=": r"\ge",
            "==": "=",
            "!=": r"\ne",
        }[expression.operator]
        return (
            r"\left("
            + _latex_expression(expression.left)
            + f" {operator} "
            + _latex_expression(expression.right)
            + r"\right)"
        )
    if isinstance(expression, Conditional):
        return (
            r"\begin{cases}"
            + _latex_expression(expression.when_true)
            + r", & \text{if } "
            + _latex_expression(expression.condition)
            + r" \\ "
            + _latex_expression(expression.when_false)
            + r", & \text{otherwise}\end{cases}"
        )
    if isinstance(expression, Call):
        arguments = [_latex_expression(item) for item in expression.arguments]
        if expression.function == "der":
            return r"\frac{d " + arguments[0] + r"}{d t}"
        if expression.function == "sqrt":
            return r"\sqrt{" + arguments[0] + "}"
        if expression.function == "abs":
            return r"\left|" + arguments[0] + r"\right|"
        if expression.function == "exp":
            return r"\exp\left(" + arguments[0] + r"\right)"
        if expression.function == "log10":
            return r"\log_{10}\left(" + arguments[0] + r"\right)"
        if expression.function in {
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "tanh",
            "log",
            "min",
            "max",
        }:
            return (
                "\\" + expression.function + r"\left("
                + ", ".join(arguments)
                + r"\right)"
            )
        return (
            r"\operatorname{" + _latex_escape(expression.function) + r"}\left("
            + ", ".join(arguments)
            + r"\right)"
        )
    raise TypeError(f"unsupported expression node {type(expression).__name__}")


def expression_to_latex(source: str) -> str:
    """Render one admitted scalar DSL expression as deterministic LaTeX."""

    return _latex_expression(parse_expression(source))


def expression_comments(source: str) -> tuple[str, ...]:
    """Return Python-tokenizer comments admitted inside a DSL expression string."""

    try:
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        return tuple(
            item.string[1:].strip()
            for item in tokens
            if item.type == tokenize.COMMENT and item.string[1:].strip()
        )
    except (IndentationError, tokenize.TokenError):
        return ()


def _bounds(lower: float | None, upper: float | None) -> str:
    left = r"-\infty" if lower is None else _number(lower)
    right = r"\infty" if upper is None else _number(upper)
    return f"$[{left}, {right}]$"


def _append_json(lines: list[str], heading: str, value: Any) -> None:
    plain = _plain(value)
    if plain in ({}, [], None, ""):
        return
    lines.extend((f"#### {heading}", "", "```json", _json(plain), "```", ""))


def _interface_paths(part_directory: Path, catalog_root: Path) -> tuple[Path, ...]:
    contract = part_directory.parent.parent
    relative = contract.relative_to(catalog_root)
    if len(relative.parts) not in {2, 3}:
        raise ValueError(
            f"{part_directory}: expected domain/category[/device]/instantiations/part"
        )
    paths = [catalog_root / relative.parts[0] / "interface.pmdl"]
    paths.append(catalog_root.joinpath(*relative.parts[:2]) / "interface.pmdl")
    if len(relative.parts) == 3:
        paths.append(contract / "interface.pmdl")
    return tuple(paths)


def _infer_catalog_root(part_directory: Path) -> Path:
    if part_directory.parent.name != "instantiations":
        raise ValueError(f"{part_directory}: parent directory must be named 'instantiations'")
    contract = part_directory.parent.parent
    interface = load_interface(contract / "interface.pmdl")
    if isinstance(interface, CategoryInterface):
        return contract.parent.parent.resolve()
    if isinstance(interface, DeviceInterface):
        return contract.parent.parent.parent.resolve()
    raise ValueError(f"{contract}: part contract must be a category or device interface")


def _model_source_paths(catalog_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in concrete_model_paths(catalog_root):
        model = load_model(path)
        if model.id in result:
            raise ValueError(f"duplicate PMDL model id {model.id!r}")
        result[model.id] = path
    return result


def _source_manifest(
    part_directory: Path,
    catalog_root: Path,
    interfaces: Sequence[Path],
    model_paths: Sequence[Path],
) -> list[tuple[str, str]]:
    paths = {
        path.resolve()
        for path in (*interfaces, *model_paths)
        if path.is_file()
    }
    paths.update(
        path.resolve()
        for path in part_directory.rglob("*")
        if path.is_file() and path.name != PART_README_FILENAME
    )
    return [
        (
            path.relative_to(catalog_root).as_posix(),
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(paths)
    ]


def _render_interface(lines: list[str], interface: ModelInterface, path: Path, root: Path) -> None:
    lines.extend(
        (
            f"### {interface.kind.title()}: {interface.name} ({_code(interface.id)})",
            "",
            f"Source: {_code(path.relative_to(root).as_posix())}; version {_code(interface.version)}.",
            "",
            "**Programmatically enforced contract**",
            "",
        )
    )
    if isinstance(interface, DomainInterface):
        lines.append(
            "- Required physics domains: "
            + ", ".join(_code(item) for item in interface.requires_physics)
            + "."
        )
        lines.append(
            "- Allowed power-port domains: "
            + ", ".join(_code(item) for item in interface.allowed_port_domains)
            + "."
        )
        desires: Sequence[str] = ()
    else:
        if isinstance(interface, CategoryInterface):
            lines.append(
                "- Implemented domains: "
                + ", ".join(_code(item) for item in interface.domains)
                + "."
            )
            lines.append(
                "- Registered ideal models: "
                + (", ".join(_code(item) for item in interface.ideal_models) or "none")
                + "."
            )
        else:
            lines.append(f"- Parent category: {_code(interface.parent)}.")
            lines.append(
                f"- Contract changes are {_code(interface.changes_contract)}; registered models: "
                + (", ".join(_code(item) for item in interface.models) or "none")
                + "."
            )
        for port in interface.required_power_ports:
            lines.append(
                f"- Required power port {_code(port.name)}: domain {_code(port.domain)}, "
                f"effort {_code(port.effort_unit)}, flow {_code(port.flow_unit)}."
            )
        for port in interface.required_signal_ports:
            lines.append(
                f"- Required signal port {_code(port.name)}: {_code(port.direction)}, "
                f"unit {_code(port.unit)}."
            )
        desires = interface.constraints
    lines.extend(("", "**Text-based explanation or desire (not executable by itself)**", ""))
    if interface.description:
        lines.append(f"- Description: {interface.description}")
    if isinstance(interface, DeviceInterface):
        lines.append(f"- Model-specificity rationale: {interface.model_specificity}")
    for constraint in desires:
        lines.append(f"- Interface desire: {constraint}")
    if not interface.description and not desires and not isinstance(interface, DeviceInterface):
        lines.append("- None declared.")
    lines.append("")


def _render_physical_part(
    lines: list[str], part: PartInstantiation, procurement_records: tuple[Any, ...]
) -> None:
    static = part.static
    lines.extend(
        (
            "## Physical part",
            "",
            "The static record below is invariant across the model hypotheses listed later.",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Part id | {_code(static.id)} |",
            f"| Display name | {_markdown_text(static.name)} |",
            f"| Static version | {_code(static.version)} |",
            f"| Physical role | {_code(static.physical_role)} |",
            f"| Provenance kind | {_code(static.provenance.kind)} |",
            f"| Provenance source | {_markdown_text(static.provenance.source)} |",
            "",
            "### Programmatically enforced physical structure",
            "",
        )
    )
    if static.bodies:
        lines.extend(("| Body / solid | Geometry | Dimensions (m) | Evidence |", "|---|---|---|---|"))
        for body in static.bodies:
            for solid in body.solids:
                geometry = solid.geometry
                dimensions = " × ".join(_number(value) for value in geometry.dimensions_m)
                lines.append(
                    f"| {_code(body.id + '.' + solid.id)} | {_code(geometry.kind)} | "
                    f"{dimensions} | {_code(solid.provenance.kind)}: "
                    f"{_markdown_text(solid.provenance.source)} |"
                )
        lines.append("")
    else:
        lines.extend(("- No material bodies are declared for this boundary/software part.", ""))
    if static.connectors:
        lines.extend(
            (
                "| Connector | Model port | Domain | Interface | Fabrication | Evidence |",
                "|---|---|---|---|---|---|",
            )
        )
        for connector in static.connectors:
            if connector.fabrication is None:
                fabrication = "missing record"
            else:
                missing = connector.fabrication.connector_missing_fields()
                fabrication = connector.fabrication.status
                if missing:
                    fabrication += ": " + ", ".join(missing)
            lines.append(
                f"| {_code(connector.id)} | {_code(connector.model_port) if connector.model_port else '—'} | "
                f"{_code(connector.domain)} | {_code(connector.interface)} | "
                f"{_markdown_text(fabrication)} | "
                f"{_code(connector.provenance.kind)}: {_markdown_text(connector.provenance.source)} |"
            )
        lines.append("")
    else:
        lines.extend(("- No connectors are declared.", ""))
    if static.parameter_bindings:
        lines.extend(("Parameter bindings are validated against physical geometry:", ""))
        for binding in static.parameter_bindings:
            lines.append(
                f"- {_code(binding.model_parameter)} in {_code(binding.unit)} "
                f"with absolute tolerance {_number(binding.absolute_tolerance)}."
            )
        lines.append("")
    _append_json(lines, "Static-part metadata", static.metadata)
    lines.extend(("### External procurement records", ""))
    if not procurement_records:
        lines.extend(
            (
                "- No evidence-backed procurement record currently provides this exact "
                "static-part version and hash.",
                "",
            )
        )
    for record in procurement_records:
        identifiers = ", ".join(
            f"{item.scheme}={item.value}" for item in record.identifiers
        )
        lines.append(
            f"- {_code(record.id)} ({_code(record.sha256)}): "
            f"{_markdown_text(record.manufacturer or 'manufacturer unspecified')}; "
            f"{_markdown_text(identifiers)}."
        )
        for document in record.documents:
            lines.append(
                f"  - {document.kind}: [{_markdown_text(document.url)}]({document.url})"
            )
        for offer in record.offers:
            lines.append(
                f"  - purchase offer: [{_markdown_text(offer.supplier)}]"
                f"({offer.purchase_url}) ({offer.availability}, observed {offer.observed_at})"
            )
        lines.append("")


def _annotation_rows(model: ModelSpec) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if model.description:
        rows.append(("model", model.id, model.description))
    collections = (
        ("state", model.states),
        ("algebraic", model.algebraics),
        ("parameter", model.parameters),
        ("power port", model.power_ports),
        ("signal port", model.signal_ports),
        ("artifact port", model.artifact_ports),
        ("relation", model.relations),
        ("stored energy", model.stored_energy),
        ("dissipation", model.dissipation),
        ("source", model.sources),
        ("property", model.properties),
        ("fidelity", model.fidelity_levels),
        ("noise channel", model.process_noise.channels),
        ("noise increment", model.process_noise.increments),
    )
    for kind, values in collections:
        for value in values:
            description = getattr(value, "description", "")
            name = getattr(value, "name", getattr(value, "target", ""))
            if description:
                rows.append((kind, name, description))
    for port in model.power_ports:
        if port.reference:
            rows.append(("power-port reference", port.name, port.reference))
    return rows


def _render_expression(
    lines: list[str],
    label: str,
    source: str,
    *,
    suffix: str = "",
    description: str = "",
) -> None:
    lines.extend((f"#### {label}", "", "$$", expression_to_latex(source) + suffix, "$$", ""))
    lines.append(f"DSL source: {_code(source)}")
    if description:
        lines.append(f"Annotation: {description}")
    for comment in expression_comments(source):
        lines.append(f"Expression comment: {comment}")
    lines.append("")


def _render_model(
    lines: list[str],
    instance: PartInstantiation,
    model: ModelSpec,
    source_path: Path,
    catalog_root: Path,
) -> None:
    spec = instance.model_instance
    lines.extend(
        (
            f"## Model hypothesis {spec.variant}: {model.name}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Variant | {_code(spec.variant)} |",
            f"| Model name | {_markdown_text(model.name)} |",
            f"| PMDL id | {_code(model.id)} |",
            f"| PMDL version | {_code(model.version)} |",
            f"| Exact PMDL hash | {_code(spec.model.sha256)} |",
            f"| PMDL source | {_code(source_path.relative_to(catalog_root).as_posix())} |",
            f"| Instance condition | {_code(spec.condition)} |",
            f"| Relative compute cost | {_number(spec.compute.relative_cost)} |",
            "",
            "### Programmatically enforced parameters and constraints",
            "",
            "Every declared PMDL parameter must be initialized exactly once. Values are checked for finiteness and bounds.",
            "",
            "| Parameter | Value | Unit | Allowed bounds | Instance uncertainty | Learnable |",
            "|---|---:|---|---|---|---|",
        )
    )
    for parameter in model.parameters:
        value = spec.parameters[parameter.name]
        scalar = value.get("value") if isinstance(value, Mapping) else value
        uncertainty = spec.parameter_uncertainty.get(parameter.name)
        lines.append(
            f"| {_code(parameter.name)} | {_markdown_text(scalar)} | {_code(parameter.unit)} | "
            f"{_bounds(parameter.bounds.lower, parameter.bounds.upper)} | "
            f"{_code(_json(uncertainty)) if uncertainty is not None else 'fixed/not separately declared'} | "
            f"{_code(parameter.learnable)} |"
        )
    if not model.parameters:
        lines.append("| _None_ | — | — | — | — | — |")
    lines.append("")
    for relation in model.relations:
        _render_expression(
            lines,
            f"Residual relation: {relation.name}",
            relation.expression,
            suffix=" = 0",
            description=relation.description,
        )
    for heading, values in (
        ("Stored energy", model.stored_energy),
        ("Dissipation", model.dissipation),
        ("Source", model.sources),
    ):
        for value in values:
            _render_expression(
                lines,
                f"{heading}: {value.name} [{value.unit}]",
                value.expression,
                description=value.description,
            )
    for constraint in model.initialization.constraints:
        _render_expression(
            lines,
            f"Initialization constraint: {constraint.name}",
            constraint.expression,
            suffix=" = 0",
            description=constraint.description,
        )
    for increment in model.process_noise.increments:
        _render_expression(
            lines,
            f"Accepted-step stochastic increment for {increment.target}",
            increment.expression,
            description=increment.description,
        )
    if model.modes:
        lines.extend(("### Programmatically enforced discrete modes", ""))
        for mode in model.modes:
            lines.append(
                f"- Mode {_code(mode.name)}{' (initial)' if mode.initial else ''}; active relations: "
                + (", ".join(_code(item) for item in mode.active_relations) or "none")
                + "."
            )
            for transition in mode.transitions:
                lines.append(
                    f"  - Transition to {_code(transition.target)} when "
                    f"${expression_to_latex(transition.guard)}$."
                )
                for target, reset in transition.resets.items():
                    lines.append(
                        f"    - Reset {_code(target)} to ${expression_to_latex(reset)}$."
                    )
        lines.append("")
    lines.extend(("### Additional machine-checked declarations", ""))
    if model.validity.ranges:
        for name, bounds in model.validity.ranges.items():
            lines.append(
                f"- Validity range for {_code(name)}: {_bounds(bounds.lower, bounds.upper)}."
            )
    if model.validity.max_timestep is not None:
        lines.append(f"- Maximum supported timestep: {_number(model.validity.max_timestep)} s.")
    if model.initialization.required:
        lines.append(
            "- Consistent initialization requires: "
            + ", ".join(_code(item) for item in model.initialization.required)
            + "."
        )
    for prop in model.properties:
        lines.append(
            f"- Property {_code(prop.name)} ({_code(prop.kind)}), expected {_code(prop.expected)}, "
            f"sample count {prop.samples}, tolerance {_number(prop.tolerance)}: "
            f"${expression_to_latex(prop.expression)}$."
        )
    if not (
        model.validity.ranges
        or model.validity.max_timestep is not None
        or model.initialization.required
        or model.properties
    ):
        lines.append("- None beyond schema, interface, equation, and parameter validation.")
    lines.extend(("", "### Text-based desires, assumptions, and explanatory notes", ""))
    rows = _annotation_rows(model)
    if rows:
        lines.extend(("| Location | Name | Text |", "|---|---|---|"))
        for kind, name, text in rows:
            lines.append(
                f"| {_markdown_text(kind)} | {_code(name)} | {_markdown_text(text)} |"
            )
        lines.append("")
    else:
        lines.extend(("- No descriptions were declared.", ""))
    for assumption in model.validity.assumptions:
        lines.append(f"- Validity assumption (text only): {assumption}")
    for fidelity in model.fidelity_levels:
        lines.append(
            f"- Fidelity {_code(fidelity.name)} approximation note: {fidelity.approximation_error}"
        )
    if spec.compute.notes:
        lines.append(f"- Compute note: {spec.compute.notes}")
    if model.trust.evidence:
        for evidence in model.trust.evidence:
            detail = f" — {evidence.summary}" if evidence.summary else ""
            lines.append(
                f"- Evidence claim {_code(evidence.kind)}: {_markdown_text(evidence.reference + detail)}"
            )
    lines.append("")
    lines.extend(
        (
            "Trust labels are recorded claims, not automatic physical qualification:",
            "",
            f"- Structural: {_code(model.trust.structural)}",
            f"- Physical: {_code(model.trust.physical)}",
            f"- Numerical: {_code(model.trust.numerical)}",
            f"- Empirical: {_code(model.trust.empirical)}",
            "",
        )
    )
    _append_json(lines, "PMDL metadata", model.metadata)
    _append_json(lines, "Model-instance metadata", spec.metadata)


def render_part_markdown(
    part_directory: str | Path, *, catalog_root: str | Path | None = None
) -> str:
    """Validate and render one catalog instantiation directory as standalone Markdown."""

    directory = Path(part_directory).resolve()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"part_directory must be a regular directory: {directory}")
    root = (
        Path(catalog_root).resolve()
        if catalog_root is not None
        else _infer_catalog_root(directory)
    )
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"part directory {directory} is outside catalog root {root}") from exc
    if directory.parent.name != "instantiations" or len(relative.parts) not in {4, 5}:
        raise ValueError(
            f"{relative}: expected domain/category[/device]/instantiations/part"
        )

    interface_catalog = load_interface_catalog(root)
    models = ModelRegistry()
    models.load_directory(root, interfaces=interface_catalog)
    registry = PartInstantiationRegistry.load_catalog(root, models=models)
    variants = sorted(
        (item for item in registry.values() if item.directory.resolve() == directory),
        key=lambda item: int(item.model_instance.variant[1:]),
    )
    if not variants:
        raise ValueError(f"{relative}: no validated model instances found")
    static_ids = {item.static.id for item in variants}
    if len(static_ids) != 1:
        raise ValueError(f"{relative}: model variants do not share one physical part")

    parent_paths = _interface_paths(directory, root)
    parents = tuple(load_interface(path) for path in parent_paths)
    source_by_model = _model_source_paths(root)
    resolved_models = tuple(models[item.model_instance.model.id] for item in variants)
    model_paths = tuple(source_by_model[model.id] for model in resolved_models)
    sources = _source_manifest(directory, root, parent_paths, model_paths)

    static = variants[0].static
    lines: list[str] = [
        f"# {static.name}",
        "",
        "> This file is generated deterministically from a validated part directory. "
        "Edit the authoritative `static.part`, `vN.model`, PMDL, or interface records and "
        "regenerate it; do not edit this file by hand.",
        "",
        "## How to read this document",
        "",
        "A **part** is one physical item described by `static.part`. A **model hypothesis** "
        "is a `vN.model` choice that binds that same item to one exact Physical Model DSL "
        "(PMDL) program and initializes all of its parameters. One part may therefore have "
        "several model hypotheses with different fidelity, assumptions, or cost.",
        "",
        "- **Programmatically enforced** means a parser, catalog validator, dimensional checker, "
        "or runtime consumes and checks the declaration. Equations, bounds, exact hashes, ports, "
        "and initialized parameters fall in this category.",
        "- **Text-based desire or explanation** means prose is preserved for people and agents but "
        "does not itself create an executable constraint. Descriptions, interface `constraints`, "
        "assumptions, notes, evidence summaries, and metadata fall in this category unless an "
        "independent executable declaration also enforces the same idea.",
        "- A model being structurally valid does **not** establish empirical accuracy, safety, or "
        "regulatory qualification. Trust labels and evidence claims are reported below without "
        "upgrading them.",
        "- Scalar equations use SI-compatible declared units. PMDL relations are acausal residuals: "
        "each displayed residual is enforced as equal to zero, and all active relations are solved "
        "simultaneously.",
        "",
        "## Model hypotheses at a glance",
        "",
        "| Variant | Model name | PMDL id | PMDL version | Exact hash | Condition |",
        "|---|---|---|---|---|---|",
    ]
    for instance, model in zip(variants, resolved_models, strict=True):
        spec = instance.model_instance
        lines.append(
            f"| {_code(spec.variant)} | {_markdown_text(model.name)} | {_code(model.id)} | "
            f"{_code(model.version)} | {_code(spec.model.sha256)} | {_code(spec.condition)} |"
        )
    lines.extend(("", "## Parent interface chain", ""))
    lines.append(
        "Interfaces describe the domain/category/device contracts inherited by this part. "
        "Required domains and ports are enforced; prose constraints are listed separately."
    )
    lines.append("")
    for interface, path in zip(parents, parent_paths, strict=True):
        _render_interface(lines, interface, path, root)
    _render_physical_part(
        lines,
        variants[0],
        registry.procurement.for_part(variants[0].static.id),
    )
    for instance, model, path in zip(variants, resolved_models, model_paths, strict=True):
        _render_model(lines, instance, model, path, root)
    lines.extend(
        (
            "## Authoritative source manifest",
            "",
            "These hashes bind the inputs used for this rendering. The generated README is excluded "
            "to avoid a self-referential digest.",
            "",
            "| Catalog-relative source | SHA-256 |",
            "|---|---|",
        )
    )
    for path, digest in sources:
        lines.append(f"| {_code(path)} | {_code(digest)} |")
    lines.append("")
    return "\n".join(lines)


def write_part_markdown(
    part_directory: str | Path,
    *,
    catalog_root: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    """Write a validated part's deterministic standalone Markdown file."""

    directory = Path(part_directory).resolve()
    target = Path(output).resolve() if output is not None else directory / PART_README_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = render_part_markdown(directory, catalog_root=catalog_root)
    target.write_text(payload, encoding="utf-8", newline="\n")
    return target


__all__ = [
    "PART_README_FILENAME",
    "expression_comments",
    "expression_to_latex",
    "render_part_markdown",
    "write_part_markdown",
]
