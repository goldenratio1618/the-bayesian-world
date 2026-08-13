"""Target-neutral controller compiler intermediate representation."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import weakref

from ...physics.dsl import Call, Expression, parse_expression
from ...physics.specs import FrozenDict
from ..observer import (
    AffineObserverModel,
    ObservabilityDiagnostic,
    ObserverDerivationError,
    derive_affine_observer,
)
from ..specs import ControlSpec, control_digest


class ControlCompilerError(ValueError):
    """A controller cannot be represented by the requested target."""


_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")
_TARGET_KEYWORDS = frozenset(
    """
    always and assign auto automatic begin bool break buf case char complex const continue
    default do double else end endcase endfunction endmodule enum extern float
    false for function generate genvar goto if imaginary initial inline inout input int integer
    localparam long module negedge or output parameter posedge real reg register
    restrict return short signed sizeof static struct switch task typedef union
    true unsigned void volatile while wire
    """.split()
)
_PENDING_IR_ISSUANCE: ContextVar[object | None] = ContextVar(
    "contraption_control_ir_issuance", default=None
)
_ISSUED_IR: dict[
    int,
    tuple[
        weakref.ReferenceType["ControlIR"],
        object,
        str,
        str | None,
        str | None,
    ],
] = {}


def target_identifier(value: str) -> str:
    """Return a deterministic C/Verilog-safe identifier."""

    result = _IDENTIFIER.sub("_", value)
    if not result or not result[0].isalpha():
        result = f"control_{result}"
    result = result.lower()
    return f"control_{result}" if result in _TARGET_KEYWORDS else result


def admit_target_symbol_table(
    target: str,
    namespaces: Mapping[str, tuple[tuple[str, str], ...]],
) -> None:
    """Require each emitted target namespace to be injective."""

    for namespace, entries in namespaces.items():
        claimed: dict[str, str] = {}
        for source, emitted in entries:
            previous = claimed.get(emitted)
            if previous is not None and previous != source:
                raise ControlCompilerError(
                    f"{target} symbol collision in {namespace}: {previous!r} and "
                    f"{source!r} both emit {emitted!r}"
                )
            claimed[emitted] = source


def _freeze_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict(
            (str(key), _freeze_data(item)) for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_data(item) for item in value)
    return value


def plain_compiler_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): plain_compiler_data(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [plain_compiler_data(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FixedPointFormat:
    """Signed two's-complement Q format used by the Verilog target."""

    total_bits: int = 48
    fractional_bits: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.total_bits, bool) or not isinstance(self.total_bits, int):
            raise TypeError("total_bits must be an integer")
        if isinstance(self.fractional_bits, bool) or not isinstance(self.fractional_bits, int):
            raise TypeError("fractional_bits must be an integer")
        if not 8 <= self.total_bits <= 62:
            raise ValueError("total_bits must be between 8 and 62")
        if not 0 <= self.fractional_bits <= self.total_bits - 2:
            raise ValueError("fractional_bits must leave a sign and integer bit")

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def minimum(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def maximum(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    def quantize(self, value: float) -> int:
        if not math.isfinite(float(value)):
            raise ControlCompilerError("fixed-point constants must be finite")
        result = int(round(float(value) * self.scale))
        if result < self.minimum or result > self.maximum:
            raise ControlCompilerError(
                f"constant {value!r} does not fit Q{self.total_bits - self.fractional_bits - 1}."
                f"{self.fractional_bits}"
            )
        return result

    def to_dict(self) -> dict[str, int]:
        return {
            "total_bits": self.total_bits,
            "fractional_bits": self.fractional_bits,
        }


@dataclass(frozen=True, slots=True)
class ControlExpression:
    path: str
    source: str
    expression: Expression
    symbols: frozenset[str]


def _contains_der(node: Expression) -> bool:
    if isinstance(node, Call) and node.function == "der":
        return True
    for attribute in ("operand", "left", "right", "condition", "when_true", "when_false"):
        child = getattr(node, attribute, None)
        if isinstance(child, Expression) and _contains_der(child):
            return True
    return any(_contains_der(child) for child in getattr(node, "arguments", ()))


def _controller_expressions(spec: ControlSpec) -> tuple[ControlExpression, ...]:
    sources: list[tuple[str, str]] = []
    for item in spec.derived:
        sources.append((f"derived.{item.name}", item.expression))
    if spec.emergency_when is not None:
        sources.append(("emergency_when", spec.emergency_when))
    for mode in spec.modes:
        for name, source in mode.outputs.items():
            sources.append((f"modes.{mode.name}.outputs.{name}", source))
        for name, source in mode.updates.items():
            sources.append((f"modes.{mode.name}.updates.{name}", source))
        for index, transition in enumerate(mode.transitions):
            sources.append(
                (f"modes.{mode.name}.transitions[{index}].guard", transition.guard)
            )
    result: list[ControlExpression] = []
    for path, source in sources:
        expression = parse_expression(source)
        if _contains_der(expression):
            raise ControlCompilerError(f"{path}: der() is not legal in a controller")
        result.append(
            ControlExpression(path, source, expression, expression.variables())
        )
    return tuple(result)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ControlIR:
    """Validated, immutable input shared by every code generator."""

    spec: ControlSpec
    identifier: str
    source_digest: str
    expressions: tuple[ControlExpression, ...]
    observability: tuple[ObservabilityDiagnostic, ...]
    observer: AffineObserverModel | None = None
    closure: Mapping[str, Any] | None = None
    _issuance: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        issuance = _PENDING_IR_ISSUANCE.get()
        if issuance is None:
            raise ControlCompilerError(
                "ControlIR instances are factory-issued; use ControlIR.from_spec or "
                "ControlIR.from_resolved"
            )
        object.__setattr__(self, "_issuance", issuance)
        if not isinstance(self.spec, ControlSpec):
            raise ControlCompilerError("ControlIR.spec must be a ControlSpec")
        if self.identifier != target_identifier(self.identifier):
            raise ControlCompilerError(
                "ControlIR.identifier must already be a canonical target identifier"
            )
        expected_digest = control_digest(self.spec)
        if self.source_digest != expected_digest:
            raise ControlCompilerError(
                "ControlIR.source_digest does not match canonical controller content"
            )
        expected_expressions = _controller_expressions(self.spec)
        if self.expressions != expected_expressions:
            raise ControlCompilerError(
                "ControlIR expressions do not match the canonical controller graph"
            )
        if bool(self.spec.implicit_inputs) != (self.observer is not None):
            raise ControlCompilerError(
                "ControlIR observer presence does not match implicit inputs"
            )
        if self.observer is not None:
            if (
                self.observer.controller_id != self.spec.id
                or self.observer.controller_digest != expected_digest
                or self.observability != self.observer.observability
                or self.closure is None
            ):
                raise ControlCompilerError(
                    "ControlIR observer/provenance does not match the controller"
                )
            expected_closure = {
                "assembly_sha256": self.observer.assembly_sha256,
                "pmdl_sha256": self.observer.pmdl_sha256,
                "controller_link_digest": self.observer.controller_link_digest,
                "dynamics_completeness": plain_compiler_data(
                    self.observer.dynamics_completeness
                ),
                "observer_digest": self.observer.digest,
                "observer_derivation": plain_compiler_data(self.observer.derivation),
                "observer_operating_point": plain_compiler_data(
                    self.observer.operating_point
                ),
                "observer_validity": plain_compiler_data(self.observer.validity),
            }
            if plain_compiler_data(self.closure) != expected_closure:
                raise ControlCompilerError(
                    "ControlIR closure does not exactly match resolved observer provenance"
                )
        elif self.closure is not None:
            plain_closure = plain_compiler_data(self.closure)
            expected_keys = {
                "assembly_sha256",
                "pmdl_sha256",
                "controller_link_digest",
                "dynamics_completeness",
                "observer_digest",
                "observer_derivation",
                "observer_operating_point",
                "observer_validity",
            }
            if (
                not isinstance(plain_closure, dict)
                or set(plain_closure) != expected_keys
                or any(
                    plain_closure[name] is not None
                    for name in (
                        "observer_digest",
                        "observer_derivation",
                        "observer_operating_point",
                        "observer_validity",
                    )
                )
            ):
                raise ControlCompilerError(
                    "observer-free resolved ControlIR closure is malformed"
                )
        if self.closure is not None:
            object.__setattr__(self, "closure", _freeze_data(self.closure))

    @classmethod
    def from_spec(
        cls, spec: ControlSpec, *, identifier: str | None = None
    ) -> "ControlIR":
        if not isinstance(spec, ControlSpec):
            raise TypeError("ControlIR.from_spec requires a ControlSpec")
        if spec.implicit_inputs:
            raise ControlCompilerError(
                "controllers with implicit inputs must be compiled from a canonical "
                "ResolvedAssembly via ControlIR.from_resolved"
            )
        return _issue_ir(
            cls,
            spec=spec,
            identifier=target_identifier(spec.id if identifier is None else identifier),
            source_digest=control_digest(spec),
            expressions=_controller_expressions(spec),
            observability=(),
            observer=None,
            closure=None,
        )

    @classmethod
    def from_resolved(
        cls,
        assembly: Any,
        controller_id: str,
        *,
        identifier: str | None = None,
    ) -> "ControlIR":
        """Build a target IR from one canonical resolved controller closure."""

        from ...physics.resolved import ResolutionError, ResolvedAssembly

        if not isinstance(assembly, ResolvedAssembly):
            raise ControlCompilerError(
                "resolved controller compilation requires a ResolvedAssembly"
            )
        try:
            controller = assembly.controllers[controller_id]
        except KeyError as exc:
            raise ControlCompilerError(
                f"resolved assembly has no controller {controller_id!r}"
            ) from exc
        try:
            canonical_system = assembly.attest_pmdl_system()
        except ResolutionError as exc:
            raise ControlCompilerError(
                f"resolved PMDL system provenance attestation failed: {exc}"
            ) from exc
        canonical_links = {
            link.id: link for link in assembly.specification.controllers
        }
        try:
            link = canonical_links[controller_id]
        except KeyError as exc:
            raise ControlCompilerError(
                f"resolved controller {controller_id!r} has no canonical controller link"
            ) from exc
        if controller.id != link.id:
            raise ControlCompilerError(
                "resolved controller id does not match its canonical controller link"
            )
        spec_digest = control_digest(controller.spec)
        if link.program.sha256 != spec_digest:
            raise ControlCompilerError(
                "resolved controller content does not match the canonical program sha256"
            )
        expected_link_digest = "sha256:" + hashlib.sha256(
            json.dumps(
                link.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if controller.controller_link_digest != expected_link_digest:
            raise ControlCompilerError(
                "resolved controller link digest does not match the canonical link"
            )
        authored_explicit = dict(link.explicit_inputs)
        if set(controller.explicit_input_bindings) != set(authored_explicit):
            raise ControlCompilerError(
                "resolved explicit-input bindings do not match canonical link coverage"
            )
        for name, resolved_binding in controller.explicit_input_bindings.items():
            authored = authored_explicit[name]
            expected_kind = "sensor" if authored.signal is not None else "external"
            expected_source = authored.signal if authored.signal is not None else authored.external
            canonical_state_name: str | None = None
            canonical_state_index: int | None = None
            if expected_kind == "sensor":
                assert authored.signal is not None
                try:
                    canonical_state_name, canonical_state_index = (
                        assembly.resolve_signal_state(
                            authored.signal,
                            direction="output",
                            system=canonical_system,
                            context=f"controller {controller_id!r} input {name!r}",
                        )
                    )
                except ResolutionError as exc:
                    raise ControlCompilerError(str(exc)) from exc
            if (
                resolved_binding.kind != expected_kind
                or resolved_binding.source != expected_source
                or (expected_kind == "sensor"
                    and (
                        resolved_binding.state_name != canonical_state_name
                        or resolved_binding.state_index != canonical_state_index
                    ))
                or (expected_kind == "external"
                    and (
                        resolved_binding.state_name is not None
                        or resolved_binding.state_index is not None
                    ))
            ):
                raise ControlCompilerError(
                    f"resolved explicit-input binding {name!r} does not match the canonical link"
                )
        authored_implicit = dict(link.implicit_inputs)
        if set(controller.implicit_input_bindings) != set(authored_implicit):
            raise ControlCompilerError(
                "resolved implicit-input bindings do not match canonical link coverage"
            )
        for name, resolved_binding in controller.implicit_input_bindings.items():
            if (
                resolved_binding.source != authored_implicit[name]
                or resolved_binding.state_name != authored_implicit[name]
                or resolved_binding.state_index is None
                or resolved_binding.state_index >= len(canonical_system.state_names)
                or canonical_system.state_names[resolved_binding.state_index]
                != resolved_binding.state_name
            ):
                raise ControlCompilerError(
                    f"resolved implicit-input binding {name!r} does not match the canonical link"
                )
        if set(controller.output_bindings) != set(link.outputs):
            raise ControlCompilerError(
                "resolved output bindings do not match canonical link coverage"
            )
        for name, resolved_binding in controller.output_bindings.items():
            authored = link.outputs[name]
            expected_kind = "signal" if authored.signal is not None else "external"
            expected_source = (
                f"{link.id}.{name}"
                if expected_kind == "signal"
                else authored.external
            )
            canonical_state_name = None
            if expected_kind == "signal":
                assert authored.signal is not None
                try:
                    canonical_state_name, _ = assembly.resolve_signal_state(
                        authored.signal,
                        direction="input",
                        system=canonical_system,
                        context=f"controller {controller_id!r} output {name!r}",
                    )
                except ResolutionError as exc:
                    raise ControlCompilerError(str(exc)) from exc
            if (
                resolved_binding.kind != expected_kind
                or resolved_binding.source != expected_source
                or (expected_kind == "signal"
                    and resolved_binding.state_name != canonical_state_name)
                or (expected_kind == "external" and resolved_binding.state_name is not None)
            ):
                raise ControlCompilerError(
                    f"resolved output binding {name!r} does not match the canonical link"
                )
        normalized_controller_ids: dict[str, str] = {}
        for resolved_id in assembly.controllers:
            normalized = target_identifier(resolved_id)
            previous = normalized_controller_ids.get(normalized)
            if previous is not None and previous != resolved_id:
                raise ControlCompilerError(
                    "resolved controller IDs collide in the shared C/Verilog target "
                    f"namespace: {previous!r} and {resolved_id!r} both emit {normalized!r}"
                )
            normalized_controller_ids[normalized] = resolved_id
        spec = controller.spec
        observer = controller.observer
        if bool(spec.implicit_inputs) != (observer is not None):
            raise ControlCompilerError(
                "resolved controller observer presence does not match implicit inputs"
            )
        if observer is not None:
            if (
                observer.controller_id != controller.id
                or observer.controller_digest != control_digest(spec)
                or observer.controller_link_digest
                != controller.controller_link_digest
                or observer.assembly_sha256 != assembly.assembly_sha256
                or observer.pmdl_sha256 != canonical_system.pmdl_sha256
            ):
                raise ControlCompilerError(
                    "resolved observer provenance does not match the assembly/controller closure"
                )
        dynamics = assembly.dynamics_completeness
        if observer is not None:
            if dynamics is None:
                raise ControlCompilerError(
                    "implicit observer compilation requires dynamics-completeness provenance"
                )
            try:
                canonical_observer = derive_affine_observer(
                    canonical_system,
                    spec,
                    explicit_bindings=controller.explicit_input_bindings,
                    implicit_bindings=controller.implicit_input_bindings,
                    output_bindings=controller.plant_output_bindings,
                    assembly_sha256=assembly.assembly_sha256,
                    pmdl_sha256=canonical_system.pmdl_sha256,
                    controller_link_digest=controller.controller_link_digest,
                    dynamics_completeness=dynamics.to_dict(),
                )
            except ObserverDerivationError as exc:
                raise ControlCompilerError(
                    f"canonical observer re-derivation failed: {exc}"
                ) from exc
            if observer.digest != canonical_observer.digest:
                raise ControlCompilerError(
                    "resolved observer is not the canonical plant-derived observer for "
                    "this exact assembly/controller closure"
                )
            observer = canonical_observer
        closure = {
            "assembly_sha256": assembly.assembly_sha256,
            "pmdl_sha256": canonical_system.pmdl_sha256,
            "controller_link_digest": controller.controller_link_digest,
            "dynamics_completeness": None
            if dynamics is None
            else dynamics.to_dict(),
            "observer_digest": None if observer is None else observer.digest,
            "observer_derivation": None
            if observer is None
            else dict(observer.derivation),
            "observer_operating_point": None
            if observer is None
            else dict(observer.operating_point),
            "observer_validity": None
            if observer is None
            else dict(observer.validity),
        }
        return _issue_ir(
            cls,
            spec=spec,
            identifier=target_identifier(spec.id if identifier is None else identifier),
            source_digest=control_digest(spec),
            expressions=_controller_expressions(spec),
            observability=() if observer is None else observer.observability,
            observer=observer,
            closure=_freeze_data(closure),
        )


def _issue_ir(ir_type: type[ControlIR], **values: Any) -> ControlIR:
    token = object()
    pending_token = _PENDING_IR_ISSUANCE.set(token)
    try:
        result = ir_type(**values)
    finally:
        _PENDING_IR_ISSUANCE.reset(pending_token)
    closure_fingerprint = (
        None
        if result.closure is None
        else json.dumps(
            plain_compiler_data(result.closure),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    key = id(result)

    def discard(reference: weakref.ReferenceType[ControlIR]) -> None:
        issued = _ISSUED_IR.get(key)
        if issued is not None and issued[0] is reference:
            _ISSUED_IR.pop(key, None)

    reference = weakref.ref(result, discard)
    _ISSUED_IR[key] = (
        reference,
        token,
        control_digest(result.spec),
        None if result.observer is None else result.observer.digest,
        closure_fingerprint,
    )
    return result


def _require_issued_ir(value: ControlIR) -> None:
    issued = _ISSUED_IR.get(id(value))
    if (
        issued is None
        or issued[0]() is not value
        or issued[1] is not value._issuance
    ):
        raise ControlCompilerError(
            "ControlIR is not an exact factory-issued compiler input"
        )
    closure_fingerprint = (
        None
        if value.closure is None
        else json.dumps(
            plain_compiler_data(value.closure),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if (
        issued[2] != control_digest(value.spec)
        or issued[3] != (None if value.observer is None else value.observer.digest)
        or issued[4] != closure_fingerprint
    ):
        raise ControlCompilerError("factory-issued ControlIR was mutated after issuance")


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    """One deterministic generated source file."""

    path: str
    media_type: str
    content: str

    def __post_init__(self) -> None:
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("artifact paths must be safe and relative")
        if not self.content.endswith("\n"):
            object.__setattr__(self, "content", self.content + "\n")

    @property
    def sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CompilationBundle:
    """Generated files and reproducibility metadata for one controller."""

    controller_id: str
    source_digest: str
    artifacts: tuple[GeneratedArtifact, ...]
    targets: tuple[str, ...]
    fixed_point: FixedPointFormat
    observability: tuple[ObservabilityDiagnostic, ...]
    closure: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        paths = [item.path for item in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate artifact paths")

    @property
    def files(self) -> Mapping[str, str]:
        return MappingProxyType({item.path: item.content for item in self.artifacts})

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": "contraption-control-compiler/v2",
                "controller_id": self.controller_id,
                "source_digest": self.source_digest,
                "targets": list(self.targets),
                "fixed_point": self.fixed_point.to_dict(),
                "observability": [item.to_dict() for item in self.observability],
                "closure": None
                if self.closure is None
                else plain_compiler_data(self.closure),
                "files": [
                    {
                        "path": item.path,
                        "media_type": item.media_type,
                        "sha256": item.sha256,
                    }
                    for item in self.artifacts
                ],
            }
        )

    def write(self, directory: str | Path) -> tuple[Path, ...]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for artifact in self.artifacts:
            path = root / artifact.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(artifact.content, encoding="utf-8", newline="\n")
            written.append(path)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(dict(self.manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        written.append(manifest_path)
        return tuple(written)


def as_ir(value: ControlSpec | ControlIR, *, identifier: str | None = None) -> ControlIR:
    if isinstance(value, ControlIR):
        _require_issued_ir(value)
        if identifier is not None and target_identifier(identifier) != value.identifier:
            raise ValueError("cannot rename an existing ControlIR")
        return value
    return ControlIR.from_spec(value, identifier=identifier)


__all__ = [
    "CompilationBundle",
    "ControlCompilerError",
    "ControlExpression",
    "ControlIR",
    "FixedPointFormat",
    "GeneratedArtifact",
    "admit_target_symbol_table",
    "as_ir",
    "target_identifier",
    "plain_compiler_data",
]
