"""Controller compilation entry points.

Both targets consume exactly the same immutable :class:`ControlIR`.  C99 uses
native double-precision scalars; Verilog uses a declared signed fixed-point Q
format and exposes only explicit controller inputs as public data ports.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..specs import ControlSpec
from .c99 import generate_c99
from .ir import CompilationBundle, ControlCompilerError, ControlExpression, ControlIR, FixedPointFormat, GeneratedArtifact
from .verilog import generate_verilog


_TARGETS = ("c99", "verilog")


def compile_control(
    spec: ControlSpec | ControlIR,
    *,
    targets: Iterable[str] = _TARGETS,
    identifier: str | None = None,
    fixed_point: FixedPointFormat | None = None,
) -> CompilationBundle:
    """Compile a controller to deterministic C99 and/or fixed-point Verilog."""

    ir = (
        spec
        if isinstance(spec, ControlIR)
        else ControlIR.from_spec(spec, identifier=identifier)
    )
    if isinstance(spec, ControlIR) and identifier is not None:
        raise ControlCompilerError("cannot rename an existing ControlIR")
    selected = tuple(targets)
    if not selected:
        raise ControlCompilerError("at least one controller target is required")
    if len(selected) != len(set(selected)):
        raise ControlCompilerError("controller targets may not be repeated")
    unknown = sorted(set(selected) - set(_TARGETS))
    if unknown:
        raise ControlCompilerError(f"unknown controller target(s): {unknown}")
    fixed = FixedPointFormat() if fixed_point is None else fixed_point
    artifacts: list[GeneratedArtifact] = []
    for target in selected:
        if target == "c99":
            artifacts.extend(generate_c99(ir))
        else:
            artifacts.append(generate_verilog(ir, fixed_point=fixed))
    return CompilationBundle(
        ir.spec.id,
        ir.source_digest,
        tuple(artifacts),
        selected,
        fixed,
        ir.observability,
        ir.closure,
    )


def compile_resolved_controller(
    assembly: object,
    controller_id: str,
    *,
    targets: Iterable[str] = _TARGETS,
    identifier: str | None = None,
    fixed_point: FixedPointFormat | None = None,
) -> CompilationBundle:
    """Compile one controller entirely in memory from its resolved PMDL closure."""

    return compile_control(
        ControlIR.from_resolved(
            assembly, controller_id, identifier=identifier
        ),
        targets=targets,
        fixed_point=fixed_point,
    )


__all__ = [
    "CompilationBundle",
    "ControlCompilerError",
    "ControlExpression",
    "ControlIR",
    "FixedPointFormat",
    "GeneratedArtifact",
    "compile_control",
    "compile_resolved_controller",
    "generate_c99",
    "generate_verilog",
]
