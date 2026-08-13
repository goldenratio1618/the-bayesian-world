# Controller compiler contract

## One behavioral source

Online controller artifacts are generated from the same strict `control-1`
programs executed by the offline simulator. A controller is data, not a Python
callback: its expressions use the PMDL parser and dimensional type system, its
mode/register behavior is synchronous, and its latent inputs carry posterior
uncertainty.

Plant dynamics remain authoritative in PMDL. For a controller with implicit
inputs, resolution derives a coupled local-affine observer from the complete
assembled descriptor system, the exact sensor bindings, a hash-bound operating
point, and a qualified continuous-to-discrete transition at the controller
period. Authors do not provide a second set of transition or
measurement matrices. Generated controller code consumes only explicit pins,
runs that resolved observer internally, and produces declared outputs.

## Canonical admission path

The normal entry point is a self-contained `contraption-4` bundle:

```python
from contraption import load_contraption
from contraption.control import compile_resolved_controller

assembly = load_contraption("assembled_contraptions/scanner/contraption.json")
for controller_id in assembly.controllers:
    bundle = compile_resolved_controller(assembly, controller_id)
    bundle.write(f"outputs/controllers/{controller_id}")
```

`load_contraption` verifies catalog closure, controller paths and canonical
SHA-256 digests, explicit sensor/external bindings, and output wiring before a
`ResolvedAssembly` is returned. The resolved compiler validates the assembly,
PMDL, controller-link, and observer digests again before lowering. A bare
`ControlSpec` may be compiled only when it has no implicit inputs. The compiler
does not accept arbitrary matrices, callbacks, or caller-authored source
fragments.

## Explicit and implicit inputs

- `explicit_inputs` are physical sensor wires or external pins. They are the
  only public data inputs emitted in C or Verilog.
- `implicit_inputs` are latent PMDL quantities inferred inside the controller.
  Each manifest binding names one exact assembled variable. Their `mean`,
  `variance`, and `std` symbols are available to derived values, guards,
  register updates, and output expressions.
- Every controller output has exactly one typed binding: `signal` names a PMDL
  actuator target, while `external` names a hardware or telemetry pin. Both are
  compiled and traced; only `signal` outputs enter the plant control vector.
- The derived observer can couple many plant states, sensors, controls, and
  latent projections. Observability is calculated per latent, retained in the
  closure, and surfaced as a diagnostic and runtime warning.
- Any PMDL input that can influence the observer must be owned by that
  controller or represented at its explicit boundary. A conservative
  structural dependency proof rejects hidden/unowned influences.

Generated targets implement the same prediction and Joseph-form measurement
update used by the Python runtime.

## Complete controller behavior

Both targets include:

- all declared modes and prioritized transitions;
- persistent registers;
- sequential derived expressions;
- vector observer mean and covariance state plus latent projections;
- output bounds;
- output slew limits; and
- same-tick emergency values with deliberate slew bypass.

Mode transitions take effect on the next tick. Expressions read the previous
registered outputs and registers, while implicit estimates are updated before
derived values and controller decisions, matching offline execution.

## C99 target

The C99 target emits a header/source pair with typed explicit-input, output,
and controller-state structures plus allocation-free `init` and `step`
functions. It uses double-precision scalar arithmetic and the C math library.
PMDL mathematical functions supported by the control runtime have direct C99
lowerings.

Compilation is source generation, not target qualification. Consumers must
compile with their actual ABI/toolchain, execute golden traces, analyze timing
and ranges, and perform fault-injection and hardware-in-the-loop tests.

## Fixed-point Verilog target

The Verilog-2001 target is synchronous and synthesizable. It emits signed
two's-complement fixed-point arithmetic, bounded-loop square root, internal
mode/register/implicit state, a clock, active-low reset, and tick enable. Only
explicit inputs and declared outputs are public data ports.

The default format is Q15.32 in 48 bits and can be changed with
`FixedPointFormat`. Constants and reachable expression intervals are checked
during compilation. Reserved rails signal saturating arithmetic faults;
input, observer, and arithmetic faults hold persistent state instead of
silently wrapping it. Algebraic
primitives, comparisons, conditionals, `abs`, `sqrt`, `min`, `max`, `clip`,
`where`, and `smooth_abs` have synthesizable lowerings. Transcendental
functions such as `sin`, `exp`, or `tanh` are valid in Python and C99 but are
rejected by the Verilog generator unless the controller is rewritten using a
qualified algebraic or lookup-table representation. Nothing is silently
approximated.

Fixed-point source still requires independent quantization-error analysis,
synthesis, timing closure, reset analysis, target execution, and fault
injection. It is not an FPGA bitstream or a safety certificate.

## Reproducibility

`compile_resolved_controller` returns a deterministic `CompilationBundle`. Its
manifest records the control source, assembly, PMDL, controller-link, and
observer digests; operating point and validity; nonlinear approximation and
dynamics-completeness records; selected targets; fixed-point format;
observability diagnostics; and a SHA-256 digest for every generated source.
Each resolved controller is compiled into a separate directory so multiple
controllers never collide or masquerade as a singular global program.

## CLI

From the repository root:

```bash
contraption validate \
  --spec assembled_contraptions/scanner/contraption.json
contraption compile \
  --spec assembled_contraptions/scanner/contraption.json \
  --output outputs/scanner_demo/controllers
```

The compile command emits complete C99 and fixed-point Verilog for every entry
in `ResolvedAssembly.controllers`. A nonlinear PMDL plant may be admitted only
through an explicitly approved local affine observer derivation whose scope and
validity are retained in that manifest. Approval includes a declared
perturbation radius plus an enforced maximum sampled residual remainder over
deterministic coupled directions; it remains local evidence, not a lifetime or global nonlinear error
bound. Smooth transcendental controller
expressions lower to the Python/Torch runtime and C99; generic Verilog rejects
them until a qualified fixed-point lowering exists.
