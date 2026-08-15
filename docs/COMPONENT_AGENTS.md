# Component ingestion agents

## Trust boundary

Agent output is a proposal in `staging/`, not part of `model_catalog/`. The
classifier proposes a physical domain, category, and device placement. The
modeler proposes catalog-relative `.pmdl`, `.part`, `.model`, JSON, and Markdown
artifacts. Neither can emit executable host code. Safe paths, interface
contracts, DSL grammar, symbol references, units, equation balance, physical
properties, complete initialization, and composition are checked before a
human or automation explicitly calls `promote`.

Classification has an additional deterministic semantic gate. Every proposed
domain must exist and correspond to physics. `reuse_path` must be the exact
category/device ancestry already declared by colocated `interface.pmdl` files.
Unknown identifiers may appear only as one collision-free proposed device below
an existing or proposed category. Empty canonical/category/device values,
project names masquerading as domains, and placements whose required physics
is absent from `domains` are rejected after dispatch and never persisted as
completed proposals.

Promotion does not trust an earlier validation result. It revalidates the live
staging tree, rejects symlinks and special files, copies into a private
snapshot, validates the exact snapshot bytes again, and atomically replaces
each destination file. A staged proposal mutated after its modeling run is
therefore rejected rather than admitted.

## Models and reasoning settings

The requested configuration is preserved verbatim:

* classification: `gpt-5.6-luna`, reasoning effort `medium`, Responses API
  Structured Outputs;
* modeling: `gpt-5.6-luna`, reasoning effort `xhigh`, non-interactive
  `codex exec`, isolated workspace-write sandbox, JSON output schema.

Both identifiers are constructor/CLI settings rather than hard-coded provider
assumptions.

## Dollar limit

`BudgetLedger` defaults to a lifetime limit of `$100.00` for its ledger path.
Before every canary or full call it reserves the worst-case cost at conservative
long-context rates. The reservation is settled from provider token usage. If a
dispatched CLI call fails or does not report usage, its full reservation remains
charged. This is intentionally stricter than best-effort token counting.

The built-in `gpt-5.6-luna` standard-pricing snapshot is dated 2026-08-06:
`$0.20/M` uncached input, `$0.02/M` cached input, `$0.25/M` cache writes,
`$1.20/M` output for short context; `$0.40/M`, `$0.04/M`, `$0.50/M`, and
`$1.80/M` respectively for long context. Because usage payloads do not always
separate cache writes, the ledger charges every uncached input token at the
higher cache-write rate.
Change these explicitly when using another tier, region, model, or price date.

## Required modeling workspace

Every run copies, preserves the catalog-relative source label, then instructs
the model to read fully:

1. `prompts/model_constraints.md`;
2. every authoritative guide in `docs/structured_formats/`, including PMDL,
   physical assembly, shape, optical, control, verification, and derived
   viewer records;
3. representative concrete PMDL, `static.part`, and `vN.model` gold records;
4. all current domain/category/device `interface.pmdl` contracts;
5. only the direct ancestor and concrete-model hierarchy relevant to the item;
6. the full component information record.

The numbered canonical copies and a SHA-256 context manifest live beside the
writable `workspace/`, not inside it. The agent receives their complete text in
`AGENTS.md`; it may write only below `workspace/candidate/`. The host verifies
the protected input/control hashes before dispatch, immediately after Codex
exits, and on error paths.

The guides make the modeler aware of optical power/signal abstractions,
`artifact_ports`, sensor timing, uncertainty, and the standardized artifact
types. Geometry and optical source ingestion remain host-owned. The modeler is
explicitly forbidden to inspect, convert, infer from, or emit CAD, mesh,
texture, image, scan, shape, optical, observation, or reconstruction payloads.

This ownership rule is enforced before ordinary bundle validation. Both the
structured response's artifact list and candidate-file recovery reject
agent-authored `deterministic-part-ingestion-1`,
`deterministic-part-ingestion-staged-1`, `shape-artifact-1`,
`optical-material-1`,
`optical-sensor-1`, `optical-scene-1`, `optical-observation-1`,
`reconstruction-state-1`, CTMESH, and triangle-mesh schema markers. After that
boundary, trusted deterministic host code may append hash-bound shape/optical
outputs; the generic validator deliberately accepts and verifies those
host-owned files.

The modeler is instructed to validate the complete catalog bundle iteratively
with:

```console
python -I -m contraption.part_import.model_validation_tool --bundle candidate
```

That dedicated command rejects absolute paths, traversal, symlinks, unsupported
files, and anything outside `candidate/`. It overlays the proposed files on the
current catalog and validates the full domain/category/device interface tree,
every concrete PMDL contract, every `static.part`, and every `vN.model` hash and
parameter set. Each instantiation must sit below a declared category/device
interface. Its referenced PMDL must implement that exact contract or, for a
device instantiation, the parent category contract when that model is explicitly
listed by the device interface. It checks
protected hashes before and after parsing and never
imports or executes generated host code. Isolated Python mode plus a
trusted-interpreter-first `PATH` prevents workspace modules from shadowing the
installed validator.
Calls are recorded in `validation-calls.jsonl`; after the
agent exits, the host writes `validation-activity.json` with successful/failed
counts and flags more than five calls as a prompt/support-material smell. This
telemetry is not trusted for admission: final structured-output shape checks,
safe materialization, and full artifact validation still run independently.

## Credentials

Only `OPENAI_API_KEY` is read from `.env`; unrelated dotenv values are ignored.
Without `--env-file`, the CLI accepts exactly one `.env` found in the repository
or its parent directory. If both exist, it stops and requires an explicit path
instead of silently selecting credentials. Classification passes the key
directly to the SDK. Modeling admits it through `codex login --with-api-key`
inside a short-lived `CODEX_HOME`, removes the key from the rollout environment,
and destroys that temporary authentication directory after the subprocess
exits. The secret is never written to an agent workspace, durable staging tree,
or ledger. Existing Codex CLI authentication can be used when no key is
provided.

## Guarded full runs

These are paid, non-canary operations. They share
`outputs/agent-budget.json`, whose lifetime ceiling remains `$100.00`:

```console
contraption agent-run classification-all --job-file assembled_contraptions/scanner/agent_jobs.json --env-file ../.env
contraption agent-run modeling-one --job-file assembled_contraptions/scanner/agent_jobs.json --target romi_drive --env-file ../.env
```

`classification-all` processes the records declared by the supplied
`agent-jobs-1` inventory in deterministic authored order and
stops at the first provider or semantic-validation failure. Each completed
record is atomically written to
`outputs/agent-proposals/classification/<target>.json` with the validated
proposal, exact input hash, reported usage, and charged dollars.

`modeling-one` requires a `--target` declared by that same inventory. It
atomically writes a receipt under
`outputs/agent-proposals/modeling/` and leaves all generated, validated files in
`outputs/agent-staging/`. It never calls `promote`.
The proposal receipt includes validation-call telemetry so repeated repair
loops are visible during prompt and reference-quality review.

Both jobs skip a completed receipt only when the exact ordered input bytes and
non-secret model settings hash identically. `--force` deliberately dispatches
again. A matching modeling receipt is skipped only if its staged artifacts
still exist and pass validation. API keys are excluded from hashes, receipts,
workspaces, logs written by the harness, and terminal output.

`contraption doctor` reports the imported Torch version, its compiled CUDA
runtime, CUDA availability, device count, and selected GPU. Torch discovery,
import, metadata, and CUDA runtime failures are represented by explicit error
fields rather than being collapsed into a misleading installed/not-installed
flag.
