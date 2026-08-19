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
assumptions. `xhigh` remains the production modeling default. The preserved
20-input replay may explicitly override modeling to `low` as a Luna-low
stress/cost benchmark; the chosen effort is included in input hashes and
receipts. It is not an apples-to-apples reliability comparison with the
recorded `xhigh` baseline (12 paid dispatches, 22 validator calls, 6 of 10
targets passing on their first paid-dispatch validator call, and
`$2.23477088` charged), because reasoning effort and replay isolation differ.

## Dollar limit

`BudgetLedger` defaults to a lifetime limit of `$100.00` for its ledger path.
Before every canary or full call it reserves the worst-case cost at conservative
long-context rates. The reservation is settled from provider token usage, even
when the CLI exits nonzero. A dispatched failure without usable token usage keeps
its full reservation charged. The sole post-dispatch zero-cost exception is a
structurally parsed Codex JSONL `turn.failed` provider rejection with the exact
`invalid_request_error` / `invalid_json_schema` / HTTP 400 /
`text.format.schema` tuple, and only when there is no usage, completed agent
message or output file, candidate write, validator activity, malformed event, or
other failure event. Stderr text and matching substrings are never proof. Such an
event is recorded as `provider_rejected_before_inference` with
`cost_basis: proven_pre_inference_zero`; every ambiguous case remains a full
conservative debit. This is intentionally stricter than best-effort token
counting.

The built-in `gpt-5.6-luna` standard-pricing snapshot is dated 2026-08-06:
`$0.20/M` uncached input, `$0.02/M` cached input, `$0.25/M` cache writes,
`$1.20/M` output for short context; `$0.40/M`, `$0.04/M`, `$0.50/M`, and
`$1.80/M` respectively for long context. Because usage payloads do not always
separate cache writes, the ledger charges every uncached input token at the
higher cache-write rate.
Change these explicitly when using another tier, region, model, or price date.

## Required modeling workspace

Before dispatch, the host builds a deterministic `IMPORT_PLAN.json`. It names
the exact target ID and catalog root, published parameter facts, eligible
reusable PMDL identities with canonical hashes, the preferred family model,
the immutable-base policy, and the three-call validation limit. Every run then
copies and preserves the catalog-relative source label for only the context
relevant to that import:

1. `prompts/model_constraints.md`;
2. the authoritative guides in `docs/structured_formats/` selected for the
   component's declared domains and evidenced payload types;
3. representative concrete PMDL, `static.part`, and `vN.model` gold records;
4. only domain/category/device `interface.pmdl` ancestors governing the target;
5. only the direct ancestor and concrete-model hierarchy relevant to the item;
6. the full component information record; and
7. normalized, verified deterministic extraction JSON when host-owned document
   or design-file ingestion produced relevant textual evidence.

The numbered canonical copies and a SHA-256 context manifest live beside the
writable `workspace/`, not inside it. `IMPORT_PLAN.json` is also protected. The
agent receives the complete selected text in `AGENTS.md`; it may write only
below `workspace/candidate/`. The host verifies protected input/control hashes
before dispatch, immediately after Codex exits, and on error paths. Raw PDF,
ECAD, archive, CAD, mesh, and other binary/source design payloads never enter
the Luna workspace; deterministic host code validates and normalizes them
first, and only its extraction JSON may become modeling context.

Relevant guides make the modeler aware of optical power/signal abstractions,
`artifact_ports`, sensor timing, uncertainty, and standardized artifact types
when those concepts apply. Geometry, optical, document, archive, CAD, and ECAD
source ingestion remain host-owned. The modeler is explicitly forbidden to
inspect, convert, infer from, or emit those raw payloads.

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
Calls are recorded in `validation-calls.jsonl`. A modeling workspace may make
at most three calls: a fourth request is deterministically refused without
being appended to the call log. After the agent exits, the host writes
`validation-activity.json` with successful/failed counts. This telemetry is not
trusted for admission: safe materialization and full host validation still run
independently.

The candidate tree is the artifact authority. Luna's final structured response
is only a path manifest: strict output entries carry the path and a required
`content: null` placeholder, never a second transcription of the bytes. A valid
candidate can therefore be recovered even when the CLI exits nonzero or its
last message is malformed. Proposed files that are byte-identical to the base
catalog are stripped, while any changed file colliding with a base-catalog path
is rejected. For new `.model` files, the host parses the referenced canonical
PMDL and repairs a stale or missing PMDL hash to that exact identity before the
final validation pass; it rejects unknown identities and duplicate model
definitions rather than guessing.

### Derived standalone part documentation

After Luna's catalog proposal passes host validation, trusted host code renders
a deterministic `README.md` in every proposed part-instantiation directory. Luna
is instructed not to create that reserved file, and materialization rejects a
Luna-authored copy. The host then validates the complete proposal again with the
generated Markdown present before it can become `proposed/` or be promoted.

The renderer loads the proposal through a full catalog overlay, so it documents
the same resolved contracts and models that admission checks. Each README is
standalone: it explains the part/model distinction, acausal residual convention,
SI-unit convention, and validation limitations; describes the complete parent
domain/category/device interface chain; and separates executable declarations
from prose desires or explanations. It includes the physical part, provenance,
connectors, equations in display math plus original DSL, inline expression
comments, parameters, uncertainty, modes, transitions, properties, validity,
notes, evidence, trust labels, and metadata.

A model index and one detailed section per `vN.model` preserve every current or
future model hypothesis for the component. Both the variant and the human PMDL
model name are shown, together with the exact model id, version, and hash. The
source manifest is catalog-relative and excludes the generated README itself,
so identical validated inputs produce identical Markdown bytes on every host.
Use `contraption part-markdown --part-directory PATH` to regenerate one validated
catalog part explicitly.

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

Before a modeling reservation, deterministic preflight checks the component's
required physics. Components requiring an unimplemented domain such as thermal
physics are written as `deferred_unsupported_physics` receipts with
`charged_usd: 0.0`; they do not reserve budget, authenticate Codex, or dispatch
the modeling agent. `--force` does not bypass this physics boundary.

`contraption doctor` reports the imported Torch version, its compiled CUDA
runtime, CUDA availability, device count, and selected GPU. Torch discovery,
import, metadata, and CUDA runtime failures are represented by explicit error
fields rather than being collapsed into a misleading installed/not-installed
flag.
