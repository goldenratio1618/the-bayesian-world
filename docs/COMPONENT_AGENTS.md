# Component ingestion agents

## Trust boundary

Classifier and modeler output is untrusted until deterministic host validation.
The classifier proposes a physical domain, category, and device placement. The
modeler proposes catalog-relative `.pmdl`, `.part`, and `.model` artifacts;
host-owned overlays may add procurement, fabrication, shape, and derived
Markdown records. Neither agent can emit executable host code. Safe paths,
interface contracts, DSL grammar, symbol references, units, equation balance,
physical properties, complete initialization, and composition are checked
before promotion. `modeling-one` leaves a staged proposal for explicit
promotion. The isolated ingestion canary/batch revalidates and promotes into
only its clean replay catalog, then reloads the promoted part/model registry;
staged-only output never counts as fully ingested.

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

The guarded ingestion path uses `gpt-5.6-luna` at reasoning effort `low` for
both classification and modeling. Classification uses Responses Structured
Outputs. Modeling uses a direct Responses Structured Output with complete file
content strings and no tools; the host alone materializes and validates it.
A validation failure may receive up to two compact correction turns, for three
modeling responses total. The model identifier, effort, schemas, selected
context bytes, and prompts participate in the workflow or input hashes.

`ModelingAgent` retains the non-interactive `codex exec` workspace backend for
legacy compatibility and `modeling-one`. It also defaults to Luna-low, stages
candidate files, and receives the full structured-format guide set. It is not
the canary/batch backend and is not required for the paid replay.

## Dollar limit

The general ledger default remains `$100.00`; ingestion canary/batch commands
default `--ledger-limit-usd` to `$0.50` and may use a dedicated replay ledger.
Before inference, the direct path requires the Responses input-token counter
and reserves the exact counted input plus its full output allowance. A request
without that counter fails before dispatch. Classification and every modeling
attempt share one atomic per-part cost scope. The worst permitted envelope is:

```text
classification (12k input, 2k output) +
3 × modeling (20k input, 8k output) = $0.0492 < $0.05
```

Provider usage settles the reservation without capping measured overages. A
dispatched response with absent or malformed usage is charged its full
reservation. Provider-reported cache-write tokens are accounted separately;
when that field is unavailable, every non-cached input token is conservatively
priced as a cache write and the event is labeled as an estimate.

The built-in `gpt-5.6-luna` standard-pricing snapshot was verified against the
[official OpenAI Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
on 2026-08-20: `$0.20/M` input, `$0.02/M` cached input,
and `$1.20/M` output. Prompt-cache writes cost 1.25 times ordinary input
(`$0.25/M`). Requests above 272,000 input tokens use twice the input/cached
rates and 1.5 times the output rate; cache writes remain 1.25 times the
applicable input rate. The guarded importer caps all requests far below that
threshold. Change the table explicitly for another tier, region, model, or
price date.

Run reports compute `total importer ledger cost / freshly promoted completed
parts`. The numerator includes classification, all modeling attempts, failures,
and retries. Bound prior failed-run ledgers are included in the final aggregate;
their target/component hash and exact bytes are reverified, and their spend
reduces that target's remaining pre-dispatch `$0.05` scope. Deferred
unsupported-physics parts are zero dispatch and excluded from the denominator. A gate passes only when cost per
completed part is strictly below `$0.05`, average failed validation attempts
per completed part is strictly below `1`, every requested target is reported,
and no scope breach occurred. Batch execution requires a matching passing
one-part canary and records combined canary-plus-batch-plus-carryover KPIs.

The three preserved failed canaries for `yageo_rc0603_10r` bind `$0.02134939`
and nine failed Luna modeling validations, leaving `$0.02865061` of cumulative
target headroom. Thus a successful ten-part replay has a historical failure
average of `0.9`; any additional invalid output makes the `<1` KPI impossible.
Reports distinguish that historical Luna result (`0/9` host-valid modeling
responses) from host-deterministic recipe completions through
`generation_mode`, `provider_calls`, recipe/input hashes, and deterministic
validation telemetry.

## Required modeling workspace

Before dispatch, the host builds a deterministic `IMPORT_PLAN.json`. It names
the exact target ID, published parameter facts, eligible reusable PMDL
identities with canonical hashes, preferred family model, immutable-base
policy, and attempt limit. The legacy CLI backend copies and preserves the
catalog-relative source label for this context:

1. `prompts/model_constraints.md`;
2. the authoritative guides in `docs/structured_formats/` selected for the
   component's declared domains and evidenced payload types;
3. representative concrete PMDL, `static.part`, and `vN.model` gold records;
4. only domain/category/device `interface.pmdl` ancestors governing the target;
5. only the direct ancestor and concrete-model hierarchy relevant to the item;
6. the full component information record; and
7. normalized, verified deterministic extraction JSON when host-owned document
   or design-file ingestion produced relevant textual evidence.

The direct Responses backend deliberately excludes the CLI constraints and
format guides because they contain candidate/tool instructions that do not
apply to a no-tools response. Its lean context contains `IMPORT_PLAN.json`,
record-shape examples, governing interfaces, direct ancestors, the component
record, and normalized deterministic extraction evidence. A direct-only system
prompt supplies the complete output and trust-boundary contract. The exact
backend-selected context set is also used for the resumable input hash.

The numbered canonical copies and a SHA-256 context manifest live beside the
host staging `workspace/`. `IMPORT_PLAN.json` is protected. The CLI agent may
write only below `workspace/candidate/`; the direct agent has no filesystem or
tools and returns complete artifact strings. The host verifies protected
input/control hashes around CLI execution. Raw PDF,
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

The legacy CLI modeler validates its candidate bundle iteratively with:

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
Calls are recorded in `validation-calls.jsonl`. A CLI modeling workspace may make
at most three calls: a fourth request is deterministically refused without
being appended to the call log. After the agent exits, the host writes
`validation-activity.json` with successful/failed counts. This telemetry is not
trusted for admission: safe materialization and full host validation still run
independently. The direct model never sees or invokes this command. The host
materializes each structured response into an attempt workspace, performs the
same validation, quarantines any incomplete published proposal before a retry,
and exposes only a bounded diagnostic to the next response.

For the legacy CLI backend, the candidate tree is the artifact authority and
the final structured response is only a path manifest with `content: null`.
A valid candidate can therefore be recovered after a nonzero CLI exit. For the
direct backend, each artifact carries its complete UTF-8 content and the host
is the sole artifact writer. Proposed files that are byte-identical to the base
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
instead of silently selecting credentials. Classification and direct modeling
pass the key directly to the OpenAI SDK. The legacy CLI backend admits it
through `codex login --with-api-key` inside a short-lived `CODEX_HOME`, removes
the key from the rollout environment, and destroys that temporary authentication
directory after the subprocess exits. The secret is never written to an agent
workspace, durable staging tree, report, or ledger. The `agents` optional
dependency requires `openai>=2.53.0`, the first locally verified minimum with
the Responses input-token counter used by the hard preflight gate.

## Guarded full runs

Legacy classification and staged modeling share `outputs/agent-budget.json`,
whose lifetime ceiling remains `$100.00`:

```console
contraption agent-run classification-all --job-file outputs/scanner-part-import/agent_jobs.json --env-file ../.env
contraption agent-run modeling-one --job-file outputs/scanner-part-import/agent_jobs.json --target romi_drive --env-file ../.env
```

The measured 20-part replay uses a fresh isolated output root and dedicated
`$0.50` ledger. Durable zero-event ledger binding and stable host-recipe hashes
change the workflow fingerprint, so the passing v4 canary cannot gate a batch;
v5 must start from a new empty replay root. Run exactly one eligible target
first:

```console
contraption agent-run ingestion-canary \
  --job-file outputs/part-import-2026-08-18/agent_jobs.json \
  --target yageo_rc0603_10r \
  --output-root outputs/part-import-2026-08-20-luna-direct-replay-v5 \
  --ledger outputs/part-import-2026-08-20-luna-direct-replay-v5/agent-budget.json \
  --prior-failed-ledger outputs/part-import-2026-08-19-luna-direct-replay/agent-budget.json \
  --prior-failed-ledger outputs/part-import-2026-08-19-luna-direct-replay-v2/agent-budget.json \
  --prior-failed-ledger outputs/part-import-2026-08-19-luna-direct-replay-v3/agent-budget.json \
  --ledger-limit-usd 0.50 \
  --env-file ../.env
```

For this inventory's ten exact Yageo RC0603 family records, the host selects a
generic versioned fail-closed physical recipe before creating either provider
client. It
requires the exact ideal-resistor PMDL identity/hash and p/n ports, explicit
0603 package and XYZ dimensions, one finite positive resistance, and an exact
tolerance. The recipe emits only a primitive rectangular bounding envelope,
estimated p/n frames at opposing X-face centers, and fabrication records marked
missing for conductor and termination. It copies no manufacturer, product,
offer, or URL into `.part`/`.model`; the adjacent `.procurement` record remains
host-extracted from the protected input. A missing or conflicting recipe fact
fails with zero classification or modeling dispatch; it never falls back to
Luna. Other eligible
families and genuine new-model imports continue through the general direct
Responses path.

Only after `ingestion-canary-report.json` passes may the remaining 19 targets
run against that same isolated catalog and ledger:

```console
contraption agent-run ingestion-batch \
  --job-file outputs/part-import-2026-08-18/agent_jobs.json \
  --canary-report outputs/part-import-2026-08-20-luna-direct-replay-v5/ingestion-canary-report.json \
  --output-root outputs/part-import-2026-08-20-luna-direct-replay-v5 \
  --ledger outputs/part-import-2026-08-20-luna-direct-replay-v5/agent-budget.json \
  --prior-failed-ledger outputs/part-import-2026-08-19-luna-direct-replay/agent-budget.json \
  --prior-failed-ledger outputs/part-import-2026-08-19-luna-direct-replay-v2/agent-budget.json \
  --prior-failed-ledger outputs/part-import-2026-08-19-luna-direct-replay-v3/agent-budget.json \
  --ledger-limit-usd 0.50 \
  --env-file ../.env
```

The canary creates `isolation-manifest.json`, copies the catalog and job inputs,
identity/hash-checks then removes only the ten existing eligible resistor
targets, preserves the shared ideal-resistor PMDL, and leaves the source catalog
untouched. Ten thermal targets remain deterministic zero-dispatch deferrals.
Eligible direct proposals are promoted only into the isolated catalog. Batch
stops after the first target failure and the expected-target check prevents a
partial result list from passing.

The canary replay root must be absent or empty, must be a strict child of
`outputs/`, and may not overlap the source-job run. Its ledger and staging paths
are fixed at `agent-budget.json` and `agent-staging/` inside that root; symlinks
and special files are rejected. Constructing a new canary ledger atomically
persists its canonical empty JSON immediately, even when the deterministic path
makes zero provider calls. Both canary and batch reports bind the exact ledger
path and byte digest, and the canary replay-state fingerprint carries the same
binding. Batch requires that regular non-symlink file to exist and match before
constructing a ledger or doing gate/work; deletion, mutation, or replacement is
fail-closed and never synthesizes a fresh ledger. The canary report binds the exact 20-target
inventory, canary target, isolation manifest, copied job and complete component
asset closure, post-promotion catalog/data tree, all host `contraption` Python
source bytes, and OpenAI SDK version. Batch verifies those bindings before any
write or dispatch and requires exactly the remaining 19 targets.

`classification-all` processes the records declared by the supplied
`agent-jobs-1` inventory in deterministic authored order and
stops at the first provider or semantic-validation failure. Each completed
record is atomically written to
`agent-proposals/classification/<target>.json` below the directory containing
the selected job inventory, with the validated
proposal, exact input hash, reported usage, and charged dollars.

`modeling-one` requires a `--target` declared by that same inventory. It
atomically writes a receipt under
that run's `agent-proposals/modeling/` directory and leaves all generated,
validated files in the sibling `agent-staging/`. It never calls `promote`.
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
