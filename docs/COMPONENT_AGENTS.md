# Component ingestion agents

## Trust boundary

Agent output is a proposal in `staging/`, not part of the component registry.
The classifier can propose taxonomy placement; the modeler can propose `.pmdl`,
JSON, and Markdown artifacts. Neither can emit executable host code. Safe paths,
JSON shape, DSL grammar, symbol references, units, equation balance, physical
properties, bounds, initialization, and composition are checked before a human
or automation explicitly calls `promote`.

Classification has an additional deterministic semantic gate. Every proposed
domain must exist. `reuse_path` must be the exact, contiguous ancestry from an
existing category through its deepest reused subcategory. Unknown identifiers
are forbidden in that path and may appear only in `new_nodes`, which must be a
collision-free, parent-valid acyclic extension of the reused path. `category`
must name the root and `subcategory` a selected terminal existing or proposed
node. Empty canonical/category/subcategory values and proposals whose category
physics is missing from `domains` are rejected after dispatch and charged from
reported usage, but are never persisted as completed proposals.

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

Every run copies, then instructs the model to read fully:

1. `prompts/model_constraints.md`;
2. one electrical and one rigid-mechanical gold hierarchy;
3. the complete current taxonomy;
4. only the direct ancestor/current-instance hierarchy relevant to the item;
5. the full component information record.

The numbered canonical copies and a SHA-256 context manifest live beside the
writable `workspace/`, not inside it. The agent receives their complete text in
`AGENTS.md`; it may write only below `workspace/candidate/`. The host verifies
the protected input/control hashes before dispatch, immediately after Codex
exits, and on error paths.

The modeler is instructed to validate each PMDL draft iteratively with:

```console
python -I -m contraption.model_validation_tool candidate/<name>.pmdl
```

That dedicated command rejects absolute paths, traversal, symlinks, non-PMDL
files, and anything outside `candidate/`. It checks protected hashes before and
after parsing, then returns sorted issue codes, schema paths, and messages from
the safe PMDL parser and deterministic validator. It never imports or executes
generated host code. Isolated Python mode plus a trusted-interpreter-first
`PATH` prevents workspace modules from shadowing the installed validator.
Calls are recorded in `validation-calls.jsonl`; after the
agent exits, the host writes `validation-activity.json` with successful/failed
counts and flags more than five calls as a prompt/support-material smell. This
telemetry is not trusted for admission: final structured-output shape checks,
safe materialization, and full artifact validation still run independently.

## Credentials

Only `OPENAI_API_KEY` is read from `.env`; unrelated dotenv values are ignored.
Without `--env-file`, the CLI accepts exactly one `.env` found in the repository
or its parent directory. If both exist, it stops and requires an explicit path
instead of silently selecting credentials. The secret is passed through the
child environment and never written to an agent workspace or ledger. Existing
Codex CLI authentication can be used for the modeling run when the environment
does not provide a key.

## Guarded full runs

These are paid, non-canary operations. They share
`outputs/agent-budget.json`, whose lifetime ceiling remains `$100.00`:

```console
contraption agent-run classification-all --env-file ../.env
contraption agent-run modeling-one --target romi_drive --env-file ../.env
```

`classification-all` processes the six records in
`examples/scanner_robot/component_inputs` in deterministic filename order and
stops at the first provider or semantic-validation failure. Each completed
record is atomically written to
`outputs/agent-proposals/classification/<target>.json` with the validated
proposal, exact input hash, reported usage, and charged dollars.

`modeling-one` defaults to `romi_drive`; the other five component-input stems
are accepted by `--target`. It atomically writes a receipt under
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
