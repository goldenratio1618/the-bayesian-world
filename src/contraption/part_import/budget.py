"""Crash-resistant dollar budgeting for external model calls.

The ledger reserves a worst-case amount *before* a request is dispatched.  A
reservation is settled from provider usage afterwards; if usage is unavailable
(for example a terminated CLI process), the entire reservation is charged.
This is deliberately conservative so a sequence of locally coordinated runs
cannot exceed its configured dollar ceiling.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator


class BudgetExceeded(RuntimeError):
    """Raised before dispatch when a reservation would exceed the budget."""


@dataclass(frozen=True)
class ProvenPreInferenceProviderRejection:
    """Typed proof authorizing the one post-dispatch zero-cost settlement.

    Codex event parsing lives at the process boundary, but the ledger owns the
    accounting invariant.  Requiring this exact proof object prevents callers
    from obtaining a zero settlement with an arbitrary status string or an
    unstructured diagnostic.  Every absence claim is deliberately explicit.
    """

    source: str
    terminal_event: str
    provider_error_type: str
    provider_error_code: str
    provider_status: int
    provider_param: str
    usage_observed: bool
    completed_agent_message_observed: bool
    candidate_artifact_observed: bool
    validator_activity_observed: bool
    malformed_event_observed: bool
    unknown_failure_event_observed: bool

    def __post_init__(self) -> None:
        exact_provider_error = (
            self.source == "codex_jsonl"
            and self.terminal_event == "turn.failed"
            and self.provider_error_type == "invalid_request_error"
            and self.provider_error_code == "invalid_json_schema"
            and self.provider_status == 400
            and not isinstance(self.provider_status, bool)
            and self.provider_param == "text.format.schema"
        )
        forbidden_activity = (
            self.usage_observed
            or self.completed_agent_message_observed
            or self.candidate_artifact_observed
            or self.validator_activity_observed
            or self.malformed_event_observed
            or self.unknown_failure_event_observed
        )
        if not exact_provider_error or forbidden_activity:
            raise ValueError(
                "pre-inference provider-rejection proof does not establish an "
                "exact zero-usage invalid_json_schema rejection"
            )


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cache_write_input_tokens: int | None = None

    def __post_init__(self) -> None:
        counts = (self.input_tokens, self.cached_input_tokens, self.output_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("token counts must be integers")
        if min(counts) < 0:
            raise ValueError("token counts cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input cannot exceed total input")
        if self.cache_write_input_tokens is not None:
            cache_writes = self.cache_write_input_tokens
            if (
                isinstance(cache_writes, bool)
                or not isinstance(cache_writes, int)
                or cache_writes < 0
            ):
                raise ValueError(
                    "cache-write input tokens must be a nonnegative integer or null"
                )
            if self.cached_input_tokens + cache_writes > self.input_tokens:
                raise ValueError(
                    "cached and cache-write input cannot exceed total input"
                )


@dataclass(frozen=True)
class TokenPricing:
    """Prices in USD per one million tokens.

    Defaults are the standard gpt-5.6-luna prices verified on 2026-08-19.
    Long-context rates are selected whenever total input exceeds the threshold.
    Values remain configurable because prices and account processing tiers can
    change independently of this code.
    """

    input_per_million: float = 0.20
    cached_input_per_million: float = 0.02
    cache_write_per_million: float = 0.25
    output_per_million: float = 1.20
    long_input_per_million: float = 0.40
    long_cached_input_per_million: float = 0.04
    long_cache_write_per_million: float = 0.50
    long_output_per_million: float = 1.80
    long_context_threshold: int = 272_000

    def __post_init__(self) -> None:
        rates = (
            self.input_per_million,
            self.cached_input_per_million,
            self.cache_write_per_million,
            self.output_per_million,
            self.long_input_per_million,
            self.long_cached_input_per_million,
            self.long_cache_write_per_million,
            self.long_output_per_million,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in rates
        ):
            raise ValueError("token prices must be finite nonnegative numbers")
        if (
            isinstance(self.long_context_threshold, bool)
            or not isinstance(self.long_context_threshold, int)
            or self.long_context_threshold <= 0
        ):
            raise ValueError("long_context_threshold must be a positive integer")

    def cost(self, usage: Usage, *, force_long: bool = False) -> float:
        long = force_long or usage.input_tokens > self.long_context_threshold
        input_rate = self.long_input_per_million if long else self.input_per_million
        cached_rate = (
            self.long_cached_input_per_million if long else self.cached_input_per_million
        )
        cache_write_rate = (
            self.long_cache_write_per_million if long else self.cache_write_per_million
        )
        output_rate = self.long_output_per_million if long else self.output_per_million
        cache_writes = usage.cache_write_input_tokens
        if cache_writes is None:
            # Some provider surfaces report cached reads but not cache writes.
            # Treat every other input token as a possible cache write. The
            # ledger labels this as an estimate rather than an invoice charge.
            cache_writes = usage.input_tokens - usage.cached_input_tokens
        ordinary_input = (
            usage.input_tokens - usage.cached_input_tokens - cache_writes
        )
        return (
            ordinary_input * input_rate
            + cache_writes * cache_write_rate
            + usage.cached_input_tokens * cached_rate
            + usage.output_tokens * output_rate
        ) / 1_000_000.0

    @staticmethod
    def cost_basis(usage: Usage) -> str:
        return (
            "estimated_from_provider_tokens_cache_writes_reported"
            if usage.cache_write_input_tokens is not None
            else "estimated_from_provider_tokens_cache_writes_unknown_conservative"
        )

    def worst_case(self, *, max_input_tokens: int, max_output_tokens: int) -> float:
        if min(max_input_tokens, max_output_tokens) < 0:
            raise ValueError("worst-case token counts cannot be negative")
        long = max_input_tokens > self.long_context_threshold
        input_rate = self.long_input_per_million if long else self.input_per_million
        cache_write_rate = (
            self.long_cache_write_per_million
            if long
            else self.cache_write_per_million
        )
        output_rate = self.long_output_per_million if long else self.output_per_million
        # Assume no cached reads and price every input token at the more
        # expensive of ordinary input and a new prompt-cache write.
        return (
            max_input_tokens * max(input_rate, cache_write_rate)
            + max_output_tokens * output_rate
        ) / 1_000_000.0


class BudgetLedger:
    """A small JSON ledger protected by an exclusive lock file."""

    def __init__(self, path: str | Path, limit_usd: float = 100.0) -> None:
        if (
            isinstance(limit_usd, bool)
            or not isinstance(limit_usd, (int, float))
            or not math.isfinite(float(limit_usd))
            or limit_usd <= 0
        ):
            raise ValueError("limit_usd must be a finite positive number")
        self.path = Path(path)
        self.limit_usd = float(limit_usd)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._persist_initial_state()

    def _persist_initial_state(self) -> None:
        """Create a durable empty ledger before any reservation is possible."""

        if self.path.is_symlink():
            raise ValueError(f"budget ledger cannot be a symlink: {self.path}")
        with self._locked():
            if self.path.is_symlink():
                raise ValueError(f"budget ledger cannot be a symlink: {self.path}")
            if not self.path.exists():
                self._write(self._default())
                return
            if not self.path.is_file():
                raise ValueError(f"budget ledger must be a regular file: {self.path}")
            # Existing ledgers are validation-only at construction time. A
            # lower requested ceiling applies in memory and is persisted only
            # with the next explicit reservation/settlement state change.
            self._read()

    @contextmanager
    def _locked(self, timeout: float = 10.0) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"budget ledger lock is busy: {self.lock_path}")
                time.sleep(0.025)
        try:
            yield
        finally:
            os.close(fd)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _default(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "limit_usd": self.limit_usd,
            "spent_usd": 0.0,
            "reserved": {},
            "events": [],
        }

    def _read(self) -> dict[str, Any]:
        if self.path.is_symlink():
            raise ValueError(f"budget ledger cannot be a symlink: {self.path}")
        if not self.path.exists():
            raise FileNotFoundError(f"budget ledger disappeared: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"budget ledger must be a regular file: {self.path}")
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("unsupported budget ledger schema")
        persisted_limit = float(state["limit_usd"])
        spent = float(state["spent_usd"])
        if not math.isfinite(persisted_limit) or persisted_limit <= 0:
            raise ValueError("persisted budget limit must be finite and positive")
        if not math.isfinite(spent) or spent < 0:
            raise ValueError("persisted budget spend must be finite and nonnegative")
        reserved = state.get("reserved")
        events = state.get("events")
        if not isinstance(reserved, dict) or not isinstance(events, list):
            raise ValueError("persisted budget reservations/events are malformed")
        for item in reserved.values():
            if not isinstance(item, dict):
                raise ValueError("persisted budget reservation is malformed")
            amount = float(item.get("max_usd", float("nan")))
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError(
                    "persisted budget reservation must be finite and positive"
                )
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("persisted budget event is malformed")
            charged = float(event.get("charged_usd", float("nan")))
            if not math.isfinite(charged) or charged < 0:
                raise ValueError(
                    "persisted budget event charge must be finite and nonnegative"
                )
        # The constructor may reduce a budget but never silently increase an
        # existing ledger. A deliberate reset requires a new ledger path.
        state["limit_usd"] = min(persisted_limit, self.limit_usd)
        return state

    def _write(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _reserved_total(state: dict[str, Any]) -> float:
        return sum(float(v["max_usd"]) for v in state["reserved"].values())

    @staticmethod
    def _scope_committed(state: dict[str, Any], scope: str) -> float:
        charged = sum(
            float(event.get("charged_usd", 0.0))
            for event in state["events"]
            if isinstance(event.get("metadata"), dict)
            and event["metadata"].get("cost_scope") == scope
        )
        reserved = sum(
            float(item["max_usd"])
            for item in state["reserved"].values()
            if isinstance(item.get("metadata"), dict)
            and item["metadata"].get("cost_scope") == scope
        )
        return charged + reserved

    def reserve(
        self,
        run_id: str,
        max_usd: float,
        metadata: dict[str, Any],
        *,
        cost_scope: str | None = None,
        cost_scope_limit_usd: float | None = None,
    ) -> None:
        if (
            not run_id
            or isinstance(max_usd, bool)
            or not isinstance(max_usd, (int, float))
            or not math.isfinite(float(max_usd))
            or max_usd <= 0
        ):
            raise ValueError("run_id and a positive max_usd are required")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if (cost_scope is None) != (cost_scope_limit_usd is None):
            raise ValueError(
                "cost_scope and cost_scope_limit_usd must be provided together"
            )
        stored_metadata = dict(metadata)
        if cost_scope is not None:
            if not isinstance(cost_scope, str) or not cost_scope:
                raise ValueError("cost_scope must be a nonempty string")
            if (
                not isinstance(cost_scope_limit_usd, (int, float))
                or isinstance(cost_scope_limit_usd, bool)
                or not math.isfinite(float(cost_scope_limit_usd))
                or not 0 < float(cost_scope_limit_usd)
            ):
                raise ValueError("cost_scope_limit_usd must be positive")
            if (
                "cost_scope" in stored_metadata
                or "cost_scope_limit_usd" in stored_metadata
            ):
                raise ValueError("cost scope metadata is ledger-owned")
            stored_metadata["cost_scope"] = cost_scope
            stored_metadata["cost_scope_limit_usd"] = float(cost_scope_limit_usd)
        with self._locked():
            state = self._read()
            if run_id in state["reserved"]:
                raise ValueError(f"duplicate run id: {run_id}")
            if cost_scope is not None:
                scoped_committed = self._scope_committed(state, cost_scope)
                # The requirement is strictly less than the limit, so a
                # reservation that could reach it is rejected atomically.
                if (
                    scoped_committed + max_usd
                    >= float(cost_scope_limit_usd) - 1e-12
                ):
                    raise BudgetExceeded(
                        f"part scope {cost_scope!r} would reserve "
                        f"${scoped_committed + max_usd:.6f}; strict limit is "
                        f"<${float(cost_scope_limit_usd):.6f}"
                    )
            committed = float(state["spent_usd"]) + self._reserved_total(state)
            if committed + max_usd > float(state["limit_usd"]) + 1e-12:
                raise BudgetExceeded(
                    f"request reserves ${max_usd:.6f}; only "
                    f"${float(state['limit_usd']) - committed:.6f} remains"
                )
            state["reserved"][run_id] = {
                "max_usd": float(max_usd),
                "metadata": stored_metadata,
                "created_unix": time.time(),
            }
            self._write(state)

    def settle(
        self,
        run_id: str,
        *,
        usage: Usage | None,
        pricing: TokenPricing,
        status: str = "completed",
    ) -> float:
        with self._locked():
            state = self._read()
            reservation = state["reserved"].pop(run_id, None)
            if reservation is None:
                raise KeyError(f"unknown reservation: {run_id}")
            actual = (
                pricing.cost(usage)
                if usage is not None
                else float(reservation["max_usd"])
            )
            if not math.isfinite(actual) or actual < 0:
                raise ValueError("settled charge must be finite and nonnegative")
            usage_exceeded_reservation = (
                actual > float(reservation["max_usd"]) + 1e-9
            )
            if usage_exceeded_reservation:
                # Never understate measured usage. The event is a hard alarm;
                # future reservations in the same scope will be denied.
                status = "usage_exceeded_reservation"
            state["spent_usd"] = float(state["spent_usd"]) + actual
            scope = reservation["metadata"].get("cost_scope")
            scope_limit = reservation["metadata"].get("cost_scope_limit_usd")
            scope_total = (
                self._scope_committed(state, scope) + actual
                if isinstance(scope, str)
                else None
            )
            state["events"].append(
                {
                    "run_id": run_id,
                    "status": status,
                    "charged_usd": actual,
                    "usage": asdict(usage) if usage is not None else None,
                    "cost_basis": (
                        "estimated_from_provider_tokens_exceeded_reservation"
                        if usage_exceeded_reservation
                        else pricing.cost_basis(usage)
                        if usage is not None
                        else "full_reservation_conservative"
                    ),
                    "scope_total_usd": scope_total,
                    "scope_limit_breached": bool(
                        scope_total is not None
                        and scope_total >= float(scope_limit)
                    ),
                    "metadata": reservation["metadata"],
                    "settled_unix": time.time(),
                }
            )
            self._write(state)
            return actual

    def settle_proven_pre_inference_provider_rejection(
        self,
        run_id: str,
        *,
        proof: ProvenPreInferenceProviderRejection,
    ) -> float:
        """Release a reservation only for a typed, exact pre-inference rejection."""

        if type(proof) is not ProvenPreInferenceProviderRejection:
            raise TypeError("a ProvenPreInferenceProviderRejection is required")
        # Re-run validation even though the frozen dataclass validated at
        # construction. This keeps the ledger boundary fail-closed if a caller
        # bypasses normal construction through low-level object manipulation.
        ProvenPreInferenceProviderRejection.__post_init__(proof)
        with self._locked():
            state = self._read()
            reservation = state["reserved"].pop(run_id, None)
            if reservation is None:
                raise KeyError(f"unknown reservation: {run_id}")
            state["events"].append(
                {
                    "run_id": run_id,
                    "status": "provider_rejected_before_inference",
                    "charged_usd": 0.0,
                    "usage": None,
                    "cost_basis": "proven_pre_inference_zero",
                    "proof": asdict(proof),
                    "metadata": reservation["metadata"],
                    "settled_unix": time.time(),
                }
            )
            self._write(state)
        return 0.0

    def cancel(self, run_id: str, reason: str) -> None:
        with self._locked():
            state = self._read()
            reservation = state["reserved"].pop(run_id, None)
            if reservation is None:
                return
            state["events"].append(
                {
                    "run_id": run_id,
                    "status": "cancelled_before_dispatch",
                    "charged_usd": 0.0,
                    "cost_basis": "not_dispatched_zero",
                    "reason": reason,
                    "metadata": reservation["metadata"],
                    "settled_unix": time.time(),
                }
            )
            self._write(state)

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            state = self._read()
        state["reserved_usd"] = self._reserved_total(state)
        state["remaining_usd"] = (
            float(state["limit_usd"])
            - float(state["spent_usd"])
            - float(state["reserved_usd"])
        )
        return state
