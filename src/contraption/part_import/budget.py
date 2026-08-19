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
import os
from pathlib import Path
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

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.cached_input_tokens, self.output_tokens) < 0:
            raise ValueError("token counts cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input cannot exceed total input")


@dataclass(frozen=True)
class TokenPricing:
    """Prices in USD per one million tokens.

    Defaults are the standard gpt-5.6-luna prices published on 2026-08-06.
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

    def cost(self, usage: Usage, *, force_long: bool = False) -> float:
        long = force_long or usage.input_tokens >= self.long_context_threshold
        input_rate = self.long_input_per_million if long else self.input_per_million
        cached_rate = (
            self.long_cached_input_per_million if long else self.cached_input_per_million
        )
        cache_write_rate = (
            self.long_cache_write_per_million if long else self.cache_write_per_million
        )
        output_rate = self.long_output_per_million if long else self.output_per_million
        uncached = usage.input_tokens - usage.cached_input_tokens
        # Provider usage does not consistently separate new prompt-cache writes.
        # Treat every uncached input token at the more expensive of read/input or
        # cache-write pricing. This overcharges the local ledger but preserves the
        # promised hard upper bound.
        conservative_input_rate = max(input_rate, cache_write_rate)
        return (
            uncached * conservative_input_rate
            + usage.cached_input_tokens * cached_rate
            + usage.output_tokens * output_rate
        ) / 1_000_000.0

    def worst_case(self, *, max_input_tokens: int, max_output_tokens: int) -> float:
        # Always reserve at long-context, uncached rates. This also covers cache
        # writes, which are lower than this output-rate upper bound.
        return (
            max_input_tokens
            * max(self.long_input_per_million, self.long_cache_write_per_million)
            + max_output_tokens * self.long_output_per_million
        ) / 1_000_000.0


class BudgetLedger:
    """A small JSON ledger protected by an exclusive lock file."""

    def __init__(self, path: str | Path, limit_usd: float = 100.0) -> None:
        if limit_usd <= 0:
            raise ValueError("limit_usd must be positive")
        self.path = Path(path)
        self.limit_usd = float(limit_usd)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

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
        if not self.path.exists():
            return self._default()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("unsupported budget ledger schema")
        # The constructor may reduce a budget but never silently increase an
        # existing ledger. A deliberate reset requires a new ledger path.
        state["limit_usd"] = min(float(state["limit_usd"]), self.limit_usd)
        return state

    def _write(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _reserved_total(state: dict[str, Any]) -> float:
        return sum(float(v["max_usd"]) for v in state["reserved"].values())

    def reserve(self, run_id: str, max_usd: float, metadata: dict[str, Any]) -> None:
        if not run_id or max_usd <= 0:
            raise ValueError("run_id and a positive max_usd are required")
        with self._locked():
            state = self._read()
            if run_id in state["reserved"]:
                raise ValueError(f"duplicate run id: {run_id}")
            committed = float(state["spent_usd"]) + self._reserved_total(state)
            if committed + max_usd > float(state["limit_usd"]) + 1e-12:
                raise BudgetExceeded(
                    f"request reserves ${max_usd:.6f}; only "
                    f"${float(state['limit_usd']) - committed:.6f} remains"
                )
            state["reserved"][run_id] = {
                "max_usd": float(max_usd),
                "metadata": metadata,
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
            usage_exceeded_reservation = (
                actual > float(reservation["max_usd"]) + 1e-9
            )
            if usage_exceeded_reservation:
                # This should be impossible when callers reserve at declared
                # maxima. Charge the reservation and flag it rather than hiding
                # an accounting invariant violation.
                actual = float(reservation["max_usd"])
                status = "usage_exceeded_reservation"
            state["spent_usd"] = float(state["spent_usd"]) + actual
            state["events"].append(
                {
                    "run_id": run_id,
                    "status": status,
                    "charged_usd": actual,
                    "usage": asdict(usage) if usage is not None else None,
                    "cost_basis": (
                        "reported_usage_capped_at_reservation"
                        if usage_exceeded_reservation
                        else "reported_usage"
                        if usage is not None
                        else "full_reservation_conservative"
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
