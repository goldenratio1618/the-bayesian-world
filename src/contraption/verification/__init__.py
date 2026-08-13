"""Strict verification DSL and posterior trajectory evaluator."""

from .dsl import dump_verification, load_verification, parse_verification
from .runtime import (
    CriterionResult,
    MetricResult,
    VerificationReport,
    VerificationRuntime,
    evaluate_verification,
)
from .specs import (
    DifferentiabilityClass,
    TrajectoryMetricSpec,
    VerificationCriterionSpec,
    VerificationError,
    VerificationInputSpec,
    VerificationParameterSpec,
    VerificationProgram,
    VerificationRuntimeError,
    VerificationSpecError,
)

__all__ = [
    "CriterionResult",
    "DifferentiabilityClass",
    "MetricResult",
    "TrajectoryMetricSpec",
    "VerificationCriterionSpec",
    "VerificationError",
    "VerificationInputSpec",
    "VerificationParameterSpec",
    "VerificationProgram",
    "VerificationReport",
    "VerificationRuntime",
    "VerificationRuntimeError",
    "VerificationSpecError",
    "dump_verification",
    "evaluate_verification",
    "load_verification",
    "parse_verification",
]
