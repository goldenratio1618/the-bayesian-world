"""Uncertainty representations and propagation helpers.

Monte Carlo intervals in this module are *pointwise predictive/credible
intervals* of the simulated quantity.  They are not frequentist confidence
intervals for an unknown population mean.  The explicit label is retained in
serialized simulation metadata to avoid the common ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backend import Array, Backend, as_jsonable, get_backend, infer_backend


def split_seed(seed: int, stream: str | int) -> int:
    """Derive a stable independent 63-bit seed for a named random stream."""

    payload = f"contraption:{int(seed)}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class Normal:
    """A reparameterizable independent normal distribution.

    ``lower`` and ``upper`` apply clipping and therefore make the distribution
    a clipped normal rather than an exact truncated normal.  This inexpensive
    choice is intentional for high-throughput engineering Monte Carlo; callers
    needing a true truncated likelihood should provide a custom object with a
    compatible ``sample`` method.
    """

    mean: Any
    std: Any
    lower: Any | None = None
    upper: Any | None = None

    def sample(self, count: int, backend: Backend, rng: Any) -> Array:
        mean = backend.asarray(self.mean)
        std = backend.asarray(self.std)
        std_diagnostic = _numpy_diagnostic(std, backend)
        if np.any(~np.isfinite(std_diagnostic)) or np.any(std_diagnostic < 0.0):
            raise ValueError("Normal std must be finite and nonnegative")
        shape = (int(count),) + tuple(mean.shape)
        draws = mean + std * backend.normal(shape, rng)
        return backend.clip(draws, self.lower, self.upper)


def _numpy_diagnostic(value: Any, backend: Backend) -> np.ndarray:
    """Return a detached host copy used only for distribution validation."""

    if backend.is_torch:
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


@dataclass(frozen=True)
class LogNormal:
    """A positive log-normal distribution parameterized by arithmetic mean.

    ``log_std`` is the standard deviation in log space.  The ``-sigma^2/2``
    shift keeps ``mean`` equal to the arithmetic expectation, which makes a
    PMDL parameter's nominal default remain nominal under Monte Carlo.
    """

    mean: Any
    log_std: Any
    lower: Any | None = None
    upper: Any | None = None

    def sample(self, count: int, backend: Backend, rng: Any) -> Array:
        mean = backend.asarray(self.mean)
        log_std = backend.asarray(self.log_std)
        mean_diagnostic = _numpy_diagnostic(mean, backend)
        std_diagnostic = _numpy_diagnostic(log_std, backend)
        if np.any(~np.isfinite(mean_diagnostic)) or np.any(mean_diagnostic <= 0.0):
            raise ValueError("LogNormal mean must be finite and positive")
        if np.any(~np.isfinite(std_diagnostic)) or np.any(std_diagnostic <= 0.0):
            raise ValueError("LogNormal log_std must be finite and positive")
        shape = (int(count),) + tuple((mean + log_std).shape)
        noise = backend.normal(shape, rng)
        draws = mean * backend.exp(log_std * noise - 0.5 * log_std * log_std)
        return backend.clip(draws, self.lower, self.upper)


@dataclass(frozen=True)
class Uniform:
    """Independent uniform distribution on the closed engineering bounds."""

    lower: Any
    upper: Any

    def sample(self, count: int, backend: Backend, rng: Any) -> Array:
        lower = backend.asarray(self.lower)
        upper = backend.asarray(self.upper)
        lower_diagnostic = _numpy_diagnostic(lower, backend)
        upper_diagnostic = _numpy_diagnostic(upper, backend)
        if (
            np.any(~np.isfinite(lower_diagnostic))
            or np.any(~np.isfinite(upper_diagnostic))
            or np.any(upper_diagnostic <= lower_diagnostic)
        ):
            raise ValueError("Uniform requires finite lower < upper")
        shape = (int(count),) + tuple((lower + upper).shape)
        return lower + (upper - lower) * backend.uniform(shape, rng)


@dataclass(frozen=True)
class Triangular:
    """Independent triangular distribution with lower/mode/upper parameters."""

    lower: Any
    mode: Any
    upper: Any

    def sample(self, count: int, backend: Backend, rng: Any) -> Array:
        lower = backend.asarray(self.lower)
        mode = backend.asarray(self.mode)
        upper = backend.asarray(self.upper)
        lower_np = _numpy_diagnostic(lower, backend)
        mode_np = _numpy_diagnostic(mode, backend)
        upper_np = _numpy_diagnostic(upper, backend)
        if (
            np.any(~np.isfinite(lower_np))
            or np.any(~np.isfinite(mode_np))
            or np.any(~np.isfinite(upper_np))
            or np.any(upper_np <= lower_np)
            or np.any(mode_np < lower_np)
            or np.any(mode_np > upper_np)
        ):
            raise ValueError(
                "Triangular requires finite lower < upper and lower <= mode <= upper"
            )
        span = upper - lower
        shape = (int(count),) + tuple((lower + mode + upper).shape)
        uniform = backend.uniform(shape, rng)
        split = (mode - lower) / span
        rising = lower + backend.sqrt(uniform * span * (mode - lower))
        falling = upper - backend.sqrt((1.0 - uniform) * span * (upper - mode))
        return backend.where(uniform <= split, rising, falling)


@dataclass(frozen=True)
class Empirical:
    """Discrete empirical distribution over scalar or array-valued samples."""

    values: Sequence[Any]
    probabilities: Sequence[float] | None = None

    def sample(self, count: int, backend: Backend, rng: Any) -> Array:
        if len(self.values) == 0:
            raise ValueError("Empirical requires at least one value")
        values = backend.asarray(tuple(self.values))
        if self.probabilities is None:
            probabilities = np.full(len(self.values), 1.0 / len(self.values))
        else:
            probabilities = np.asarray(self.probabilities, dtype=float)
            if probabilities.shape != (len(self.values),):
                raise ValueError("Empirical probabilities must match values")
            if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
                raise ValueError("Empirical probabilities must be finite and nonnegative")
            total = float(probabilities.sum())
            if total <= 0.0:
                raise ValueError("Empirical probabilities must have positive total weight")
            probabilities = probabilities / total
        uniform = backend.uniform((int(count),), rng)
        event_shape = tuple(values.shape[1:])
        result = backend.broadcast_to(values[-1], (int(count),) + event_shape)
        cumulative = np.cumsum(probabilities)
        condition_shape = (int(count),) + (1,) * len(event_shape)
        for index in range(len(self.values) - 2, -1, -1):
            condition = backend.reshape(uniform <= cumulative[index], condition_shape)
            candidate = backend.broadcast_to(values[index], (int(count),) + event_shape)
            result = backend.where(condition, candidate, result)
        return result


@dataclass(frozen=True)
class GaussianParameterDistribution:
    """Correlated Gaussian distribution over scalar named parameters."""

    names: tuple[str, ...]
    mean: Any
    covariance: Any
    lower: Mapping[str, Any] = field(default_factory=dict)
    upper: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("A parameter distribution needs at least one name")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Parameter distribution names must be unique")

    def sample(self, count: int, backend: Backend, rng: Any) -> dict[str, Array]:
        mean = backend.asarray(self.mean)
        covariance = backend.asarray(self.covariance)
        n = len(self.names)
        if tuple(mean.shape) != (n,) or tuple(covariance.shape) != (n, n):
            raise ValueError(
                f"Expected mean {(n,)} and covariance {(n, n)}, got "
                f"{tuple(mean.shape)} and {tuple(covariance.shape)}"
            )
        # The tiny diagonal term handles positive-semidefinite covariance from
        # rounded engineering specifications while remaining negligible.
        scale = backend.maximum(backend.mean(backend.abs(covariance)), backend.asarray(1.0))
        chol = backend.cholesky(covariance + backend.eye(n) * scale * 1e-12)
        noise = backend.normal((int(count), n), rng)
        draws = mean[None, :] + noise @ chol.transpose(-1, -2)
        result: dict[str, Array] = {}
        for index, name in enumerate(self.names):
            result[name] = backend.clip(
                draws[:, index], self.lower.get(name), self.upper.get(name)
            )
        return result


@dataclass(frozen=True)
class DistributionSummary:
    """Pointwise empirical moments and intervals along a sample axis."""

    mean: Array
    covariance: Array
    quantiles: Mapping[float, Array]
    interval: tuple[Array, Array]
    confidence_level: float
    sample_count: int
    interval_kind: str = "pointwise predictive interval"

    @property
    def lower(self) -> Array:
        return self.interval[0]

    @property
    def upper(self) -> Array:
        return self.interval[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": as_jsonable(self.mean),
            "covariance": as_jsonable(self.covariance),
            "quantiles": {str(key): as_jsonable(value) for key, value in self.quantiles.items()},
            "confidence_interval": [as_jsonable(self.lower), as_jsonable(self.upper)],
            "confidence_level": self.confidence_level,
            "interval_kind": self.interval_kind,
            "sample_count": self.sample_count,
        }


def summarize_samples(
    samples: Array,
    *,
    backend: Backend | str | None = None,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
    confidence_level: float = 0.95,
) -> DistributionSummary:
    """Summarize ``[sample, ..., variable]`` observations.

    Covariance has shape ``[..., variable, variable]``.  This keeps a complete
    state/output covariance for every simulated time point without materializing
    cross-time covariance, which would scale quadratically with trajectory
    length.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if backend is None:
        backend = infer_backend(samples)
    else:
        backend = get_backend(backend)
    samples = backend.asarray(samples)
    if len(samples.shape) < 2:
        raise ValueError("samples must have shape [sample, ..., variable]")
    count = int(samples.shape[0])
    if count < 1:
        raise ValueError("At least one Monte Carlo sample is required")
    mean = backend.mean(samples, axis=0)
    centered = samples - mean[None, ...]
    if count == 1:
        covariance = backend.zeros(tuple(samples.shape[1:]) + (int(samples.shape[-1]),))
    else:
        covariance = backend.einsum("s...i,s...j->...ij", centered, centered) / (count - 1)
    requested = sorted(set(float(value) for value in quantiles))
    if any(value < 0.0 or value > 1.0 for value in requested):
        raise ValueError("Quantiles must lie in [0, 1]")
    q_values = {value: backend.quantile(samples, value, axis=0) for value in requested}
    alpha = (1.0 - confidence_level) / 2.0
    lower = backend.quantile(samples, alpha, axis=0)
    upper = backend.quantile(samples, 1.0 - alpha, axis=0)
    return DistributionSummary(
        mean=mean,
        covariance=covariance,
        quantiles=q_values,
        interval=(lower, upper),
        confidence_level=float(confidence_level),
        sample_count=count,
    )


def _distribution_from_specification(
    specification: Any,
    default_mean: Any,
    bounds: tuple[Any | None, Any | None],
) -> Any:
    if hasattr(specification, "sample"):
        return specification
    if isinstance(specification, Mapping):
        distribution = str(specification.get("distribution", specification.get("type", "normal"))).lower()
        nested = specification.get("parameters", {})
        if nested is None:
            nested = {}
        if not isinstance(nested, Mapping):
            raise TypeError("Distribution 'parameters' must be a mapping")
        values = dict(nested)
        values.update(
            (key, value)
            for key, value in specification.items()
            if key not in {"distribution", "type", "parameters"}
        )
        lower = values.get("lower", bounds[0])
        upper = values.get("upper", bounds[1])
        if distribution == "fixed":
            return Empirical((values.get("value", values.get("mean", default_mean)),))
        if distribution == "normal":
            std = values.get("std", values.get("standard_deviation", values.get("sigma")))
            if std is None:
                raise ValueError("Normal distribution requires 'std'")
            return Normal(values.get("mean", default_mean), std, lower, upper)
        if distribution == "lognormal":
            log_std = values.get(
                "log_std", values.get("std", values.get("sigma", values.get("relative_std")))
            )
            if log_std is None:
                raise ValueError("Lognormal distribution requires 'std' (log-space sigma)")
            return LogNormal(values.get("mean", default_mean), log_std, lower, upper)
        if distribution == "uniform":
            if lower is None or upper is None:
                raise ValueError("Uniform distribution requires finite lower and upper")
            return Uniform(lower, upper)
        if distribution == "triangular":
            if lower is None or upper is None:
                raise ValueError("Triangular distribution requires finite lower and upper")
            return Triangular(lower, values.get("mode", default_mean), upper)
        if distribution == "empirical":
            empirical_values = values.get("values", values.get("samples"))
            if empirical_values is None:
                raise ValueError("Empirical distribution requires 'values'")
            return Empirical(empirical_values, values.get("probabilities", values.get("weights")))
        raise ValueError(f"Unsupported parameter distribution {distribution!r}")
    if isinstance(specification, tuple) and len(specification) == 2:
        return Normal(specification[0], specification[1], bounds[0], bounds[1])
    # A scalar specification is interpreted as a standard deviation around the
    # model/default parameter.  This makes ``{"R": 0.2}`` concise and explicit.
    return Normal(default_mean, specification, bounds[0], bounds[1])


def sample_parameters(
    defaults: Mapping[str, Any],
    distributions: Mapping[str, Any] | GaussianParameterDistribution | Any | None,
    count: int,
    *,
    backend: Backend | str | None = None,
    rng: Any | None = None,
    seed: int = 0,
    bounds: Mapping[str, tuple[Any | None, Any | None]] | None = None,
) -> dict[str, Array]:
    """Broadcast nominal parameters and sample selected uncertain parameters."""

    backend = get_backend(backend)
    if count < 1:
        raise ValueError("count must be positive")
    if rng is None:
        rng = backend.make_rng(seed)
    bounds = {} if bounds is None else bounds
    result: dict[str, Array] = {}
    for name, value in defaults.items():
        array = backend.asarray(value)
        if len(array.shape) > 0 and int(array.shape[0]) == count:
            result[name] = array
        else:
            result[name] = backend.broadcast_to(array, (count,) + tuple(array.shape))

    if distributions is None:
        return result
    if hasattr(distributions, "sample") and not isinstance(distributions, Mapping):
        sampled = distributions.sample(count, backend, rng)
        if not isinstance(sampled, Mapping):
            raise TypeError("A joint parameter distribution must sample to a mapping")
        for name, value in sampled.items():
            result[str(name)] = backend.asarray(value)
        return result
    if not isinstance(distributions, Mapping):
        raise TypeError("parameter distributions must be a mapping or joint distribution")
    unknown = set(distributions) - set(defaults)
    if unknown:
        raise KeyError(f"Unknown uncertain parameters: {sorted(unknown)}")
    for name, specification in distributions.items():
        distribution = _distribution_from_specification(
            specification, defaults[name], bounds.get(name, (None, None))
        )
        result[name] = distribution.sample(count, backend, rng)
    return result


def sample_gaussian(
    mean: Any,
    covariance: Any,
    count: int,
    *,
    backend: Backend | str | None = None,
    rng: Any | None = None,
    seed: int = 0,
) -> Array:
    """Reparameterized multivariate-normal samples for state uncertainty."""

    backend = get_backend(backend)
    if rng is None:
        rng = backend.make_rng(seed)
    mean = backend.asarray(mean)
    covariance = backend.asarray(covariance)
    if len(mean.shape) != 1 or tuple(covariance.shape) != (mean.shape[0], mean.shape[0]):
        raise ValueError("mean/covariance shapes must be [D] and [D,D]")
    n = int(mean.shape[0])
    chol = backend.cholesky(covariance + backend.eye(n) * 1e-12)
    return mean[None, :] + backend.normal((int(count), n), rng) @ chol.transpose(-1, -2)


def finite_difference_jacobian(
    function: Callable[[Array], Array],
    point: Array,
    *,
    backend: Backend | str | None = None,
    relative_step: float = 1e-6,
) -> Array:
    """Central-difference Jacobian with graph-preserving torch operations."""

    if backend is None:
        backend = infer_backend(point)
    else:
        backend = get_backend(backend)
    point = backend.asarray(point)
    if len(point.shape) != 1:
        raise ValueError("finite_difference_jacobian currently expects a 1-D point")
    columns = []
    for index in range(int(point.shape[0])):
        basis = backend.zeros(point.shape)
        # Constructing rather than assigning keeps the torch graph uncomplicated.
        basis = backend.stack(
            [backend.asarray(1.0 if j == index else 0.0) for j in range(int(point.shape[0]))]
        )
        step = relative_step * backend.maximum(backend.abs(point[index]), backend.asarray(1.0))
        columns.append((function(point + step * basis) - function(point - step * basis)) / (2 * step))
    return backend.stack(columns, axis=-1)


def linearized_covariance(jacobian: Array, covariance: Array, process_noise: Array | None = None) -> Array:
    """Propagate covariance as ``J P J^T (+ Q)``."""

    result = jacobian @ covariance @ jacobian.transpose(-1, -2)
    return result if process_noise is None else result + process_noise


def ekf_predict(
    mean: Array,
    covariance: Array,
    transition: Callable[[Array], Array],
    process_covariance: Array,
    *,
    jacobian: Array | None = None,
    backend: Backend | str | None = None,
) -> tuple[Array, Array]:
    """One extended-Kalman prediction step for a differentiable transition."""

    if backend is None:
        backend = infer_backend(mean)
    else:
        backend = get_backend(backend)
    predicted = transition(mean)
    if jacobian is None:
        jacobian = finite_difference_jacobian(transition, mean, backend=backend)
    return predicted, linearized_covariance(jacobian, covariance, process_covariance)


def ekf_update(
    mean: Array,
    covariance: Array,
    observation: Array,
    observation_function: Callable[[Array], Array],
    observation_covariance: Array,
    *,
    jacobian: Array | None = None,
    backend: Backend | str | None = None,
) -> tuple[Array, Array]:
    """Joseph-form EKF measurement update, stable for small covariances."""

    if backend is None:
        backend = infer_backend(mean)
    else:
        backend = get_backend(backend)
    expected = observation_function(mean)
    if jacobian is None:
        jacobian = finite_difference_jacobian(observation_function, mean, backend=backend)
    innovation_covariance = jacobian @ covariance @ jacobian.transpose(-1, -2) + observation_covariance
    gain = backend.solve(innovation_covariance, jacobian @ covariance).transpose(-1, -2)
    updated_mean = mean + gain @ (observation - expected)
    identity = backend.eye(int(covariance.shape[-1]))
    residual_map = identity - gain @ jacobian
    updated_covariance = (
        residual_map @ covariance @ residual_map.transpose(-1, -2)
        + gain @ observation_covariance @ gain.transpose(-1, -2)
    )
    return updated_mean, updated_covariance


def monte_carlo_propagate(
    function: Callable[[Array], Array],
    mean: Any,
    covariance: Any,
    count: int = 1024,
    *,
    seed: int = 0,
    backend: Backend | str | None = None,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
) -> tuple[Array, DistributionSummary]:
    """Propagate a Gaussian through an arbitrary vectorized function."""

    backend = get_backend(backend)
    draws = sample_gaussian(mean, covariance, count, backend=backend, seed=seed)
    outputs = function(draws)
    return outputs, summarize_samples(outputs, backend=backend, quantiles=quantiles)


__all__ = [
    "DistributionSummary",
    "Empirical",
    "GaussianParameterDistribution",
    "LogNormal",
    "Normal",
    "Triangular",
    "Uniform",
    "ekf_predict",
    "ekf_update",
    "finite_difference_jacobian",
    "linearized_covariance",
    "monte_carlo_propagate",
    "sample_gaussian",
    "sample_parameters",
    "split_seed",
    "summarize_samples",
]
