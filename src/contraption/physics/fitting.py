"""Parameter inference and approximate Bayesian model comparison.

The core fitter has no SciPy dependency.  NumPy uses a bounded damped
Gauss--Newton/Levenberg--Marquardt iteration with central-difference
sensitivities.  The optional PyTorch path optimizes through the exact same
differentiable simulator used for prediction.

Model-selection scores are intentionally explicit approximations.  ``bic`` is
the Schwarz/BIC unit-information approximation; ``laplace`` is a local Gaussian
(Laplace) approximation around the fitted mode.  Neither is reported as an
exact marginal likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .backend import Array, Backend, as_jsonable, get_backend
from .simulator import SimulationResult, _parameter_bounds, _parameter_defaults, simulate


@dataclass(frozen=True)
class ExperimentalData:
    """Time-aligned experimental measurements used by a fit.

    Observations may be a mapping of state/output name to ``[time]`` values, or
    a ``[time, observed_variable]`` array accompanied by ``observation_names``.
    A scalar/array/mapping ``observation_std`` whitens residuals.  Missing NumPy
    observations encoded as NaN are ignored.
    """

    time: Any
    observations: Any
    observation_names: tuple[str, ...] = ()
    controls: Any = None
    initial_state: Any = None
    observation_std: Any = 1.0

    def names(self) -> tuple[str, ...]:
        if isinstance(self.observations, Mapping):
            return tuple(str(name) for name in self.observations)
        if not self.observation_names:
            raise ValueError("Array observations require observation_names")
        return tuple(self.observation_names)


@dataclass(frozen=True)
class FitOptions:
    """Numerical options shared by NumPy and Torch fitting paths."""

    max_iterations: int = 120
    tolerance: float = 1e-8
    relative_step: float = 1e-5
    damping: float = 1e-3
    learning_rate: float = 0.04
    backend: str | Backend = "numpy"
    device: str | None = None
    integrator: str = "auto"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0 or self.relative_step <= 0:
            raise ValueError("tolerance and relative_step must be positive")


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters, local covariance, diagnostics, and likelihood."""

    parameters: Mapping[str, Any]
    covariance: Array
    parameter_names: tuple[str, ...]
    residuals: Array
    loss_history: tuple[float, ...]
    converged: bool
    iterations: int
    method: str
    log_likelihood: float
    observation_count: int
    message: str = ""

    @property
    def loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def standard_errors(self) -> Mapping[str, float]:
        covariance = np.asarray(as_jsonable(self.covariance), dtype=float)
        if covariance.size == 0:
            return {}
        return {
            name: float(math.sqrt(max(covariance[index, index], 0.0)))
            for index, name in enumerate(self.parameter_names)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": as_jsonable(dict(self.parameters)),
            "parameter_names": list(self.parameter_names),
            "covariance": as_jsonable(self.covariance),
            "standard_errors": self.standard_errors,
            "residuals": as_jsonable(self.residuals),
            "loss_history": list(self.loss_history),
            "loss": self.loss,
            "converged": self.converged,
            "iterations": self.iterations,
            "method": self.method,
            "log_likelihood": self.log_likelihood,
            "observation_count": self.observation_count,
            "message": self.message,
        }


@dataclass(frozen=True)
class CandidateModel:
    """One candidate and the parameters allowed to adapt for comparison."""

    name: str
    model: Any
    initial_parameters: Mapping[str, Any] = field(default_factory=dict)
    parameter_names: tuple[str, ...] | None = None
    bounds: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    prior_std: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvidence:
    """Approximate evidence and fit diagnostics for one candidate."""

    name: str
    score: float
    posterior_probability: float
    criterion: str
    approximation_label: str
    fit: FitResult
    bic: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "posterior_probability": self.posterior_probability,
            "criterion": self.criterion,
            "approximation_label": self.approximation_label,
            "bic": self.bic,
            "fit": self.fit.to_dict(),
        }


@dataclass(frozen=True)
class ModelSelectionResult:
    """Normalized relative weights among the supplied candidate set."""

    evidences: Mapping[str, ModelEvidence]
    best_model: str
    criterion: str
    approximation_label: str

    @property
    def posterior_probabilities(self) -> dict[str, float]:
        return {name: evidence.posterior_probability for name, evidence in self.evidences.items()}

    @property
    def ranking(self) -> tuple[str, ...]:
        return tuple(
            sorted(self.evidences, key=lambda name: self.evidences[name].score, reverse=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.best_model,
            "criterion": self.criterion,
            "approximation_label": self.approximation_label,
            "ranking": list(self.ranking),
            "posterior_probabilities": self.posterior_probabilities,
            "models": {name: evidence.to_dict() for name, evidence in self.evidences.items()},
        }


def _observation_matrix(data: ExperimentalData, backend: Backend) -> tuple[tuple[str, ...], Array]:
    names = data.names()
    if isinstance(data.observations, Mapping):
        columns = [backend.asarray(data.observations[name]) for name in names]
        matrix = backend.stack(columns, axis=-1)
    else:
        matrix = backend.asarray(data.observations)
        if len(matrix.shape) == 1 and len(names) == 1:
            matrix = matrix[:, None]
    if len(matrix.shape) != 2 or int(matrix.shape[-1]) != len(names):
        raise ValueError("observations must have shape [time, observation_name]")
    if int(matrix.shape[0]) != len(data.time):
        raise ValueError("Observation time dimension does not match data.time")
    return names, matrix


def _standard_deviation_matrix(
    data: ExperimentalData,
    names: tuple[str, ...],
    shape: tuple[int, int],
    backend: Backend,
) -> Array:
    standard_deviation = data.observation_std
    if isinstance(standard_deviation, Mapping):
        columns = []
        for name in names:
            value = backend.asarray(standard_deviation.get(name, 1.0))
            if len(value.shape) == 0:
                value = backend.broadcast_to(value, (shape[0],))
            columns.append(value)
        matrix = backend.stack(columns, axis=-1)
    else:
        matrix = backend.asarray(standard_deviation)
        if len(matrix.shape) == 0:
            matrix = backend.broadcast_to(matrix, shape)
        elif len(matrix.shape) == 1:
            if int(matrix.shape[0]) == shape[1]:
                matrix = backend.broadcast_to(matrix[None, :], shape)
            elif shape[1] == 1 and int(matrix.shape[0]) == shape[0]:
                matrix = matrix[:, None]
    if tuple(matrix.shape) != shape:
        raise ValueError(f"observation_std cannot broadcast to {shape}")
    # Use backend.maximum so a tensor-valued standard deviation remains live.
    return backend.maximum(matrix, backend.asarray(1e-15))


def predict_experiment(
    model: Any,
    data: ExperimentalData,
    parameters: Mapping[str, Any],
    *,
    backend: str | Backend = "numpy",
    device: str | None = None,
    integrator: str = "auto",
) -> tuple[SimulationResult, Array, Array, Array]:
    """Simulate a model at experimental timestamps and construct residuals."""

    numerical = get_backend(backend, device=device)
    names, observed = _observation_matrix(data, numerical)
    result = simulate(
        model,
        times=data.time,
        controls=data.controls,
        parameters=parameters,
        initial_state=data.initial_state,
        num_samples=1,
        backend=numerical,
        integrator=integrator,
        use_model_uncertainty=False,
        process_noise=False,
    )
    predicted = numerical.stack(
        [result.series(name, outputs_first=True)[0] for name in names], axis=-1
    )
    standard_deviation = _standard_deviation_matrix(
        data, names, tuple(map(int, observed.shape)), numerical
    )
    residual = (predicted - observed) / standard_deviation
    return result, predicted, observed, residual


def _numpy_residual_vector(
    model: Any,
    data: ExperimentalData,
    parameters: Mapping[str, Any],
    integrator: str,
) -> np.ndarray:
    # Trial steps can temporarily enter stiff/ill-conditioned regions.  Treat a
    # non-finite prediction as a very poor candidate without leaking numerical
    # warnings or changing the residual vector's dimension.  NaN *observations*
    # remain the documented missing-data marker and are removed consistently.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _, _, observed, residual = predict_experiment(
            model, data, parameters, backend="numpy", integrator=integrator
        )
    vector = np.asarray(residual, dtype=float).reshape(-1)
    observation_vector = np.asarray(observed, dtype=float).reshape(-1)
    vector = vector[~np.isnan(observation_vector)]
    vector = np.nan_to_num(vector, nan=1e50, posinf=1e50, neginf=-1e50)
    return np.clip(vector, -1e50, 1e50)


def _log_likelihood_from_residual(residual: np.ndarray) -> float:
    n = int(residual.size)
    if n == 0:
        raise ValueError("No finite experimental observations")
    variance = max(float(residual @ residual) / n, 1e-30)
    return -0.5 * n * (math.log(2.0 * math.pi * variance) + 1.0)


def _bounds_for(
    model: Any,
    parameter_names: tuple[str, ...],
    overrides: Mapping[str, tuple[float | None, float | None]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = _parameter_bounds(model)
    if overrides:
        bounds.update(overrides)
    lower = np.asarray(
        [
            -np.inf if bounds.get(name, (None, None))[0] is None else bounds[name][0]
            for name in parameter_names
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            np.inf if bounds.get(name, (None, None))[1] is None else bounds[name][1]
            for name in parameter_names
        ],
        dtype=float,
    )
    if np.any(lower >= upper):
        raise ValueError("Every fitting bound must satisfy lower < upper")
    return lower, upper


def _fit_numpy(
    model: Any,
    data: ExperimentalData,
    base_parameters: Mapping[str, Any],
    names: tuple[str, ...],
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    options: FitOptions,
) -> FitResult:
    lower, upper = _bounds_for(model, names, bounds)
    point = np.asarray([base_parameters[name] for name in names], dtype=float)
    point = np.clip(point, lower, upper)

    def parameters_at(value: np.ndarray) -> dict[str, Any]:
        result = dict(base_parameters)
        result.update({name: value[index] for index, name in enumerate(names)})
        return result

    residual = _numpy_residual_vector(model, data, parameters_at(point), options.integrator)
    loss = 0.5 * float(residual @ residual)
    history = [loss]
    damping = options.damping
    converged = len(names) == 0
    message = "No free parameters" if converged else "Maximum iterations reached"
    jacobian = np.empty((residual.size, len(names)), dtype=float)
    iteration = 0
    for iteration in range(1, options.max_iterations + 1):
        if not names:
            break
        for index in range(len(names)):
            step = options.relative_step * max(abs(point[index]), 1.0)
            above = point.copy()
            below = point.copy()
            above[index] = min(point[index] + step, upper[index])
            below[index] = max(point[index] - step, lower[index])
            denominator = above[index] - below[index]
            if denominator <= 0:
                jacobian[:, index] = 0.0
            else:
                jacobian[:, index] = (
                    _numpy_residual_vector(model, data, parameters_at(above), options.integrator)
                    - _numpy_residual_vector(model, data, parameters_at(below), options.integrator)
                ) / denominator
        gradient = jacobian.T @ residual
        hessian = jacobian.T @ jacobian
        scale = np.maximum(np.diag(hessian), 1.0)
        try:
            step_vector = -np.linalg.solve(
                hessian + damping * np.diag(scale), gradient
            )
        except np.linalg.LinAlgError:
            step_vector = -np.linalg.pinv(hessian + damping * np.diag(scale)) @ gradient
        candidate = np.clip(point + step_vector, lower, upper)
        candidate_residual = _numpy_residual_vector(
            model, data, parameters_at(candidate), options.integrator
        )
        candidate_loss = 0.5 * float(candidate_residual @ candidate_residual)
        if candidate_loss < loss:
            improvement = loss - candidate_loss
            point, residual, loss = candidate, candidate_residual, candidate_loss
            history.append(loss)
            damping = max(damping / 3.0, 1e-12)
            if np.linalg.norm(step_vector) <= options.tolerance * (
                np.linalg.norm(point) + options.tolerance
            ) or improvement <= options.tolerance * max(1.0, loss):
                converged = True
                message = "Converged"
                break
        else:
            damping = min(damping * 10.0, 1e18)
            history.append(loss)
            if damping >= 1e17:
                message = "Damping overflow: no improving step found"
                break

    fitted = {name: float(point[index]) for index, name in enumerate(names)}
    if names:
        # Refresh the Jacobian at the accepted point for covariance reporting.
        for index in range(len(names)):
            step = options.relative_step * max(abs(point[index]), 1.0)
            above, below = point.copy(), point.copy()
            above[index] = min(point[index] + step, upper[index])
            below[index] = max(point[index] - step, lower[index])
            denominator = above[index] - below[index]
            jacobian[:, index] = (
                _numpy_residual_vector(model, data, parameters_at(above), options.integrator)
                - _numpy_residual_vector(model, data, parameters_at(below), options.integrator)
            ) / max(denominator, np.finfo(float).eps)
        dof = max(residual.size - len(names), 1)
        residual_variance = float(residual @ residual) / dof
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * residual_variance
    else:
        covariance = np.empty((0, 0), dtype=float)
    return FitResult(
        parameters=fitted,
        covariance=covariance,
        parameter_names=names,
        residuals=residual,
        loss_history=tuple(history),
        converged=converged,
        iterations=iteration,
        method="bounded Levenberg-Marquardt (NumPy)",
        log_likelihood=_log_likelihood_from_residual(residual),
        observation_count=int(residual.size),
        message=message,
    )


def _fit_torch(
    model: Any,
    data: ExperimentalData,
    base_parameters: Mapping[str, Any],
    names: tuple[str, ...],
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    options: FitOptions,
) -> FitResult:
    numerical = get_backend("torch", device=options.device)
    torch = numerical.torch
    lower_np, upper_np = _bounds_for(model, names, bounds)
    initial = numerical.stack(
        [numerical.asarray(base_parameters[name]) for name in names], axis=0
    )
    point = torch.nn.Parameter(initial.clone())
    if not names:
        _, _, _, residual_matrix = predict_experiment(
            model,
            data,
            base_parameters,
            backend=numerical,
            integrator=options.integrator,
        )
        residual = residual_matrix.reshape(-1)
        residual_np = numerical.to_numpy(residual)
        return FitResult(
            {},
            np.empty((0, 0)),
            (),
            residual,
            (0.5 * float(residual_np @ residual_np),),
            True,
            0,
            "Adam/autograd (PyTorch)",
            _log_likelihood_from_residual(residual_np),
            int(residual.numel()),
            "No free parameters",
        )
    optimizer = torch.optim.Adam([point], lr=options.learning_rate)
    lower = numerical.asarray(lower_np)
    upper = numerical.asarray(upper_np)
    history: list[float] = []
    converged = False
    message = "Maximum iterations reached"

    def parameter_mapping(value: Array) -> dict[str, Any]:
        result = dict(base_parameters)
        result.update({name: value[index] for index, name in enumerate(names)})
        return result

    previous = math.inf
    iteration = 0
    for iteration in range(1, options.max_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        _, _, _, residual_matrix = predict_experiment(
            model,
            data,
            parameter_mapping(point),
            backend=numerical,
            integrator=options.integrator,
        )
        residual = residual_matrix.reshape(-1)
        loss = 0.5 * torch.sum(residual * residual)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            point.copy_(torch.maximum(torch.minimum(point, upper), lower))
            current = float(loss.detach().cpu())
        history.append(current)
        if abs(previous - current) <= options.tolerance * max(1.0, current):
            converged = True
            message = "Converged"
            break
        previous = current

    def residual_from_vector(value: Array) -> Array:
        _, _, _, matrix = predict_experiment(
            model,
            data,
            parameter_mapping(value),
            backend=numerical,
            integrator=options.integrator,
        )
        return matrix.reshape(-1)

    residual = residual_from_vector(point)
    jacobian = torch.autograd.functional.jacobian(residual_from_vector, point, create_graph=False)
    dof = max(int(residual.numel()) - len(names), 1)
    variance = torch.sum(residual * residual) / dof
    covariance = torch.linalg.pinv(jacobian.transpose(-1, -2) @ jacobian) * variance
    residual_np = numerical.to_numpy(residual)
    fitted = {name: point[index].detach().clone() for index, name in enumerate(names)}
    return FitResult(
        parameters=fitted,
        covariance=covariance.detach(),
        parameter_names=names,
        residuals=residual.detach(),
        loss_history=tuple(history),
        converged=converged,
        iterations=iteration,
        method="Adam/autograd (PyTorch)",
        log_likelihood=_log_likelihood_from_residual(residual_np),
        observation_count=int(residual.numel()),
        message=message,
    )


def fit_parameters(
    model: Any,
    data: ExperimentalData,
    initial_parameters: Mapping[str, Any] | None = None,
    *,
    parameter_names: Sequence[str] | None = None,
    fixed_parameters: Mapping[str, Any] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    options: FitOptions | None = None,
    backend: str | Backend | None = None,
    **option_overrides: Any,
) -> FitResult:
    """Fit selected component/model parameters to experimental trajectories."""

    if options is not None and option_overrides:
        raise ValueError("Pass FitOptions or option keyword overrides, not both")
    if options is None:
        if backend is not None:
            option_overrides["backend"] = backend
        options = FitOptions(**option_overrides)
    elif backend is not None:
        raise ValueError("backend is already specified by FitOptions")
    defaults = _parameter_defaults(model)
    supplied = {} if initial_parameters is None else dict(initial_parameters)
    defaults.update(supplied)
    if fixed_parameters:
        defaults.update(fixed_parameters)
    if parameter_names is None:
        parameter_names = tuple(supplied) if supplied else tuple(defaults)
    names = tuple(str(name) for name in parameter_names)
    missing = set(names) - set(defaults)
    if missing:
        raise KeyError(f"No initial/default value for parameters {sorted(missing)}")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")
    numerical = get_backend(options.backend, device=options.device)
    if numerical.is_torch:
        return _fit_torch(model, data, defaults, names, bounds, options)
    return _fit_numpy(model, data, defaults, names, bounds, options)


def _coerce_candidates(candidates: Mapping[str, Any] | Sequence[CandidateModel]) -> list[CandidateModel]:
    if isinstance(candidates, Mapping):
        result = []
        for name, value in candidates.items():
            if isinstance(value, CandidateModel):
                result.append(value)
            elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
                result.append(CandidateModel(str(name), value[0], value[1]))
            else:
                # A bare model is treated as fixed.  Requiring an explicit
                # CandidateModel to fit parameters avoids silently rewarding a
                # candidate whose flexibility the caller did not intend.
                result.append(CandidateModel(str(name), value, {}, ()))
        return result
    return list(candidates)


def select_models(
    candidates: Mapping[str, Any] | Sequence[CandidateModel],
    data: ExperimentalData,
    *,
    criterion: str = "bic",
    fit_options: FitOptions | None = None,
) -> ModelSelectionResult:
    """Fit and rank candidate probabilistic component models.

    Returned probabilities are normalized *relative weights over only the
    supplied candidates*.  They inherit the approximation assumptions and are
    not calibrated posterior probabilities over omitted model families.
    """

    criterion = criterion.lower()
    if criterion not in {"bic", "laplace"}:
        raise ValueError("criterion must be 'bic' or 'laplace'")
    candidate_list = _coerce_candidates(candidates)
    if not candidate_list:
        raise ValueError("At least one candidate model is required")
    if len({candidate.name for candidate in candidate_list}) != len(candidate_list):
        raise ValueError("Candidate names must be unique")
    options = fit_options or FitOptions()
    fits: dict[str, FitResult] = {}
    raw_scores: dict[str, float] = {}
    bics: dict[str, float] = {}
    for candidate in candidate_list:
        fit = fit_parameters(
            candidate.model,
            data,
            candidate.initial_parameters,
            parameter_names=candidate.parameter_names,
            bounds=candidate.bounds,
            options=options,
        )
        fits[candidate.name] = fit
        parameter_count = len(fit.parameter_names)
        observation_count = max(fit.observation_count, 1)
        bic = parameter_count * math.log(observation_count) - 2.0 * fit.log_likelihood
        bics[candidate.name] = bic
        if criterion == "bic":
            raw_scores[candidate.name] = -0.5 * bic
        else:
            covariance = np.asarray(as_jsonable(fit.covariance), dtype=float)
            if parameter_count:
                sign, logdet_covariance = np.linalg.slogdet(covariance + np.eye(parameter_count) * 1e-18)
                if sign <= 0 or not np.isfinite(logdet_covariance):
                    logdet_covariance = -math.inf
                log_prior = 0.0
                for index, name in enumerate(fit.parameter_names):
                    center = float(candidate.initial_parameters.get(name, _parameter_defaults(candidate.model)[name]))
                    scale = float(candidate.prior_std.get(name, max(abs(center), 1.0) * 10.0))
                    fitted_value = float(as_jsonable(fit.parameters[name]))
                    log_prior += -0.5 * ((fitted_value - center) / scale) ** 2 - math.log(
                        scale * math.sqrt(2.0 * math.pi)
                    )
                raw_scores[candidate.name] = (
                    fit.log_likelihood
                    + log_prior
                    + 0.5 * parameter_count * math.log(2.0 * math.pi)
                    + 0.5 * logdet_covariance
                )
            else:
                raw_scores[candidate.name] = fit.log_likelihood
    maximum = max(raw_scores.values())
    weights = {name: math.exp(score - maximum) for name, score in raw_scores.items()}
    normalization = sum(weights.values())
    probabilities = {name: value / normalization for name, value in weights.items()}
    label = (
        "BIC approximation (unit-information prior)"
        if criterion == "bic"
        else "Laplace approximation around fitted mode"
    )
    evidences = {
        name: ModelEvidence(
            name=name,
            score=raw_scores[name],
            posterior_probability=probabilities[name],
            criterion=criterion,
            approximation_label=label,
            fit=fits[name],
            bic=bics[name],
        )
        for name in raw_scores
    }
    best = max(raw_scores, key=raw_scores.get)
    return ModelSelectionResult(evidences, best, criterion, label)


__all__ = [
    "CandidateModel",
    "ExperimentalData",
    "FitOptions",
    "FitResult",
    "ModelEvidence",
    "ModelSelectionResult",
    "fit_parameters",
    "predict_experiment",
    "select_models",
]
