"""General multi-view inverse optical problem solver built on Torch autograd."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .renderer import CompiledScene, RuntimeScene
from .schemas import ObservationArtifact, OpticalSensor, Pose
from .torch_backend import TorchOpticalBackend, TorchScene, require_torch


class InverseProblemError(ValueError):
    """Raised when an inverse problem is under-specified or numerically invalid."""


@dataclass(frozen=True, slots=True)
class Prior:
    distribution: str = "normal"
    location: float = 0.0
    scale: float = 1.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.distribution not in {"normal", "lognormal", "laplace", "uniform"}:
            raise InverseProblemError("prior distribution must be normal/lognormal/laplace/uniform")
        if not math.isfinite(self.location) or not math.isfinite(self.scale) or self.scale <= 0:
            raise InverseProblemError("prior location/scale must be finite and scale positive")
        if self.distribution == "uniform":
            if self.lower is None or self.upper is None or not self.lower < self.upper:
                raise InverseProblemError("uniform priors require lower < upper")

    def negative_log_probability(self, value: Any) -> Any:
        torch = require_torch()
        if self.distribution == "normal":
            standardized = (value - self.location) / self.scale
            return 0.5 * (standardized * standardized).sum() + value.numel() * math.log(self.scale)
        if self.distribution == "lognormal":
            safe = torch.clamp(value, min=torch.finfo(value.dtype).tiny)
            standardized = (torch.log(safe) - self.location) / self.scale
            penalty = 0.5 * standardized * standardized + torch.log(safe) + math.log(self.scale)
            invalid = torch.where(value > 0, torch.zeros_like(value), torch.full_like(value, 1e12))
            return (penalty + invalid).sum()
        if self.distribution == "laplace":
            return (torch.abs(value - self.location) / self.scale).sum() + value.numel() * math.log(self.scale)
        assert self.lower is not None and self.upper is not None
        violation = torch.relu(self.lower - value) + torch.relu(value - self.upper)
        return (violation * 1e12).sum() + value.numel() * math.log(self.upper - self.lower)


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """A trainable slice of one physical scene/camera/sensor tensor.

    Target names are discovered from ``OpticalInverseProblem.available_targets``.
    ``selector`` uses normal Torch indexing and permits an arbitrary subset of a
    geometry, material, light, or per-view camera tensor to be inferred.
    """

    name: str
    target: str
    initial: Any | None = None
    selector: Any | None = None
    transform: str = "identity"
    lower: float | None = None
    upper: float | None = None
    prior: Prior | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.target:
            raise InverseProblemError("inverse parameters require name and target")
        if self.transform not in {"identity", "positive", "bounded", "unit_vector"}:
            raise InverseProblemError("parameter transform must be identity/positive/bounded/unit_vector")
        if self.transform == "bounded" and (self.lower is None or self.upper is None or not self.lower < self.upper):
            raise InverseProblemError("bounded transforms require lower < upper")
        if self.transform == "positive" and self.lower is not None and not math.isfinite(self.lower):
            raise InverseProblemError("positive-transform lower bound must be finite")

    def constrain(self, raw: Any) -> Any:
        torch = require_torch()
        if self.transform == "identity":
            return raw
        if self.transform == "positive":
            return torch.nn.functional.softplus(raw) + (0.0 if self.lower is None else self.lower)
        if self.transform == "bounded":
            assert self.lower is not None and self.upper is not None
            return self.lower + (self.upper - self.lower) * torch.sigmoid(raw)
        return torch.nn.functional.normalize(raw, dim=-1)

    def unconstrain(self, constrained: Any) -> Any:
        torch = require_torch()
        epsilon = torch.finfo(constrained.dtype).eps
        if self.transform in {"identity", "unit_vector"}:
            return constrained.clone()
        if self.transform == "positive":
            shifted = torch.clamp(constrained - (0.0 if self.lower is None else self.lower), min=epsilon)
            return torch.log(torch.expm1(shifted))
        assert self.lower is not None and self.upper is not None
        unit = torch.clamp((constrained - self.lower) / (self.upper - self.lower), min=epsilon, max=1.0 - epsilon)
        return torch.log(unit) - torch.log1p(-unit)


@dataclass(frozen=True, slots=True)
class Likelihood:
    output: str
    distribution: str = "gaussian"
    scale: float = 0.01
    weight: float = 1.0
    degrees_of_freedom: float = 4.0
    huber_delta: float = 1.0
    use_predicted_uncertainty: bool = False
    missing_observation_penalty: float = 25.0

    def __post_init__(self) -> None:
        if self.output not in {"rgb_linear", "depth_m", "uncertainty"}:
            raise InverseProblemError("inverse likelihood supports differentiable RGB/depth/uncertainty products")
        if self.distribution not in {"gaussian", "student_t", "huber", "poisson"}:
            raise InverseProblemError("likelihood distribution must be gaussian/student_t/huber/poisson")
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.weight) or self.weight <= 0:
            raise InverseProblemError("likelihood scale/weight must be finite and positive")
        if self.degrees_of_freedom <= 0 or self.huber_delta <= 0:
            raise InverseProblemError("student-t degrees of freedom and Huber delta must be positive")
        if not math.isfinite(self.missing_observation_penalty) or self.missing_observation_penalty < 0:
            raise InverseProblemError("missing-observation penalty must be finite and nonnegative")
        if self.distribution == "poisson" and self.output != "rgb_linear":
            raise InverseProblemError("Poisson likelihood is only defined for nonnegative RGB signal")

    def negative_log_likelihood(self, predicted: Any, observed: Any, uncertainty: Any | None = None) -> Any:
        torch = require_torch()
        finite = torch.isfinite(predicted) & torch.isfinite(observed)
        mismatch = torch.isfinite(predicted) ^ torch.isfinite(observed)
        missing_loss = self.missing_observation_penalty * mismatch.to(predicted.dtype).mean()
        if not bool(finite.any()):
            if bool(mismatch.any()):
                return self.weight * missing_loss
            raise InverseProblemError(f"likelihood {self.output!r} has no finite or mismatched samples")
        prediction = predicted[finite]
        target = observed[finite]
        if self.distribution == "poisson":
            rate = torch.clamp(prediction, min=torch.finfo(prediction.dtype).eps)
            return self.weight * ((rate - target * torch.log(rate)).mean() + missing_loss)
        scale: Any = torch.full_like(prediction, self.scale)
        if self.use_predicted_uncertainty:
            if uncertainty is None:
                raise InverseProblemError("heteroskedastic likelihood requested without predicted uncertainty")
            if uncertainty.ndim < predicted.ndim:
                uncertainty = uncertainty.unsqueeze(-1).expand_as(predicted)
            scale = torch.sqrt(scale * scale + torch.clamp(uncertainty[finite], min=0) ** 2)
        residual = (prediction - target) / scale
        if self.distribution == "gaussian":
            loss = 0.5 * residual * residual + torch.log(scale)
        elif self.distribution == "student_t":
            loss = 0.5 * (self.degrees_of_freedom + 1.0) * torch.log1p(residual * residual / self.degrees_of_freedom) + torch.log(scale)
        else:
            absolute = torch.abs(residual)
            loss = torch.where(absolute <= self.huber_delta, 0.5 * absolute * absolute, self.huber_delta * (absolute - 0.5 * self.huber_delta)) + torch.log(scale)
        return self.weight * (loss.mean() + missing_loss)


@dataclass(frozen=True, slots=True)
class DifferentiableConstraint:
    """User-defined differentiable penalty over physical and named values."""

    name: str
    function: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not callable(self.function) or not math.isfinite(self.weight) or self.weight <= 0:
            raise InverseProblemError("constraints require a name, callable, and positive finite weight")


@dataclass(frozen=True, slots=True)
class InverseView:
    pose: Pose
    observed: Mapping[str, Any]
    weight: float = 1.0
    id: str = "view"

    def __post_init__(self) -> None:
        if not self.id or not math.isfinite(self.weight) or self.weight <= 0:
            raise InverseProblemError("inverse views need an ID and positive finite weight")
        if not self.observed:
            raise InverseProblemError("inverse views require at least one observed product")

    @classmethod
    def from_observation(cls, observation: ObservationArtifact, *, weight: float = 1.0) -> "InverseView":
        return cls(observation.pose, observation.load_arrays(), weight, observation.id)


@dataclass(frozen=True, slots=True)
class InverseResult:
    method: str
    converged: bool
    iterations: int
    final_loss: float
    loss_history: tuple[float, ...]
    parameters: Mapping[str, np.ndarray]
    predictions: tuple[Mapping[str, np.ndarray], ...]
    posterior_covariance: np.ndarray | None = None
    posterior_parameter_order: tuple[str, ...] = ()


class OpticalInverseProblem:
    """Solve arbitrary differentiable geometry/material/camera/light inference."""

    def __init__(
        self,
        backend: TorchOpticalBackend,
        scene: RuntimeScene | CompiledScene | TorchScene,
        sensor: OpticalSensor,
        views: Sequence[InverseView],
        parameters: Sequence[ParameterSpec],
        likelihoods: Sequence[Likelihood],
        constraints: Sequence[DifferentiableConstraint] = (),
    ) -> None:
        torch = require_torch()
        if not views or not parameters or not likelihoods:
            raise InverseProblemError("inverse problem requires views, parameters, and likelihoods")
        if len({item.name for item in parameters}) != len(parameters):
            raise InverseProblemError("inverse parameter names must be unique")
        self.backend = backend
        self.scene = scene if isinstance(scene, TorchScene) else backend.compile(scene)
        self.sensor = sensor
        self.views = tuple(views)
        self.parameter_specs = tuple(parameters)
        self.likelihoods = tuple(likelihoods)
        self.constraints = tuple(constraints)
        self._base: dict[str, Any] = self.scene.differentiable_state()
        self._base["cameras.translation"] = torch.zeros((len(views), 3), device=backend.device, dtype=backend.dtype)
        self._base["cameras.rotation_vector"] = torch.zeros((len(views), 3), device=backend.device, dtype=backend.dtype)
        self._base["sensor.focal_length"] = backend.tensor(sensor.focal_length_px)
        self._base["sensor.principal_point"] = backend.tensor(sensor.principal_point_px)
        self._observed: tuple[dict[str, Any], ...] = tuple(
            {name: backend.tensor(value) for name, value in view.observed.items()} for view in views
        )
        self._raw: list[Any] = []
        for spec in self.parameter_specs:
            if spec.target not in self._base:
                raise InverseProblemError(f"unknown inverse target {spec.target!r}; available: {sorted(self._base)}")
            baseline = self._base[spec.target] if spec.selector is None else self._base[spec.target][spec.selector]
            initial = baseline if spec.initial is None else backend.tensor(spec.initial)
            if tuple(initial.shape) != tuple(baseline.shape):
                raise InverseProblemError(f"initial value for {spec.name!r} has shape {tuple(initial.shape)}, expected {tuple(baseline.shape)}")
            raw = spec.unconstrain(initial)
            if not bool(torch.isfinite(raw).all()):
                raise InverseProblemError(f"initial value for {spec.name!r} is outside its transform domain")
            self._raw.append(torch.nn.Parameter(raw))

    @property
    def available_targets(self) -> tuple[str, ...]:
        return tuple(sorted(self._base))

    def _values(self, raw_values: Sequence[Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        raws = self._raw if raw_values is None else raw_values
        values = {name: value.clone() for name, value in self._base.items()}
        constrained: dict[str, Any] = {}
        for spec, raw in zip(self.parameter_specs, raws, strict=True):
            current = spec.constrain(raw)
            constrained[spec.name] = current
            if spec.selector is None:
                values[spec.target] = current
            else:
                target = values[spec.target].clone()
                target[spec.selector] = current
                values[spec.target] = target
        return values, constrained

    def _predict(self, values: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        state_keys = set(self.scene.differentiable_state())
        scene_state = {name: values[name] for name in state_keys}
        predictions: list[dict[str, Any]] = []
        for index, view in enumerate(self.views):
            products = self.backend.render(
                self.scene,
                self.sensor,
                view.pose,
                state=scene_state,
                camera_translation_delta=values["cameras.translation"][index],
                camera_rotation_vector=values["cameras.rotation_vector"][index],
                focal_length_px=values["sensor.focal_length"],
                principal_point_px=values["sensor.principal_point"],
                apply_noise=False,
            )
            predictions.append(products.as_dict())
        return tuple(predictions)

    def negative_log_posterior(self, raw_values: Sequence[Any] | None = None) -> Any:
        values, constrained = self._values(raw_values)
        predictions = self._predict(values)
        loss = None
        for index, (view, predicted, observed) in enumerate(zip(self.views, predictions, self._observed, strict=True)):
            for likelihood in self.likelihoods:
                if likelihood.output not in observed:
                    raise InverseProblemError(f"view {view.id!r} lacks observed output {likelihood.output!r}")
                current = likelihood.negative_log_likelihood(
                    predicted[likelihood.output], observed[likelihood.output], predicted.get("uncertainty")
                ) * view.weight
                loss = current if loss is None else loss + current
        for spec in self.parameter_specs:
            if spec.prior is not None:
                current = spec.prior.negative_log_probability(constrained[spec.name])
                loss = current if loss is None else loss + current
        for constraint in self.constraints:
            current = constraint.function(values, constrained)
            if getattr(current, "numel", lambda: 0)() != 1:
                raise InverseProblemError(f"constraint {constraint.name!r} must return one scalar tensor")
            loss = current * constraint.weight if loss is None else loss + current * constraint.weight
        assert loss is not None
        return loss

    def _raw_vector(self) -> Any:
        torch = require_torch()
        return torch.cat([item.reshape(-1) for item in self._raw])

    def _split_vector(self, vector: Any) -> list[Any]:
        result: list[Any] = []
        offset = 0
        for raw in self._raw:
            length = raw.numel()
            result.append(vector[offset : offset + length].reshape(raw.shape))
            offset += length
        return result

    def _loss_from_vector(self, vector: Any) -> Any:
        return self.negative_log_posterior(self._split_vector(vector))

    def _constrained_vector(self, vector: Any) -> Any:
        torch = require_torch()
        raw_values = self._split_vector(vector)
        return torch.cat([spec.constrain(raw).reshape(-1) for spec, raw in zip(self.parameter_specs, raw_values, strict=True)])

    def posterior_covariance(self, *, maximum_parameters: int = 256, jitter: float = 1e-6) -> np.ndarray:
        """Laplace covariance in constrained physical-parameter coordinates."""
        torch = require_torch()
        vector = self._raw_vector().detach().requires_grad_(True)
        if vector.numel() > maximum_parameters:
            raise InverseProblemError(f"Laplace covariance is bounded to {maximum_parameters} raw parameters")
        hessian = torch.autograd.functional.hessian(self._loss_from_vector, vector, vectorize=True)
        hessian = 0.5 * (hessian + hessian.T)
        identity = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
        covariance_raw = torch.linalg.pinv(hessian + jitter * identity, hermitian=True)
        jacobian = torch.autograd.functional.jacobian(self._constrained_vector, vector, vectorize=True)
        covariance = jacobian @ covariance_raw @ jacobian.T
        return covariance.detach().cpu().numpy()

    def _posterior_component_order(self) -> tuple[str, ...]:
        result: list[str] = []
        for spec, raw in zip(self.parameter_specs, self._raw, strict=True):
            if raw.numel() == 1:
                result.append(spec.name)
            else:
                result.extend(f"{spec.name}[{index}]" for index in range(raw.numel()))
        return tuple(result)

    def solve(
        self,
        *,
        method: str = "adam",
        iterations: int = 200,
        learning_rate: float = 0.03,
        tolerance: float = 1e-8,
        gradient_clip_norm: float | None = 100.0,
        compute_posterior_covariance: bool = False,
        covariance_maximum_parameters: int = 256,
    ) -> InverseResult:
        torch = require_torch()
        if method not in {"adam", "lbfgs"} or iterations < 1 or learning_rate <= 0 or tolerance < 0:
            raise InverseProblemError("invalid inverse optimizer configuration")
        history: list[float] = []
        converged = False
        if method == "adam":
            optimizer = torch.optim.Adam(self._raw, lr=learning_rate)
            previous: float | None = None
            for _iteration in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                loss = self.negative_log_posterior()
                if not bool(torch.isfinite(loss)):
                    raise InverseProblemError("inverse objective became NaN or infinity")
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._raw, gradient_clip_norm)
                optimizer.step()
                current = float(loss.detach().cpu())
                history.append(current)
                if previous is not None and abs(previous - current) <= tolerance * max(1.0, abs(previous)):
                    converged = True
                    break
                previous = current
        else:
            optimizer = torch.optim.LBFGS(self._raw, lr=learning_rate, max_iter=iterations, tolerance_grad=tolerance, tolerance_change=tolerance, line_search_fn="strong_wolfe")

            def closure() -> Any:
                optimizer.zero_grad(set_to_none=True)
                loss = self.negative_log_posterior()
                if not bool(torch.isfinite(loss)):
                    raise InverseProblemError("inverse objective became NaN or infinity")
                loss.backward()
                history.append(float(loss.detach().cpu()))
                return loss

            optimizer.step(closure)
            converged = len(history) < iterations or (len(history) > 1 and abs(history[-2] - history[-1]) <= tolerance * max(1.0, abs(history[-2])))
        values, constrained = self._values()
        predictions = self._predict(values)
        final_loss = float(self.negative_log_posterior().detach().cpu())
        covariance = self.posterior_covariance(maximum_parameters=covariance_maximum_parameters) if compute_posterior_covariance else None
        return InverseResult(
            method=method,
            converged=converged,
            iterations=len(history),
            final_loss=final_loss,
            loss_history=tuple(history),
            parameters={name: value.detach().cpu().numpy().copy() for name, value in constrained.items()},
            predictions=tuple({name: value.detach().cpu().numpy().copy() for name, value in prediction.items()} for prediction in predictions),
            posterior_covariance=covariance,
            posterior_parameter_order=self._posterior_component_order(),
        )


__all__ = [
    "DifferentiableConstraint", "InverseProblemError", "InverseResult", "InverseView", "Likelihood",
    "OpticalInverseProblem", "ParameterSpec", "Prior",
]
