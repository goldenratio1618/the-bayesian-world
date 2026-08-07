"""Numerical backends used by the offline simulator.

NumPy is the dependency-free (beyond the package's core dependency) reference
backend.  PyTorch is imported only when :class:`TorchBackend` is constructed;
installing ``contraption`` therefore does not import, or require, PyTorch.

The small adapter deliberately contains only operations needed by the engine.
Keeping backend conversion in one place is important: a simulation executed by
the torch backend never passes through NumPy, so gradients from a loss on a
trajectory reach model parameters and initial conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


Array = Any


@runtime_checkable
class Backend(Protocol):
    """Structural protocol implemented by numerical backends."""

    name: str
    is_torch: bool
    dtype: Any
    device: Any

    def asarray(self, value: Any, *, dtype: Any | None = None) -> Array: ...

    def zeros(self, shape: Sequence[int] | int) -> Array: ...

    def ones(self, shape: Sequence[int] | int) -> Array: ...

    def full(self, shape: Sequence[int] | int, value: Any) -> Array: ...

    def stack(self, values: Sequence[Array], axis: int = 0) -> Array: ...

    def concatenate(self, values: Sequence[Array], axis: int = 0) -> Array: ...

    def broadcast_to(self, value: Array, shape: Sequence[int]) -> Array: ...

    def atan2(self, y: Array, x: Array) -> Array: ...

    def tanh(self, value: Array) -> Array: ...

    def remainder(self, value: Array, divisor: Array) -> Array: ...

    def where(self, condition: Array, when_true: Array, when_false: Array) -> Array: ...

    def logical_not(self, value: Array) -> Array: ...

    def logical_and(self, left: Array, right: Array) -> Array: ...

    def logical_or(self, left: Array, right: Array) -> Array: ...

    def make_rng(self, seed: int) -> Any: ...

    def normal(self, shape: Sequence[int], rng: Any) -> Array: ...

    def uniform(self, shape: Sequence[int], rng: Any) -> Array: ...


@dataclass(frozen=True)
class NumpyBackend:
    """Portable vectorized backend implemented with NumPy."""

    dtype: Any = np.float64
    name: str = "numpy"
    is_torch: bool = False
    device: str = "cpu"

    def asarray(self, value: Any, *, dtype: Any | None = None) -> np.ndarray:
        return np.asarray(value, dtype=self.dtype if dtype is None else dtype)

    def zeros(self, shape: Sequence[int] | int) -> np.ndarray:
        return np.zeros(shape, dtype=self.dtype)

    def ones(self, shape: Sequence[int] | int) -> np.ndarray:
        return np.ones(shape, dtype=self.dtype)

    def full(self, shape: Sequence[int] | int, value: Any) -> np.ndarray:
        return np.full(shape, value, dtype=self.dtype)

    def eye(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=self.dtype)

    def stack(self, values: Sequence[Array], axis: int = 0) -> np.ndarray:
        return np.stack(values, axis=axis)

    def concatenate(self, values: Sequence[Array], axis: int = 0) -> np.ndarray:
        return np.concatenate(values, axis=axis)

    def broadcast_to(self, value: Array, shape: Sequence[int]) -> np.ndarray:
        return np.broadcast_to(value, tuple(shape))

    def reshape(self, value: Array, shape: Sequence[int]) -> np.ndarray:
        return np.reshape(value, tuple(shape))

    def transpose(self, value: Array, axes: Sequence[int]) -> np.ndarray:
        return np.transpose(value, tuple(axes))

    def sin(self, value: Array) -> np.ndarray:
        return np.sin(value)

    def cos(self, value: Array) -> np.ndarray:
        return np.cos(value)

    def tan(self, value: Array) -> np.ndarray:
        return np.tan(value)

    def tanh(self, value: Array) -> np.ndarray:
        return np.tanh(value)

    def asin(self, value: Array) -> np.ndarray:
        return np.arcsin(value)

    def acos(self, value: Array) -> np.ndarray:
        return np.arccos(value)

    def atan(self, value: Array) -> np.ndarray:
        return np.arctan(value)

    def atan2(self, y: Array, x: Array) -> np.ndarray:
        return np.arctan2(y, x)

    def remainder(self, value: Array, divisor: Array) -> np.ndarray:
        return np.remainder(value, divisor)

    def exp(self, value: Array) -> np.ndarray:
        return np.exp(value)

    def log(self, value: Array) -> np.ndarray:
        return np.log(value)

    def log10(self, value: Array) -> np.ndarray:
        return np.log10(value)

    def sqrt(self, value: Array) -> np.ndarray:
        return np.sqrt(value)

    def abs(self, value: Array) -> np.ndarray:
        return np.abs(value)

    def sign(self, value: Array) -> np.ndarray:
        return np.sign(value)

    def where(self, condition: Array, when_true: Array, when_false: Array) -> np.ndarray:
        return np.where(condition, when_true, when_false)

    def logical_not(self, value: Array) -> np.ndarray:
        return np.logical_not(value)

    def logical_and(self, left: Array, right: Array) -> np.ndarray:
        return np.logical_and(left, right)

    def logical_or(self, left: Array, right: Array) -> np.ndarray:
        return np.logical_or(left, right)

    def maximum(self, left: Array, right: Array) -> np.ndarray:
        return np.maximum(left, right)

    def minimum(self, left: Array, right: Array) -> np.ndarray:
        return np.minimum(left, right)

    def clip(self, value: Array, low: Any | None, high: Any | None) -> np.ndarray:
        if low is None and high is None:
            return value
        return np.clip(value, low, high)

    def mean(self, value: Array, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
        return np.mean(value, axis=axis)

    def sum(self, value: Array, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
        return np.sum(value, axis=axis)

    def quantile(self, value: Array, q: float | Sequence[float], axis: int = 0) -> np.ndarray:
        return np.quantile(value, q, axis=axis)

    def einsum(self, expression: str, *values: Array) -> np.ndarray:
        return np.einsum(expression, *values)

    def solve(self, matrix: Array, rhs: Array) -> np.ndarray:
        return np.linalg.solve(matrix, rhs)

    def pinv(self, matrix: Array) -> np.ndarray:
        return np.linalg.pinv(matrix)

    def cholesky(self, matrix: Array) -> np.ndarray:
        return np.linalg.cholesky(matrix)

    def slogdet(self, matrix: Array) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.slogdet(matrix)

    def norm(self, value: Array) -> np.ndarray:
        return np.linalg.norm(value)

    def make_rng(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(int(seed))

    def normal(self, shape: Sequence[int], rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(tuple(shape)).astype(self.dtype, copy=False)

    def uniform(self, shape: Sequence[int], rng: np.random.Generator) -> np.ndarray:
        return rng.random(tuple(shape), dtype=self.dtype)

    def clone(self, value: Array) -> np.ndarray:
        return np.array(value, copy=True)

    def to_numpy(self, value: Array) -> np.ndarray:
        return np.asarray(value)


class TorchBackend:
    """Lazy PyTorch backend supporting CPU and CUDA execution.

    Parameters are not copied when they already have the requested dtype and
    device.  This preserves leaf tensors supplied by callers and, consequently,
    their autograd history.
    """

    name = "torch"
    is_torch = True

    def __init__(self, device: str | None = None, dtype: Any = "float64") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The torch backend is optional; install contraption[gpu] to use it"
            ) from exc

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        if isinstance(dtype, str):
            aliases = {
                "float32": torch.float32,
                "single": torch.float32,
                "float64": torch.float64,
                "double": torch.float64,
            }
            try:
                dtype = aliases[dtype.lower()]
            except KeyError as exc:
                raise ValueError(f"Unsupported torch dtype: {dtype!r}") from exc
        self.dtype = dtype

    def asarray(self, value: Any, *, dtype: Any | None = None) -> Array:
        target_dtype = self.dtype if dtype is None else dtype
        if isinstance(value, self.torch.Tensor):
            if value.device == self.device and value.dtype == target_dtype:
                return value
            return value.to(device=self.device, dtype=target_dtype)
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, self.torch.Tensor) for item in value
        ):
            return self.torch.stack(
                tuple(self.asarray(item, dtype=target_dtype) for item in value), dim=0
            )
        return self.torch.as_tensor(value, device=self.device, dtype=target_dtype)

    def zeros(self, shape: Sequence[int] | int) -> Array:
        return self.torch.zeros(shape, device=self.device, dtype=self.dtype)

    def ones(self, shape: Sequence[int] | int) -> Array:
        return self.torch.ones(shape, device=self.device, dtype=self.dtype)

    def full(self, shape: Sequence[int] | int, value: Any) -> Array:
        if isinstance(value, self.torch.Tensor):
            return self.ones(shape) * value
        return self.torch.full(shape, value, device=self.device, dtype=self.dtype)

    def eye(self, n: int) -> Array:
        return self.torch.eye(n, device=self.device, dtype=self.dtype)

    def stack(self, values: Sequence[Array], axis: int = 0) -> Array:
        return self.torch.stack(tuple(values), dim=axis)

    def concatenate(self, values: Sequence[Array], axis: int = 0) -> Array:
        return self.torch.cat(tuple(values), dim=axis)

    def broadcast_to(self, value: Array, shape: Sequence[int]) -> Array:
        return self.torch.broadcast_to(value, tuple(shape))

    def reshape(self, value: Array, shape: Sequence[int]) -> Array:
        return self.torch.reshape(value, tuple(shape))

    def transpose(self, value: Array, axes: Sequence[int]) -> Array:
        return value.permute(tuple(axes))

    def sin(self, value: Array) -> Array:
        return self.torch.sin(value)

    def cos(self, value: Array) -> Array:
        return self.torch.cos(value)

    def tan(self, value: Array) -> Array:
        return self.torch.tan(value)

    def tanh(self, value: Array) -> Array:
        return self.torch.tanh(value)

    def asin(self, value: Array) -> Array:
        return self.torch.asin(value)

    def acos(self, value: Array) -> Array:
        return self.torch.acos(value)

    def atan(self, value: Array) -> Array:
        return self.torch.atan(value)

    def atan2(self, y: Array, x: Array) -> Array:
        return self.torch.atan2(self.asarray(y), self.asarray(x))

    def remainder(self, value: Array, divisor: Array) -> Array:
        return self.torch.remainder(value, divisor)

    def exp(self, value: Array) -> Array:
        return self.torch.exp(value)

    def log(self, value: Array) -> Array:
        return self.torch.log(value)

    def log10(self, value: Array) -> Array:
        return self.torch.log10(value)

    def sqrt(self, value: Array) -> Array:
        return self.torch.sqrt(value)

    def abs(self, value: Array) -> Array:
        return self.torch.abs(value)

    def sign(self, value: Array) -> Array:
        return self.torch.sign(value)

    def where(self, condition: Array, when_true: Array, when_false: Array) -> Array:
        if not isinstance(condition, self.torch.Tensor):
            condition = self.torch.as_tensor(condition, device=self.device, dtype=self.torch.bool)
        elif condition.device != self.device:
            condition = condition.to(device=self.device)
        tensor_branch = next(
            (
                branch
                for branch in (when_true, when_false)
                if isinstance(branch, self.torch.Tensor)
            ),
            None,
        )
        if tensor_branch is None:
            true_value = self.asarray(when_true)
            false_value = self.asarray(when_false)
        else:
            target_dtype = tensor_branch.dtype
            true_value = self.torch.as_tensor(
                when_true, device=self.device, dtype=target_dtype
            )
            false_value = self.torch.as_tensor(
                when_false, device=self.device, dtype=target_dtype
            )
        return self.torch.where(condition, true_value, false_value)

    def logical_not(self, value: Array) -> Array:
        return self.torch.logical_not(self.torch.as_tensor(value, device=self.device))

    def logical_and(self, left: Array, right: Array) -> Array:
        return self.torch.logical_and(
            self.torch.as_tensor(left, device=self.device),
            self.torch.as_tensor(right, device=self.device),
        )

    def logical_or(self, left: Array, right: Array) -> Array:
        return self.torch.logical_or(
            self.torch.as_tensor(left, device=self.device),
            self.torch.as_tensor(right, device=self.device),
        )

    def maximum(self, left: Array, right: Array) -> Array:
        left = self.asarray(left)
        right = self.asarray(right)
        return self.torch.maximum(left, right)

    def minimum(self, left: Array, right: Array) -> Array:
        left = self.asarray(left)
        right = self.asarray(right)
        return self.torch.minimum(left, right)

    def clip(self, value: Array, low: Any | None, high: Any | None) -> Array:
        if low is not None:
            value = self.torch.maximum(value, self.asarray(low))
        if high is not None:
            value = self.torch.minimum(value, self.asarray(high))
        return value

    def mean(self, value: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        if axis is None:
            return self.torch.mean(value)
        return self.torch.mean(value, dim=axis)

    def sum(self, value: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        if axis is None:
            return self.torch.sum(value)
        return self.torch.sum(value, dim=axis)

    def quantile(self, value: Array, q: float | Sequence[float], axis: int = 0) -> Array:
        q_tensor = self.asarray(q)
        return self.torch.quantile(value, q_tensor, dim=axis)

    def einsum(self, expression: str, *values: Array) -> Array:
        return self.torch.einsum(expression, *values)

    def solve(self, matrix: Array, rhs: Array) -> Array:
        return self.torch.linalg.solve(matrix, rhs)

    def pinv(self, matrix: Array) -> Array:
        return self.torch.linalg.pinv(matrix)

    def cholesky(self, matrix: Array) -> Array:
        return self.torch.linalg.cholesky(matrix)

    def slogdet(self, matrix: Array) -> tuple[Array, Array]:
        result = self.torch.linalg.slogdet(matrix)
        return result.sign, result.logabsdet

    def norm(self, value: Array) -> Array:
        return self.torch.linalg.vector_norm(value)

    def make_rng(self, seed: int) -> Any:
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def normal(self, shape: Sequence[int], rng: Any) -> Array:
        return self.torch.randn(
            tuple(shape), generator=rng, device=self.device, dtype=self.dtype
        )

    def uniform(self, shape: Sequence[int], rng: Any) -> Array:
        return self.torch.rand(
            tuple(shape), generator=rng, device=self.device, dtype=self.dtype
        )

    def clone(self, value: Array) -> Array:
        return value.clone()

    def to_numpy(self, value: Array) -> np.ndarray:
        # Serialization/inspection is an explicit graph boundary.  Core engine
        # code never calls this method during simulation or fitting.
        return value.detach().cpu().numpy()


def get_backend(
    backend: str | Backend | None = None,
    *,
    device: str | None = None,
    dtype: Any | None = None,
) -> Backend:
    """Return a backend adapter.

    ``"auto"`` uses CUDA when a CUDA-enabled PyTorch installation is already
    available, otherwise NumPy.  Merely importing this module stays lazy: torch
    is probed only when ``"auto"`` or ``"torch"`` is requested.
    """

    cuda_requested = device is not None and str(device).lower().startswith("cuda")
    if backend is not None and not isinstance(backend, str):
        if cuda_requested:
            actual_device = str(getattr(backend, "device", ""))
            if not bool(getattr(backend, "is_torch", False)) or not actual_device.startswith(
                "cuda"
            ):
                raise ValueError(
                    f"device={device!r} requires a CUDA torch backend; received "
                    f"{getattr(backend, 'name', type(backend).__name__)!r} on "
                    f"device={actual_device or 'unknown'!r}"
                )
        return backend
    choice = "numpy" if backend is None else backend.lower()
    if choice in {"numpy", "np", "cpu"}:
        if cuda_requested:
            raise ValueError(
                f"device={device!r} cannot be used with the NumPy backend; "
                "select backend='torch' or backend='auto'"
            )
        return NumpyBackend(np.float64 if dtype is None else dtype)
    if choice in {"torch", "pytorch", "cuda"}:
        if choice == "cuda" and device is not None and not cuda_requested:
            raise ValueError(
                f"backend='cuda' conflicts with device={device!r}; request a CUDA device"
            )
        selected_device = "cuda" if choice == "cuda" and device is None else device
        return TorchBackend(selected_device, "float64" if dtype is None else dtype)
    if choice == "auto":
        # An explicit CUDA request is a requirement, not a preference.  Let the
        # lazy torch constructor report a missing install or unavailable GPU;
        # silently dropping to NumPy would run the requested workload on the
        # wrong device while still producing plausible numerical output.
        if cuda_requested:
            return TorchBackend(device, "float64" if dtype is None else dtype)
        try:
            import torch

            if torch.cuda.is_available():
                return TorchBackend(device or "cuda", "float64" if dtype is None else dtype)
        except ImportError:
            pass
        return NumpyBackend(np.float64 if dtype is None else dtype)
    raise ValueError(f"Unknown numerical backend: {backend!r}")


def infer_backend(value: Any, *, default: str = "numpy") -> Backend:
    """Infer torch from a tensor without importing it on NumPy-only installs."""

    module = type(value).__module__.split(".", 1)[0]
    if module == "torch":
        return TorchBackend(device=str(value.device), dtype=value.dtype)
    return get_backend(default)


def as_jsonable(value: Any) -> Any:
    """Convert numerical values to standard JSON-compatible containers."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if type(value).__module__.split(".", 1)[0] == "torch":
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


__all__ = [
    "Array",
    "Backend",
    "NumpyBackend",
    "TorchBackend",
    "as_jsonable",
    "get_backend",
    "infer_backend",
]
