"""Asynchronous optical capture scheduling and observation artifact creation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from .renderer import NumpyOpticalBackend, RenderProducts, RuntimeScene
from .schemas import ObservationArtifact, OpticalSensor, Pose


class OpticalSimulationError(RuntimeError):
    """Raised when a scheduled capture cannot be represented deterministically."""


@dataclass(frozen=True, slots=True)
class PendingCapture:
    id: str
    frame_index: int
    ready_at_s: float
    future: Future[ObservationArtifact]

    def result(self, timeout: float | None = None) -> ObservationArtifact:
        return self.future.result(timeout=timeout)


class AsyncOpticalSimulator:
    """Run captures concurrently while preserving deterministic simulation time.

    ``ready_at_s`` is simulated time. The executor never sleeps to emulate
    exposure; it computes and persists the result as soon as host resources are
    available, so offline simulations remain fast and reproducible.
    """

    def __init__(
        self,
        output_root: str | Path,
        *,
        backend: Any | None = None,
        max_workers: int = 2,
    ) -> None:
        if max_workers < 1:
            raise OpticalSimulationError("optical simulator needs at least one worker")
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or NumpyOpticalBackend()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="contraption-optics")
        self._lock = threading.Lock()
        self._reserved_paths: set[Path] = set()

    def submit_capture(
        self,
        scene: RuntimeScene,
        sensor: OpticalSensor,
        pose: Pose = Pose(),
        *,
        frame_index: int,
        requested_at_s: float,
        id: str | None = None,
        seed: int | None = None,
        assembly_id: str | None = None,
        assembly_sha256: str | None = None,
        assembly_frame: str = "world",
        assembly_mount_connector: str | None = None,
    ) -> PendingCapture:
        if isinstance(frame_index, bool) or frame_index < 0:
            raise OpticalSimulationError("frame_index must be a nonnegative integer")
        if requested_at_s < 0:
            raise OpticalSimulationError("requested_at_s must be nonnegative")
        capture_id = id or f"{sensor.id}-frame-{frame_index:08d}"
        manifest = self.output_root / f"{capture_id}.optical-observation.json"
        with self._lock:
            if manifest in self._reserved_paths or manifest.exists():
                raise OpticalSimulationError(f"capture artifact already exists: {manifest}")
            self._reserved_paths.add(manifest)
        ready_at_s = requested_at_s + sensor.exposure_duration_s + sensor.readout_duration_s + sensor.processing_latency_s
        render_seed = sensor.noise.seed if seed is None else int(seed)
        mount_connector = assembly_mount_connector
        if assembly_mount_connector is not None and (not assembly_mount_connector.strip() or "." not in assembly_mount_connector):
            raise OpticalSimulationError("assembly_mount_connector must be an assembly-qualified '<component>.<connector>' ID")
        if assembly_mount_connector is not None and sensor.mount_connector is not None and not assembly_mount_connector.endswith(f".{sensor.mount_connector}"):
            raise OpticalSimulationError("assembly_mount_connector does not bind the sensor's local mount connector")
        if any(value is not None for value in (assembly_id, assembly_sha256, assembly_mount_connector)) and not all(value is not None for value in (assembly_id, assembly_sha256, assembly_mount_connector)):
            raise OpticalSimulationError("assembly-bound observations require assembly_id, assembly_sha256, and assembly_mount_connector together")
        if assembly_sha256 is not None and sensor.mount_connector is not None and assembly_mount_connector is None:
            raise OpticalSimulationError("assembly-bound observations require assembly_mount_connector")

        def work() -> ObservationArtifact:
            try:
                products: RenderProducts = self.backend.render(
                    scene, sensor, pose, frame_index=frame_index, seed=render_seed, apply_noise=True
                )
                if hasattr(products, "numpy"):
                    rendered = products.numpy()
                    arrays = {name: rendered[name] for name in sensor.outputs}
                else:
                    arrays = products.as_dict(sensor.outputs)
                return ObservationArtifact.from_arrays(
                    path=manifest,
                    arrays=arrays,
                    id=capture_id,
                    sensor=sensor,
                    scene_sha256=scene.artifact_sha256,
                    frame_index=frame_index,
                    requested_at_s=requested_at_s,
                    exposure_started_at_s=requested_at_s,
                    ready_at_s=ready_at_s,
                    pose=pose,
                    seed=render_seed,
                    assembly_id=assembly_id,
                    assembly_sha256=assembly_sha256,
                    assembly_frame=assembly_frame,
                    mount_connector=mount_connector,
                    mount_transform_sha256=pose.artifact_sha256,
                    metadata={"backend": getattr(self.backend, "name", type(self.backend).__name__)},
                )
            finally:
                with self._lock:
                    self._reserved_paths.discard(manifest)

        future = self._executor.submit(work)
        return PendingCapture(capture_id, frame_index, ready_at_s, future)

    def capture(self, *args: Any, **kwargs: Any) -> ObservationArtifact:
        return self.submit_capture(*args, **kwargs).result()

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> "AsyncOpticalSimulator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.shutdown(wait=True)


__all__ = ["AsyncOpticalSimulator", "OpticalSimulationError", "PendingCapture"]
