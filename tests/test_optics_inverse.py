from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from contraption.optics import (
    InverseView,
    Likelihood,
    MeshInstance,
    NumpyOpticalBackend,
    OpticalInverseProblem,
    OpticalLight,
    OpticalSensor,
    ParameterSpec,
    Pose,
    Prior,
    RuntimeMaterial,
    RuntimeScene,
    SensorNoise,
    TorchOpticalBackend,
)
from contraption.shape import TriangleMesh


def _scene(intensity: float) -> RuntimeScene:
    mesh = TriangleMesh(
        np.asarray([[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 2.0]]),
        np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
    )
    return RuntimeScene(
        "inverse",
        (MeshInstance("target", mesh, 1, (RuntimeMaterial("white", (0.7, 0.7, 0.7), roughness=0.9),)),),
        (OpticalLight("lamp", "point", (1, 1, 1), intensity, position_m=(0.0, 0.0, 0.0)),),
        (0.01, 0.01, 0.01),
    )


def _sensor() -> OpticalSensor:
    return OpticalSensor(
        "camera", (3, 3), (4.0, 4.0), (1.5, 1.5),
        near_clip_m=0.1, far_clip_m=10.0, exposure_duration_s=1.0,
        noise=SensorNoise("none"),
    )


def test_torch_backend_has_geometry_material_camera_and_light_gradients() -> None:
    torch = pytest.importorskip("torch")
    backend = TorchOpticalBackend(device="cpu", dtype="float64")
    scene = backend.compile(_scene(20.0))
    state = scene.differentiable_state()
    for name in ("geometry.vertices", "materials.base_color", "materials.refractive_index", "lights.intensity"):
        state[name] = state[name].clone().requires_grad_(True)
    translation = torch.zeros(3, dtype=backend.dtype, device=backend.device, requires_grad=True)
    rotation = torch.zeros(3, dtype=backend.dtype, device=backend.device, requires_grad=True)
    focal = backend.tensor(_sensor().focal_length_px).requires_grad_(True)
    principal = backend.tensor(_sensor().principal_point_px).requires_grad_(True)
    rendered = backend.render(scene, _sensor(), state=state, camera_translation_delta=translation, camera_rotation_vector=rotation, focal_length_px=focal, principal_point_px=principal)
    weights = torch.linspace(0.7, 1.3, rendered.rgb_linear.numel(), dtype=backend.dtype).reshape_as(rendered.rgb_linear)
    loss = (rendered.rgb_linear * weights).sum() + torch.where(torch.isfinite(rendered.depth_m), rendered.depth_m, 0).sum()
    loss.backward()
    assert state["geometry.vertices"].grad is not None
    assert state["materials.base_color"].grad is not None
    assert state["materials.refractive_index"].grad is not None
    assert state["lights.intensity"].grad is not None
    assert translation.grad is not None
    assert rotation.grad is not None
    assert focal.grad is not None
    assert principal.grad is not None


def test_numpy_and_torch_dielectric_fresnel_match_across_refractive_indices() -> None:
    sensor = _sensor()
    mesh = TriangleMesh(
        np.asarray(
            [
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [1.0, 1.0, 2.0],
                [-1.0, 1.0, 2.0],
            ]
        ),
        np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
    )
    numpy_backend = NumpyOpticalBackend(cast_shadows=False)
    torch_backend = TorchOpticalBackend(
        device="cpu", dtype="float64", cast_shadows=False
    )
    center_values: list[float] = []
    for refractive_index in (1.1, 1.5, 2.0):
        scene = RuntimeScene(
            "fresnel-parity",
            (
                MeshInstance(
                    "dielectric",
                    mesh,
                    1,
                    (
                        RuntimeMaterial(
                            "dielectric",
                            (0.0, 0.0, 0.0),
                            roughness=0.5,
                            refractive_index=refractive_index,
                        ),
                    ),
                ),
            ),
            (
                OpticalLight(
                    "headlight",
                    "point",
                    (1.0, 1.0, 1.0),
                    100.0,
                    position_m=(0.0, 0.0, 0.0),
                ),
            ),
            (0.0, 0.0, 0.0),
        )
        numpy_products = numpy_backend.render(
            scene, sensor, apply_noise=False
        )
        torch_products = torch_backend.render(
            scene, sensor, apply_noise=False
        ).numpy()
        assert np.allclose(
            numpy_products.rgb_linear,
            torch_products["rgb_linear"],
            rtol=1e-6,
            atol=1e-8,
        )
        center_values.append(float(numpy_products.rgb_linear[1, 1, 0]))
    assert center_values[0] < center_values[1] < center_values[2]


def test_inverse_problem_recovers_light_intensity_with_adam_and_covariance() -> None:
    backend = TorchOpticalBackend(device="cpu", dtype="float64")
    sensor = _sensor()
    target = backend.render(_scene(35.0), sensor).numpy()
    problem = OpticalInverseProblem(
        backend,
        _scene(8.0),
        sensor,
        (InverseView(Pose(), {"rgb_linear": target["rgb_linear"]}),),
        (
            ParameterSpec(
                "lamp_intensity",
                "lights.intensity",
                transform="positive",
                prior=Prior("lognormal", location=np.log(30.0), scale=1.0),
            ),
        ),
        (Likelihood("rgb_linear", "gaussian", scale=0.02),),
    )
    assert "geometry.vertices" in problem.available_targets
    assert "materials.base_color" in problem.available_targets
    assert "cameras.translation" in problem.available_targets
    result = problem.solve(method="adam", iterations=300, learning_rate=0.12, compute_posterior_covariance=True)
    assert result.final_loss < result.loss_history[0]
    assert result.parameters["lamp_intensity"][0] > 25.0
    assert result.posterior_covariance is not None
    assert result.posterior_covariance.shape == (1, 1)


def test_lbfgs_solves_material_inverse_problem() -> None:
    backend = TorchOpticalBackend(device="cpu", dtype="float64")
    sensor = _sensor()
    target_scene = _scene(20.0)
    target = backend.render(target_scene, sensor).numpy()
    problem = OpticalInverseProblem(
        backend,
        RuntimeScene(
            target_scene.id,
            (MeshInstance("target", target_scene.instances[0].mesh, 1, (RuntimeMaterial("white", (0.15, 0.15, 0.15), roughness=0.9),)),),
            target_scene.lights,
            target_scene.environment_linear_rgb,
        ),
        sensor,
        (InverseView(Pose(), {"rgb_linear": target["rgb_linear"]}),),
        (ParameterSpec("albedo", "materials.base_color", transform="bounded", lower=0.0, upper=1.0),),
        (Likelihood("rgb_linear", "huber", scale=0.02),),
    )
    result = problem.solve(method="lbfgs", iterations=40, learning_rate=0.8)
    assert result.final_loss < result.loss_history[0]
    assert np.all((result.parameters["albedo"] > 0.4) & (result.parameters["albedo"] < 0.95))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_backend_matches_cpu_and_backpropagates() -> None:
    sensor = _sensor()
    cpu = TorchOpticalBackend(device="cpu", dtype="float32").render(_scene(20.0), sensor).numpy()
    backend = TorchOpticalBackend(device="cuda", dtype="float32")
    scene = backend.compile(_scene(20.0))
    intensity = scene.light_intensity.clone().requires_grad_(True)
    rendered = backend.render(scene, sensor, state={"lights.intensity": intensity})
    assert backend.hardware_accelerated
    assert np.allclose(rendered.rgb_linear.detach().cpu().numpy(), cpu["rgb_linear"], atol=1e-5, rtol=1e-5)
    rendered.rgb_linear.sum().backward()
    assert intensity.grad is not None
    assert torch.isfinite(intensity.grad).all()
