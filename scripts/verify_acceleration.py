#!/usr/bin/env python3
"""Verify portable and accelerated simulator paths, including autograd.

This is a runtime smoke test, not a benchmark.  It executes NumPy's analytic RC
baseline, a seeded batched PyTorch simulation on the selected device, a matrix
kernel, and reverse-mode differentiation through the trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def nvidia_smi_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        return {"visible": False, "executable": None}
    try:
        process = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        devices = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        return {"visible": bool(devices), "executable": executable, "devices": devices}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"visible": False, "executable": executable, "error": str(exc)}


def verify_numpy() -> dict[str, Any]:
    import numpy as np

    from contraption.simulator import RCCircuit, simulate

    result = simulate(
        RCCircuit(resistance=2.0, capacitance=0.5),
        duration=0.5,
        dt=0.01,
        controls={"voltage": 5.0},
        num_samples=1,
        backend="numpy",
        use_model_uncertainty=False,
        process_noise=False,
    )
    observed = float(result.mean[-1, 0])
    expected = 5.0 * (1.0 - math.exp(-0.5))
    error = abs(observed - expected)
    if error > 1e-8:
        raise RuntimeError(f"NumPy analytic baseline error {error:g} exceeds tolerance")
    return {
        "version": np.__version__,
        "analytic_rc_absolute_error": error,
        "trajectory_shape": list(result.samples.shape),
    }


def verify_torch(expectation: str, samples: int, steps: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc

    cuda_available = bool(torch.cuda.is_available())
    if expectation == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was required but torch.cuda.is_available() is false: "
            f"torch CUDA runtime={torch.version.cuda!r}"
        )
    device = "cuda" if expectation == "cuda" or (expectation == "auto" and cuda_available) else "cpu"

    from contraption.simulator import RCCircuit, simulate

    resistance = torch.tensor(2.0, dtype=torch.float64, device=device, requires_grad=True)
    model = RCCircuit(resistance=resistance, capacitance=0.5)
    options = dict(
        duration=steps * 0.01,
        dt=0.01,
        controls={"voltage": 5.0},
        parameter_distribution={
            "resistance": {"mean": resistance, "std": 0.05, "lower": 0.2, "upper": 10.0}
        },
        num_samples=samples,
        seed=20260806,
        backend="torch",
        device=device,
        use_model_uncertainty=False,
        process_noise=False,
    )
    result = simulate(model, **options)
    repeated = simulate(model, **options)
    if result.samples.device.type != device:
        raise RuntimeError(f"Trajectory is on {result.samples.device}, expected {device}")
    if not torch.equal(result.samples, repeated.samples):
        raise RuntimeError("Seeded PyTorch Monte Carlo trajectory is not deterministic")
    if not bool(torch.isfinite(result.samples).all()):
        raise RuntimeError("PyTorch trajectory contains non-finite values")

    objective = result.mean[-1, 0]
    objective.backward()
    gradient = resistance.grad
    if gradient is None or not bool(torch.isfinite(gradient)) or abs(float(gradient)) < 1e-9:
        raise RuntimeError("Autograd did not produce a finite, nonzero parameter gradient")

    # This ensures an actual BLAS/CUDA kernel launches in addition to the
    # simulator's elementwise kernels.  synchronize() surfaces asynchronous
    # device failures before setup is declared successful.
    generator = torch.Generator(device=device)
    generator.manual_seed(19)
    matrix = torch.randn((256, 256), generator=generator, dtype=torch.float32, device=device)
    checksum = float((matrix @ matrix.transpose(0, 1)).mean())
    if device == "cuda":
        torch.cuda.synchronize()

    report: dict[str, Any] = {
        "version": torch.__version__,
        "compiled_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "selected_device": device,
        "trajectory_shape": list(result.samples.shape),
        "seed_reproducible": True,
        "autograd_gradient": float(gradient),
        "matrix_kernel_checksum": checksum,
    }
    if cuda_available:
        report.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "device_count": torch.cuda.device_count(),
            }
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="required execution device (auto prefers CUDA)",
    )
    parser.add_argument("--samples", type=int, default=128, help="Monte Carlo batch size")
    parser.add_argument("--steps", type=int, default=20, help="simulation integration steps")
    parser.add_argument("--json", action="store_true", help="emit one JSON document")
    arguments = parser.parse_args()
    if arguments.samples < 2 or arguments.steps < 1:
        parser.error("--samples must be >= 2 and --steps must be >= 1")
    return arguments


def main() -> int:
    arguments = parse_args()
    report: dict[str, Any] = {
        "ok": False,
        "expectation": arguments.expect,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "nvidia_smi": nvidia_smi_report(),
    }
    try:
        report["numpy"] = verify_numpy()
        report["torch"] = verify_torch(arguments.expect, arguments.samples, arguments.steps)
        report["ok"] = True
    except Exception as exc:  # A smoke-test CLI should report the complete reason.
        report["error"] = f"{type(exc).__name__}: {exc}"

    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"Acceleration verification: {status}")
        print(f"  Python: {report['python']}")
        if "numpy" in report:
            print(f"  NumPy: {report['numpy']['version']} (analytic baseline passed)")
        if "torch" in report:
            torch_report = report["torch"]
            print(
                f"  PyTorch: {torch_report['version']} on {torch_report['selected_device']} "
                f"(built for CUDA {torch_report['compiled_cuda_runtime']})"
            )
            if "device_name" in torch_report:
                print(f"  GPU: {torch_report['device_name']}")
            print("  Seeded Monte Carlo and autograd: passed")
        if not report["ok"]:
            print(f"  Error: {report['error']}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

