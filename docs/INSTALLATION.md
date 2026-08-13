# Linux installation

The supported development environment is a recent 64-bit Linux distribution
with Python 3.10 or newer. A C99 compiler is required for the generated online
runtime checks. GPU-enabled PyTorch is the preferred simulator backend; NumPy
remains the explicit CPU fallback.

On Windows, use Ubuntu WSL and keep the checkout in the Linux filesystem rather
than under `/mnt/c`:

```bash
cd ~/src_bayesian/the-bayesian-world
```

## 1. Install distribution packages

Choose the command for the host distribution.

Debian, Ubuntu, or Linux Mint:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential git
```

Fedora or RHEL-family distributions:

```bash
sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ make git
```

Arch Linux:

```bash
sudo pacman -S --needed python python-pip base-devel git
```

Verify the interpreter and compiler before continuing:

```bash
python3 --version
cc --version
```

## 2. Create an isolated environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Do not reuse a Windows virtual environment from a mounted drive. Native Linux
environments avoid incompatible launchers and generally provide much faster
package and filesystem behavior.

## 3. Install the GPU runtime first

For an NVIDIA GPU, confirm that the Linux environment can see the device:

```bash
nvidia-smi
```

When using WSL, install or update the NVIDIA display driver on **Windows**.
Do not install a Linux NVIDIA display driver inside WSL: Microsoft/NVIDIA map
the Windows host driver into WSL. Follow the official
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) for supported
driver and toolkit setup.

Then use the [official PyTorch Linux selector](https://pytorch.org/get-started/locally/)
to choose the stable `pip` wheel for the CUDA version supported by the installed
driver. The selector is authoritative; the following is only an example for a
host where CUDA 12.6 is the appropriate wheel channel:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

For AMD hardware, select the supported ROCm wheel from the same official
selector. Do not install a CPU wheel and report the system as accelerated.

Install this package and the development and agent dependencies after PyTorch:

```bash
python -m pip install -e ".[gpu,agents,dev]"
```

The `gpu` extra requires PyTorch but does not override an already compatible
wheel selected for the host. The `agents` extra installs the OpenAI Python SDK;
it is needed only for component-classification API calls. The `dev` extra adds
the test and package-build tools.

If the host has no supported GPU, install the explicit CPU wheel and omit the
`gpu` extra:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[agents,dev]"
```

## 4. Verify the selected runtime

GPU verification is intentionally fail-closed:

```bash
contraption doctor
python scripts/verify_acceleration.py --expect cuda
python -m pytest
contraption validate --spec assembled_contraptions/scanner/contraption.json
```

`--expect cuda` exits nonzero if CUDA is unavailable, if the installed wheel is
CPU-only, if a device kernel fails, or if autograd does not propagate through a
trajectory. On a deliberate CPU installation, use `--expect cpu` instead.

Run the full GPU scanner fixture with:

```bash
contraption simulate \
  --spec assembled_contraptions/scanner/contraption.json \
  --backend torch --device cuda \
  --controller-input armed=true \
  --output outputs/scanner_demo

contraption view \
  --spec assembled_contraptions/scanner/contraption.json \
  --trajectory outputs/scanner_demo/trajectory.json \
  --output outputs/scanner_demo/viewer

python -m http.server 8000 --directory outputs/scanner_demo/viewer
```

Open <http://127.0.0.1:8000>. The simulator and viewer resolve the same
contraption/catalog-instantiation/PMDL/controller closure and require the same assembly hash.
The browser is display-only: it does not infer placement or execute a second
model. The CLI reconstructs poses from the trajectory's exact per-sample states
through that resolved assembly; detached scene JSON is not an admitted input.
Repeat `--controller-input NAME=JSON` for external controller pins, or supply a
strict JSON object through `--controller-input-file`. When `--dt` is omitted,
the runtime derives the greatest common subdivision of all controller periods;
an explicit `--dt` must divide every period within the scheduler tolerance.
Programmatic `simulate(..., controls=...)` providers are open-loop and may depend
only on time (plus the optional backend); plant-state feedback must be authored
as a resolved controller so hidden state cannot bypass explicit sensor wiring.

To change the controller's declared external inputs interactively, run the
loopback live server instead of `http.server`:

```bash
contraption serve \
  --spec assembled_contraptions/scanner/contraption.json \
  --backend torch --device cuda \
  --controller-input armed=true \
  --host 127.0.0.1 --port 8000
```

Each UI change reruns the canonical Python simulation and returns a strict
hash-bound scene. The server refuses unknown/out-of-range controls and stale
assembly hashes. It intentionally refuses non-loopback binding; authenticated
remote deployment is outside Phase 1.

Generate and host-compile the DAE-derived onboard reference separately:

```bash
contraption compile --spec assembled_contraptions/scanner/contraption.json --output outputs/scanner_demo/online
contraption build --spec assembled_contraptions/scanner/contraption.json --output outputs/scanner_demo/build
```

## 5. Configure optional component agents

Agent calls read only `OPENAI_API_KEY`. Either export it in the shell or place
it in a local `.env` file. The CLI searches the repository and then its parent
directory, but requires `--env-file` if both locations contain a file:

```bash
contraption agent-canary --kind both --env-file ../.env
contraption agent-run classification-all --job-file assembled_contraptions/scanner/agent_jobs.json --env-file ../.env
```

Never commit `.env`. Paid runs use the persistent
`outputs/agent-budget.json` ledger and write validated receipts beneath
`outputs/agent-proposals`; the entire `outputs/` tree is intentionally local
and gitignored.

## 6. Build package artifacts

After the tests pass:

```bash
python -m build
```

The generated `build/`, `dist/`, virtual environment, caches, and runtime
outputs are machine-local artifacts and are excluded from version control.
