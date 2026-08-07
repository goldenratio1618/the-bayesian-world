# Linux installation

The supported development environment is a recent 64-bit Linux distribution
with Python 3.10 or newer. A C99 compiler is required for the generated online
runtime checks. GPU-enabled PyTorch is the preferred simulator backend; NumPy
remains the explicit CPU fallback.

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

Then use the [official PyTorch Linux selector](https://pytorch.org/get-started/locally/)
to choose the stable `pip` wheel for the CUDA version supported by the installed
driver. For example, when CUDA 12.6 is the appropriate wheel channel:

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
contraption validate
```

`--expect cuda` exits nonzero if CUDA is unavailable, if the installed wheel is
CPU-only, if a device kernel fails, or if autograd does not propagate through a
trajectory. On a deliberate CPU installation, use `--expect cpu` instead.

Run the full GPU scanner fixture with:

```bash
contraption demo --backend torch --device cuda --output outputs/scanner_demo
```

## 5. Configure optional component agents

Agent calls read only `OPENAI_API_KEY`. Either export it in the shell or place
it in a local `.env` file. The CLI searches the repository and then its parent
directory, but requires `--env-file` if both locations contain a file:

```bash
contraption agent-canary --kind classification --env-file ../.env
contraption agent-run classification-all --env-file ../.env
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
