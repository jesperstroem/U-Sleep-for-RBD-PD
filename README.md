# Automatic Sleep Staging for RBD and PD Populations

[![CI](https://github.com/jesperstroem/U-Sleep-for-RBD-PD/actions/workflows/ci.yml/badge.svg)](https://github.com/jesperstroem/U-Sleep-for-RBD-PD/actions/workflows/ci.yml)

This repository shows how to install and use the U-Sleep model fine-tuned for individuals with
REM Sleep Behavior Disorder (RBD) and Parkinson's Disease (PD).

Two model variants are included in `weights/`:

| Model | Suitable for |
|-------|-------------|
| `Pretrained_Model.ckpt` | General healthy population |
| `Generalized_Model.ckpt` | Healthy population **and** individuals with RBD / PD |

Supported input formats: any MNE-compatible file (`.set`, `.vhdr`, `.edf`).

Tested on a local Windows machine (CPU) and a compute cluster (NVIDIA GPU, CUDA 11.8).

Questions? Write to: js@ece.au.dk

---

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [Git](https://git-scm.com/install/)

### Step 1 — Clone the repository

```bash
git clone https://github.com/jesperstroem/U-Sleep-for-RBD-PD
cd U-Sleep-for-RBD-PD
```

### Step 2 — Create and activate a conda environment

```bash
conda create -n usleep-rbdpd python=3.10
conda activate usleep-rbdpd
```

### Step 3 — Install PyTorch

PyTorch must be installed **before** the package so pip picks the correct CPU or GPU wheel.
Choose **one** of the following:

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Install the package

```bash
pip install .
```

This pulls in `sleep_preprocessing_pipeline` and `ml_architectures` from the project's
private GitLab instance, along with `numpy` and `matplotlib`.

---

## Demo

Open `demo.ipynb` in Jupyter to run sleep staging on your own MNE-compatible file.
The notebook walks through:

1. Exploring available channels in your file
2. Building a dataset and running inference
3. Plotting the predicted hypnogram
4. Visualising per-stage confidence scores
5. Inspecting the sleep-stage distribution
6. Comparing the Pretrained and Generalized models side-by-side

Start Jupyter with:

```bash
pip install jupyter
jupyter notebook demo.ipynb
```

---

## Development Workflow

### Install dev tools

```bash
pip install ".[dev]"
```

### Lint and format

```bash
ruff check .          # lint
ruff format .         # auto-format
ruff format --check . # format check only (non-destructive)
```

### Run tests

```bash
pytest tests/ -v
```

### Pre-commit hooks (recommended)

```bash
pip install pre-commit
pre-commit install     # runs ruff automatically before each commit
```

---

## CI/CD

Every push and pull request triggers the GitHub Actions pipeline:

- **Lint & Format** — Ruff (no external deps, ~30 s)
- **Build Wheel** — validates `pyproject.toml` packaging
- **Structural Tests** — checks weights, notebook structure, and packaging (no private deps)

See [docs/ci-cd.md](docs/ci-cd.md) for full details.

---

## Project Structure

```
U-Sleep-for-RBD-PD/
├── weights/
│   ├── Pretrained_Model.ckpt   # General population model
│   └── Generalized_Model.ckpt  # RBD/PD-adapted model
├── demo.ipynb                  # End-to-end inference demo
├── pyproject.toml              # Package metadata and tooling config
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── tests/                      # Structural tests (no external deps)
└── docs/ci-cd.md               # CI/CD documentation
```
