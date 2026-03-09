# BootSTOP Multi-Correlator

## Overview

BootSTOP Multi-Correlator is a Python package for determining CFT data (scaling dimensions and OPE
coefficients) which minimize a theory's truncated crossing equations in the multi-correlator setting.
Optimization is performed using algorithms from the [PyGMO package](https://esa.github.io/pygmo2/).

---

## Installation

### 1. Create the conda environment

PyGMO cannot be installed via pip and requires conda. Use either
[Anaconda](https://www.anaconda.com/docs/main) or
[Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
The commands are identical — just replace `conda` with `micromamba` if using Micromamba.

Create and activate the environment from the provided `conda_env.yml`:

```bash
conda env create -f conda_env.yml
conda activate multicorrelator
```

This installs all dependencies and the local `multicorrelator` package in editable mode.

### 2. Initialise the goblocks submodule

The conformal block engine is provided as a Git submodule. After cloning the repo, run:

```bash
git submodule update --init
```

### 3. Build the goblocks shared library

Go must be installed ([https://go.dev/doc/install](https://go.dev/doc/install)). Then, from within
the `goblocks/server/src` directory, run:

```bash
make
```

This produces:
- `goblocks/server/bin/goblocks` — the CLI binary
- `goblocks/server/lib/librecursive.so` — the shared library used by the Python client

For GPU-accelerated block evaluation:

```bash
make GPU=1
```

### 4. Install the goblocks Python client

From within your activated environment, install the Python wrapper:

```bash
pip install goblocks/client
```

---

## Running the Code

The main entry point is `multicorrelator/run_pygmo_scalar_3d_ising.py`. It takes two config files:
an optimiser config and a spin partition config.

To see all available options:

```bash
python multicorrelator/run_pygmo_scalar_3d_ising.py --help
```

### Arguments

| Flag | Short | Description |
|------|-------|-------------|
| `--optimiser-config` | `-o` | Path to the optimiser config YAML/JSON file |
| `--spin-config` | `-s` | Path to the spin partition config YAML/JSON file |

### Example

```bash
python multicorrelator/run_pygmo_scalar_3d_ising.py \
    --optimiser-config path/to/optimiser_config.json \
    --spin-config path/to/spin_config.json
```

### OPE coefficient models

Two models are available for parameterizing the OPE coefficients, selectable via the optimiser
config:

- **Four-value model** — uses four independent OPE coefficient values
- **Three-value model** — uses three independent OPE coefficient values

Specify the desired model using the `lambda_model` field in your optimiser config file.