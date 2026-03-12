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

---

## Config Files

Example config files are provided under `multicorrelator/config_files/`, split into two
subdirectories:

### Optimiser configs (`config_files/optimiser/`)

These control the PyGMO optimisation settings and the conformal block backend.

| File | Description |
|------|-------------|
| `opt_config_3.json` | Standard run using the goblocks z-point interpolation backend |
| `opt_config_3_deriv_blocks.json` | Run using the goblocks derivative blocks backend |

Key fields:

| Field | Description |
|-------|-------------|
| `pop_size` | Population size for the evolutionary algorithm |
| `num_islands` | Number of parallel islands |
| `max_iter` | Maximum number of iterations per evolution |
| `evolutions` | Number of evolutions to run |
| `lambda_model` | OPE coefficient model: `"three_value_lambdas"` or `"four_value_lambdas"` |
| `scaling` | Whether to apply scaling to the search space |
| `F_block_interpolation.name` | Block backend: `"goblocks"` or `"goblocks_derivs"` |
| `use_wandb` | Set to `true` to log results to Weights & Biases |
| `outdir` | Directory where output files are written |

### Spin partition configs (`config_files/spin/`)

These define the operator spectrum — which spins and how many operators per spin — along with
the search bounds on scaling dimensions and OPE coefficients.

| File | Description |
|------|-------------|
| `config_3_lambdas.json` | Broad search over the 3D Ising spectrum |
| `config_3_lambdas_pbounds_25.json` | Tighter bounds based on a previous optimisation run |

### Quick start

To run with the provided example configs:

```bash
python multicorrelator/run_pygmo_scalar_3d_ising.py \
    --optimiser-config multicorrelator/config_files/optimiser/opt_config_3.json \
    --spin-config multicorrelator/config_files/spin/config_3_lambdas.json
```

## Citation

If you use this repository, please cite!

@article{Chryssanthacopoulos:2026GoBlocks,
  author  = {Chryssanthacopoulos, James and
             Niarchos, Vasilis and
             Papageorgakis, Constantinos and
             Stapleton, Alexander G.},
  title   = {Efficient Conformal Block Evaluation with GoBlocks},
  year    = {2026},
  eprint  = {2603.10627},
  archivePrefix = {arXiv},
  primaryClass  = {hep-th},
  doi     = {10.48550/arXiv.2603.10627},
  url     = {https://arxiv.org/abs/2603.10627}
}

@software{bootstop-multi-correlator,
  author  = {Chryssanthacopoulos, James and
             Niarchos, Vasilis and
             Papageorgakis, Constantinos and
             Stapleton, Alexander G.},
  title   = {{BootSTOP-multi-correlator}},
  year    = {2026},
  url     = {https://github.com/jchryssanthacopoulos/BootSTOP-multi-correlator}
}

@software{goblocks,
  author  = {Chryssanthacopoulos, James and
             Niarchos, Vasilis and
             Papageorgakis, Constantinos and
             Stapleton, Alexander G.},
  title   = {{GoBlocks}},
  year    = {2026},
  url     = {https://github.com/xand-stapleton/goblocks}
}
