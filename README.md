# BootSTOP (Bootstrap STochastic OPtimizer)

## Overview

BootSTOP is a Python package for determining CFT data (scaling dimensions and OPE coefficients) which minimize a 
theory's truncated crossing equation. To do this, the code applies algorithms within the 
[PyGMO package](https://esa.github.io/pygmo2/).

## Installation

Unfortunately, PyGMO cannot be installed easily using pip. Instead, use [Anaconda](https://www.anaconda.com/docs/main) 
or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). In the following, 
Micromamba is used, but the commands are the same for Anaconda, replacing `micromamba` with `anaconda`.

To install the dependencies, run
```
micromamba env create -f conda_env.yml
```

Note that the dependencies in `conda_env.yml` are unpinned, which can be helpful in terms of locating proper versions 
for your operating environment, but may lead to dependency conflicts.

After installing the dependencies, activate the environment:
```
micromamba activate bootstop
```

In addition to the dependencies, `multiobjective` is a local package that needs to be installed. If it wasn't properly 
installed in the first step above, simply run this within your activated environment:
```
pip install -e .
```

## F Block Data

To run the code, you need to provide a set of F blocks for each spin in your spin partition. For optimizations of the 
3D Ising model, you need to provide these under `multiobjective/block_lattices/3d`. See one of the project creators 
to get access to these files.

## Running the Code

The main run scripts are located in the `multiobjective` folder with names beginning with `run`. These run scripts use 
a config file system, where the spin partition structure is specified in a JSON configuration file which is later 
loaded up. The config files are located in the `multiobjective/config_files` directory.

### PyGMO Optimization

To optimize the 3D Ising model using PyGMO, run the script called `run_pygmo_scalar_3d_ising.py`. To get a menu of 
options you can pass in, run
```
python multiobjective/run_pygmo_scalar_3d_ising.py --help
```
There are two main way of modeling the OPE coefficients: using four values or three values. To specify which one you 
want, use the `-l` option.

### PyTorch Optimization

There is also a more experimental optimization which involves modeling the OPE coefficients of the tail operators using 
a neural network. To get the options for this script, run
```
python3 run_pytorch_scalar_3d_ising.py --help
```
Use this in conjunction with the neural network config files like `config_nn.json`.

## Tests and Linting

To execute unit tests, run
```
pytest
```
For linting, run
```
flake8
```
