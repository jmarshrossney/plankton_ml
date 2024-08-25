# Plankton ML

This repository contains code and configuration for processing and analysing images of plankton samples. It's experimental, serving as much as a proposed template for new projects than as a project in itself.

It's a companion project to an R-shiny based image annotation app that is not yet released, written by researchers and data scientists at the UK Centre for Ecology and Hydrology in the early stages of a collaboration that was placed on hold.

## Setup

### Environment setup and package installation

#### Using pip

Create a fresh virtual environment in the repository root using Python >=3.12 and (e.g.) `venv`: 

```
python -m venv venv
```

Next, install the package using `pip`:

```
python -m pip install .
```

Most likely you are interested in developing and/or experimenting, so you will probably want to install the package in 'editable' mode (`-e`), along with dev tools and jupyter notebook functionality

```
python -m pip install -e .[dev,jupyter]
```

#### Using conda

Use anaconda or miniconda to create a python environment using the included `environment.yml`

```
conda env create -f environment.yml
conda activate cyto_ml
```

Next install this package _without dependencies_:

```
python -m pip install --no-deps -e .
```

### Object store connection

`.env` contains environment variable names for S3 connection details for the [JASMIN object store](https://github.com/NERC-CEH/object_store_tutorial/). Fill these in with your own credentials. If you're not sure what the `ENDPOINT` should be, please reach out to one of the project contributors listed below. 

### Running tests

Run `pytest` in the root of the repository


#### Reproducible conda environments

## Contents

### Catalogue creation

`scripts/intake_metadata.py` is a proof of concept that creates a configuration file for an [intake](https://intake.readthedocs.io/en/latest/) catalogue - a utility to make reading analytical datasets into analysis workflows more reproducible and less effortful.

### Feature extraction

Experiment testing workflows by using [this plankton model from SciVision](https://sci.vision/#/model/resnet50-plankton) to extract features from images for use in similarity search, clustering, etc.

### Running Jupyter notebooks

The `notebooks/` directory contains Markdown (`.md`) representations of the notebooks.
To create Jupyter notebooks (`.ipynb`), run the following command with the conda environment activated:

```sh
jupytext --sync notebooks/*
```

If you modify the contents of a notebook, run the command after closing the notebook to re-sync the `.ipynb` and `.md` representations before committing.

For more information see the [Jupytext docs](https://jupytext.readthedocs.io/en/latest/).


### TBC (object store upload, derived classifiers, etc)


## Reproducible environments

In some situations, e.g. running experiments intended for publication, you might want to be able to reproduce exactly your working environment.

### Reproducible Python environments

Rather than installing via `python -m pip install -e .[dev,jupyter]` as described above, you can install all dependencies using the `requirements.txt` provided, and then install `cyto_ml` with the `--no-deps` option

```
python -m venv venv
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

To update the lockfile you can run

```
pip-compile --upgrade
```

Note that non-python dependencies (such as cuda, blas etc.) are not locked using this approach. If this matters to you consider using `conda` environments as described next.


### Reproducible Conda environments

Rather than installing via `conda env create -f environment.yml` as described above, you can use the lockfile provided, which is called `conda-lock.yml`.

For this you need [`conda-lock`](https://github.com/conda/conda-lock) to be already installed. Although this can be done easily by a `pip install` into your system Python or `conda install` into your `base` environment, it is preferable to us `pipx` or `condax` as described in the `conda-lock` installation instructions.

Once you have `conda-lock`, you can simply run

```
conda-lock install
conda activate cyto_ml
python -m pip install --no-deps .
```

## Contributors

[Jo Walsh](https://github.com/metazool/)
[Alba Gomez Segura](https://github.com/albags)
[Ezra Kitson](http://github.com/Kzra)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md)

