# [insert name here] 

This repository contains code and configuration for processing and analysing images of plankton samples. 

It's a sister project to an image annotation app that is not yet released, written by researchers and data scientists at the UK Centre for Ecology and Hydrology in the early stages of a collaborative project that wasn't taken forward.

## Installation

### Python environment setup

Use anaconda or miniconda to create a python environment using the included `environment.yml`

```
conda create -n cyto_39 python=3.9
conda env update
```

Please note that this is specifically pinned to python 3.9 due to dependency versions; we make experimental use of the [https://sci.vision/#/model/resnet50-plankton](CEFAS plankton model available through SciVision), which in turn uses an older version of pytorch that isn't packaged above python 3.9.

### Object store connection

`.env` contains environment variable names for S3 connection details for the [JASMIN object store](https://github.com/NERC-CEH/object_store_tutorial/). Fill these in with your own credentials. If you're not sure what the `ENDPOINT` should be, please reach out to one of the project contributors listed below. 


### Package installation

Get started by cloning this repository and running

`pip install -e .`

### Running tests

`python -m pytest` or `py.test`

## Contents

### Catalogue creation

`scripts/intake_metadata.py` is a proof of concept that creates a configuration file for an [intake](https://intake.readthedocs.io/en/latest/) catalogue - a utility to make reading analytical datasets into analysis workflows more reproducible and less effortful.

### Feature extraction

Experiment testing workflows by using [https://sci.vision/#/model/resnet50-plankton](this plankton model from SciVision) to extract features from images for use in similarity search, clustering, etc. 

### TBC (object store upload, derived classifiers, etc)


## Contributors

[Jo Walsh](https://github.com/metazool/)
[Alba Gomez Segura](https://github.com/albags)
[Ezra Kitson](http://github.com/Kzra)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md)

