# [insert name here] 

This repository contains code and configuration for processing and analysing images of plankton samples. 

It's a sister project to an image annotation app that is not yet released, written by researchers and data scientists at the UK Centre for Ecology and Hydrology in the early stages of a collaborative project that wasn't taken forward.

## Installation

### Environment setup

Use anaconda or miniconda to create a python environment using the included `environment.yml`

```
conda create -n cyto_39 python=3.9
conda env update
```

Please note that this is specifically pinned to python 3.9 due to dependency versions; we make experimental use of the [https://sci.vision/#/model/resnet50-plankton](CEFAS plankton model available through SciVision), which in turn uses an older version of pytorch that isn't packaged above python 3.9.

### Package installation

Get started by cloning this repository and running

`pip install -e .`

### Running tests

`pytest`

## Contents 


### Feature extraction

Experiment testing workflows by using [https://sci.vision/#/model/resnet50-plankton](this plankton model from SciVision) to extract features from images for use in similarity search, clustering, etc. 

### TBC (object store upload, derived classifiers, etc)


## Contributors

Jo Walsh
Alba Gomez Segura
Ezra Kitson

## Contributing

Please see [CONTRIBUTION.md](CONTRIBUTION.md)

