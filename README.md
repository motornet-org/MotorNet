# MotorNet

This repository contains **MotorNet**, a python package that allows training recurrent neural networks to control for
biomechanically realistic effectors. This toolbox is designed to meet several goals:

- No hard dependencies beyond typically available python packages. This should make it easier to run this package on remote computing units.
- Provide users with a variety of muscle types to chose from.
- Flexibility in creating new, arbitrary muscle wrappings around the skeleton, to enable fast exploration of
different potential effectors and how to control them. The moment arm are calculated online according to the 
geometry of the skeleton and the (user-defined) paths of the muscles.
- User-friendly API, to allow for easier familiarization and intuitive usage. We want to enable a focus on ideas, not implementation.
The toolbox focuses on subclassing to allow users to implement their custom task designs, custom plants, and custom controller networks.
- Open-source, to allow for user contribution and community-driven incremental progress.

## State of the project

The package is feature complete, and is used by several people to progress in their research.
We are currently in beta phase, meaning the toolbox is publicly available but we are still on the lookout for potential
bugs and fixes to apply. Please feel free to log an issue if you think you found a bug, we appreciate any contribution. 
Stay tuned for more!

An [online documentation](https://motornet.org) is available. Feel free to check it out.

## How to Install

### Install from source

To install the latest development release, you can install directly from GitHub's repository.

```
pip install git+https://github.com/OlivierCodol/MotorNet.git
```

### Install with `pip`

First, please make sure that the latest `pip` version is installed in your working environment.

```
python3 -m pip install -U pip
```

Then you can install `motornet` using a simple `pip install` command.
```
python3 -m pip install motornet
```

### Install with Anaconda

Installation via Anaconda is currently not supported.


## Requirements

There is no third-party software required to run MotorNet. However, some freely available python dependencies are 
required.

If you are running the current development release (version 0.2.0), the requirements are as follows.

- [PyTorch](https://pytorch.org/docs/stable/torch.html): MotorNet relies on PyTorch to create tensors and build the 
graph.
- [NumPy](https://numpy.org/): For array and matrix computations when not using tensors.
- [Gymnasium](https://numpy.org/): `motornet` environments are child classes of `gymnasium` environments.
- [Matplotlib](https://matplotlib.org/): For plotting utilities, mainly in the 
[plotor](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/plotor.py) module.


If you are running a release version < 0.2.0, which relies on TensorFlow, the requirements are as follows.

- [TensorFlow](https://www.tensorflow.org/): MotorNet is first and foremost built on TensorFlow. However, the standard
TensorFlow toolbox is not compatible with recent Apple machines equipped with M1 silicon chips, and users must rely on 
an adapted version called [tensorflow-macos](https://pypi.org/project/tensorflow-macos/). When installing MotorNet, the 
`setup.py` routine will automatically check the machine's OS platform and hardware to assess whether to solve for the 
`tensorflow` or `tensorflow-macos` dependency. 
- [NumPy](https://numpy.org/): For array and matrix computations when not using tensors.
- [Matplotlib](https://matplotlib.org/): For plotting utilities, mainly in the 
[plotor](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/utils/plotor.py) module.
- [IPython](https://ipython.org/): Mainly for
[callbacks](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/nets/callbacks.py) that output training 
metrics during model training.
- [joblib](https://joblib.readthedocs.io/en/latest/): For parallelization routines in the 
[parallelizer](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/utils/parallelizer.py) script.


## Tutorials

There are several tutorials available to get you started, available in the
[repository](https://github.com/OlivierCodol/MotorNet)'s
[<em>examples</em>](https://github.com/OlivierCodol/MotorNet/tree/master/examples) folder, as well as on the 
[documentation website](https://motornet.org). Hopefully they will give a sense
of how the  API is supposed to work.

Tutorials and API documentation for version 0.1.5 are still available on the website and GitHib repository for those
who wish to consult them. They will remain available for the foreseeable future.

## Citation

If you use MotorNet in your research, please cite the following paper:

> Codol O, Michaels JA, Kashefi M, Pruszynski JA, Gribble PL (2023). MotorNet: a Python toolbox for controlling differentiable biomechanical effectors with artificial neural networks. *eLife* 12:RP88591. https://doi.org/10.7554/eLife.88591.3

BibTeX:

```bibtex
@article{Codol_2024,
  title     = {MotorNet: a Python toolbox for controlling differentiable biomechanical effectors with artificial neural networks},
  author    = {Codol, Olivier and Michaels, Jonathan A and Kashefi, Mehrdad and Pruszynski, J Andrew and Gribble, Paul L},
  journal   = {eLife},
  volume    = {12},
  pages     = {RP88591},
  year      = {2024},
  month     = {July},
  publisher = {eLife Sciences Publications, Ltd},
  doi       = {10.7554/eLife.88591.3},
  url       = {https://doi.org/10.7554/eLife.88591.3}
}
```

## Changelog

See [here](https://motornet.org/documentation/changelog) for a curated log of update contents.
