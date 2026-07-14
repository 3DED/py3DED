# py3DED - Simulation of 3D ED data sets

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

py3DED is a Python package that makes use of the *ab*TEM package to simulate 3D electron diffraction data using either the Bloch wave approach or the multislice approach.

Simulated data sets consist of diffracted intensities as a function of Laue indices *hkl*, thickness, and orientation. For each reflection *hkl* and thickness, rocking curves (reflections profiles) can be extracted. Data sets can be further processed to generate, e.g., typical hkl files for further crystallographic analysis.

With a minimal input, i.e. a configuration file in JSON format and a CIF file of the crystal structure model, are sufficient to generate a data set.

py3DED is also useful to compare multislice with Bloch wave calculations.

## Publication
A manuscript is in preparation and will be published soon. A preprint will be made available on chemarxiv.


## Installation in virtual environment
Create a new virtual environment:
```sh
python -m venv py3DED
```

Activate the new environment (Windows 7/10/11):
```sh
.\py3DED\Scripts\Activate.ps1
```

You can install py3DED using `pip`:
```sh
pip install py3DED
```

## GPU-support
If you have a CuPy-compatible GPU and the CUDA Toolkit installed, a GPU will drastically speed up calculations of *ab*TEM.

Install CuPy using `pip`:
```sh
pip install cupy-cuda12x==13.6.0
```

This setup was tested with an Nvidida GeForce RTX 3060 and [CUDA Toolkit 12.9](https://developer.nvidia.com/cuda-12-9-1-download-archive) installed.

Note that you need CUDA Toolkit Version 12 and the corresponding version of cupy.



## Compatible *ab*TEM versions
The current version of py3DED requires the version with git commit hash `c6cdd78b` (version 1.0.6, from 16 October 2024).
This version is automatically installed with the above instructions.

Note that py3DED is not compatible with the latest version of *ab*TEM. Future versions of py3DED will catch up.



