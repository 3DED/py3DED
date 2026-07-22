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

On one test system, using the latest cupy-cuda12x version (14.*) resulted in a drastically longer runtime. This was not further investigated.

## Example
Using the Python programs in the `scripts` folder, you can generate and analyse data sets like this:
```
python run_py3DED.py path_to_input.json
```
This requires that `path_to_input.json` defines an absolute or relative path to an existing CIF file. Depending on the hardware and settings, this will run for a couple of minutes or for many hours.

The results from the multislice calculations are written to a folder called `ms.zarr` in a subfolder with a unique name based on the current date, time, and CIF file, and selected settings. Bloch wave results are written to `bw.zarr`.

To plot the intensity of a certain reflection, a script like this can be used:
```python
import abtem

# read data as xarray
data = abtem.from_zarr(r"path/to/bw.zarr").to_data_array()

# depending on next steps, you may want to load the entire data set into memory.
# this is needed to, e.g., identify the strongest reflections
# data = data.compute() 

# plot intensity of reflection '0 0 0' at a thickness close to 500 Å
data_bw.sel(hkl='0 0 0').sel(z=500.0, method='nearest').plot()

# plot intensity of integrated rocking curve of the reflection
# with ID 42 as a function of thickness
data_bw.isel(hkl=42).sum(dim='x_rotation').plot()
```

The script `run_py3DED_hkl+Rint.py` can be used to write hkl files as a function of thickness. Intensities correspond to Lorentz-corrected, integrated intensities. If a CIF file is found with the relevant symmetry, the internal R factor is calculated as a function of thickness.

```
python run_py3DED_hkl+Rint.py path_to_bw_or_ms/bw.zarr
```

If `ms.zarr` and a `bw.zarr` are generated with the same simulation run, i.e., with identical settings and thus with the same set of reflections in the data set, the two files can be compared with `run_py3DED_compare2.py`. The script furthermore extracts some example rocking curves and quantifies the overall agreement.

```
python run_py3DED_compare2.py path_to_bw_AND_ms_directory/
```

## Example runtimes
Multislice calculations of 451 orientations using the input files in `examples/Si_orientation_omega/*.json` took about 15 hours, the Bloch wave calculations took 4 minutes. The calculations ran on a Windows desktop computer with a GeForce RTX 3060 GPU.

Multislice calculations of 16 orientations using the input files in `examples/convergence_test/Si_MS_boxsize/*.json` took between 2 minutes (`Si_box_25.json`) and 60 minutes (`Si_box_250.json`).

## Compatible *ab*TEM versions
The current version of py3DED requires the version with git commit hash `c6cdd78b` (version 1.0.6, from 16 October 2024).
This version is automatically installed with the above instructions.

Note that py3DED is not compatible with the latest version of *ab*TEM. Future versions of py3DED will catch up.

