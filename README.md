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
pip install git+https://github.com/3DED/py3DED.git
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
This requires that `path_to_input.json` gives the path to an existing CIF file (`cif_file`) and an output folder (`output_folder`). Relative paths are taken relative to the folder of the JSON file, not the current directory, and the output folder is created if needed. Running `run_py3DED.py` without arguments prints an example input file. Depending on the hardware and settings, a run takes from a couple of minutes to many hours.

Each run writes to a new subfolder of `output_folder`, named after the current date and time, the CIF file and selected settings. It contains:

- `ms.zarr.zip`: the multislice results;
- `bw.zarr.zip`: the Bloch-wave results;
- `supercell.zarr.zip`: the crystallite used for multislice;
- `settings.json`, a copy of the CIF file, and the log `py3DED.log`.

Runs made with versions before this one wrote plain `.zarr` folders instead of `.zarr.zip` files. All scripts read both.

To plot the intensity of a certain reflection, a script like this can be used:
```python
import abtem

# read data as xarray
data = abtem.from_zarr(r"path/to/bw.zarr.zip").to_data_array()

# depending on next steps, you may want to load the entire data set into memory.
# this is needed to, e.g., identify the strongest reflections
# data = data.compute() 

# plot intensity of reflection '0 0 0' at a thickness close to 500 Å
data.sel(hkl='0 0 0').sel(z=500.0, method='nearest').plot()

# plot intensity of integrated rocking curve of the reflection
# with ID 42 as a function of thickness
data.isel(hkl=42).sum(dim='x_rotation').plot()
```

The script `run_py3DED_hkl+Rint.py` can be used to write hkl files as a function of thickness. Intensities correspond to Lorentz-corrected, integrated intensities. If a CIF file is found with the relevant symmetry, the internal R factor is calculated as a function of thickness.

```
python run_py3DED_hkl+Rint.py path_to_run_folder/bw.zarr.zip
```

If `ms.zarr.zip` and `bw.zarr.zip` are generated in the same simulation run, i.e., with identical settings and thus with the same set of reflections in the data set, the two can be compared with `run_py3DED_compare2.py`. The script also extracts some example rocking curves and quantifies the overall agreement.

```
python run_py3DED_compare2.py path_to_run_folder/
```

The multislice intensities are already corrected for the power gain of the detector window (`window_func`), so `compare2` applies no further scaling by default (`'scale': False`). For data written by versions before this one, which were not corrected, set `'scale': 'Hann'`.

## Further settings
Besides the settings printed by `run_py3DED.py`, the input file accepts:

- **`propagator`:** `"paraxial"` (default) or `"exact"`. It sets the propagator for both methods, so that they solve the same equation:
  - `"paraxial"` uses the first-order Fresnel propagator for multislice, and the Bloch waves of `use_wave_eq` (default `true`);
  - `"exact"` uses the exact propagator for multislice, and the matching non-paraxial Bloch waves. This requires `use_wave_eq: true`. The exact Bloch waves need a larger `sg_max` to converge.
- **`energy`:** a single value, or a list of energies to simulate together, e.g. `[100000, 200000, 300000]`. The results then have an `Energy` dimension. See `examples/Si_kinetic_energy/all_energies.json`.
- **`to_cpu`** (default `false`): copy each multislice chunk to the host as it is computed. The default keeps the result on the GPU until the end, which is faster.
- **`potential_chunk_size`** (default `"auto"`): the number of potential slices computed at once. A larger value can be much faster on a GPU with memory to spare, but can run out of memory, especially for energy lists.

## Further scripts
- **`plot_rz_vs_energy.py`:** for a run with a list of energies, plots the Bragg R factor of the integrated intensities against thickness, with one curve per energy.

  ```
  python plot_rz_vs_energy.py path_to_run_folder/
  ```
- **`run_convergence_sweep.sh`:** runs every input file in one folder of `examples/convergence_tests` (e.g. `Si_MS_boxsize`).
- **`run_convergence_analysis.sh`:** runs `run_py3DED_hkl+Rint.py` and `run_py3DED_compare2.py` on each result of such a sweep.
- **`plot_convergence.py`:** plots the convergence of the sweep against the swept parameter.

Each script describes its options at the top of the file.

## Changes in results since V2026.7
Results from this version differ from those of V2026.7 in three ways; all three are corrections:

- **Bloch waves:** *ab*TEM applied the Bloch-wave metric inconsistently, so the paraxial Bloch waves (`use_wave_eq: true`) did not solve the same equation as the first-order Fresnel multislice. The fix changes the Bloch-wave intensities, mostly in weak frames at large thickness, and improves the agreement with multislice.
- **Non-cubic crystals:** for a non-cubic unit cell with a nonzero `rotation_axis_orientation`, the crystallite cut for multislice did not fill the simulation box. Cubic cells are unaffected.
- **Sampling:** grid sizes are rounded to fast FFT sizes, which shifts the real-space sampling slightly from the requested value.

## Example runtimes
These times were measured with V2026.7. This version's multislice is faster.

Multislice calculations of 451 orientations using the input files in `examples/Si_orientation_omega/*.json` took about 15 hours, the Bloch wave calculations took 4 minutes. The calculations ran on a Windows desktop computer with a GeForce RTX 3060 GPU.

Multislice calculations of 16 orientations using the input files in `examples/convergence_tests/Si_MS_boxsize/*.json` took between 2 minutes (`Si_box_25.json`) and 60 minutes (`Si_box_250.json`).

## Compatible *ab*TEM versions
This branch requires abTEM commit [`cabdb3ba`](https://github.com/abTEM/abTEM/commit/cabdb3bae71879f66bf21894f1756aefd1f30911), the head of [abTEM#644](https://github.com/abTEM/abTEM/pull/644). That PR is the top of a stack on [abTEM#384](https://github.com/abTEM/abTEM/pull/384), which upstreams py3DED's crop/window, finite-crystal and rotation-series tooling into abTEM; #644 adds the multislice speedups. It is installed automatically with the instructions above. Once the stack is merged into abTEM's `dev`, the pin should move to that merge commit or to a released version.

