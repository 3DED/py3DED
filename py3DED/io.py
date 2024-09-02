import os

import xarray as xr
import numpy as np
import ase
import dask


def atoms_to_xarray(atoms):
    atomic_positions = atoms.positions  # replace with your actual data
    atomic_numbers = atoms.numbers  # replace with your actual data
    unit_cell = atoms.cell

    ds = xr.Dataset(
        data_vars={
            "atomic_positions": (("atom", "dimension"), atomic_positions),
            "atomic_numbers": (("atom",), atomic_numbers),
            "unit_cell": (("cell_row", "cell_column"), unit_cell),
        },
        coords={
            "atom": np.arange(len(atoms)),
            "dimension": ["x", "y", "z"],
            "cell_row": ["a", "b", "c"],
            "cell_column": ["a", "b", "c"],
        },
    )
    return ds


def strip_to_filename(file_name):
    return os.path.splitext(os.path.split(file_name)[-1])[0]


def filename_from_config(filename, config):
    config_dict_store = {k: v for k, v in config.__dict__.items()}
    return config.store_path.format(**config_dict_store)


def xarray_to_atoms(ds):
    
    positions = ds.atomic_positions.data
    numbers = ds.atomic_numbers.data
    cell = ds.unit_cell.data
    atoms = ase.Atoms(numbers, positions=positions, cell=cell)
    return atoms


def save_atoms_to_zarr(filename, atoms):
    ds = atoms_to_xarray(atoms)
    ds.to_zarr(filename, mode="w")


def read_atoms_from_zarr(filename, lazy=False):
    ds = xr.open_zarr(filename)

    if lazy:
        return dask.delayed(xarray_to_atoms)(ds)
    else:
        ds = ds.compute()
        return xarray_to_atoms(ds)