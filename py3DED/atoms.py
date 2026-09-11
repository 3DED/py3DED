from __future__ import annotations

import dataclasses
from typing import Optional

import ase.io
import dask
import numpy as np
from abtem.atoms import cut_ball, cut_disk
from ase import Atoms

from py3DED.config import BaseConfig, parse_arguments
from py3DED.io import filename_from_config, save_atoms_to_zarr


@dataclasses.dataclass
class Config(BaseConfig):
    """
    Configuration object for generating a supercell of atoms tbhe can be rotated and cropped to a box.

    Parameters
    ----------
    atoms : str | Atoms
        Path to a file readable by ASE or an ASE Atoms object.
    shape : str
        The shape of the created atoms, either "disk" or "ball". Using 'disk' will create a disk-shaped supercell
        allowinf for rotation around a single axis, while 'ball' will create a ball-shaped supercell, allowing for
        rotation around any axis.
    box : str | tuple[float, float, float]
        The size of the box to crop the supercell to in Angstroms.
    rotation_axis : float
        The rotation axis in degrees, used when shape is "disk".
    store_path : str
        The path to store the generated atoms in a zarr file.
    vacancies : float
        The fraction of vacancies to add to the generated atoms. A value of 0.0 will not remove any atoms, while a
        value of 1.0 will remove all atoms.
    seed : int
        The seed for the random number generator used to add vacancies.
    rotation_range : tuple[float, float] | None
        If the disk will only ever be rotated within this angular range [deg] (matching the actual
        rotation_min/rotation_max of the simulation), restricts the disk to just the atoms that can appear inside
        the box for some angle in that range, instead of sizing it to survive an arbitrary angle -- exact, not an
        approximation, and can shrink the disk substantially for a limited tilt series on a thick sample. Only
        used when shape is "disk"; None keeps the previous full-disk behavior.
    """

    atoms: str | Atoms = "structures/si.cif"
    shape: str = "disk"
    box: str | tuple[float, float, float] = "100,100,1000"
    rotation_axis: float = 0.0
    store_path: str = "structures/{atoms}_{shape}.zarr"
    vacancies: float = 0.0
    seed: int = 1337
    rotation_range: tuple[float, float] | None = None


def validate_box(box: str | tuple[float, float, float]):
    if isinstance(box, str):
        box = np.array(tuple(float(d) for d in box.split(",")))
    else:
        box = np.array(box)
    assert len(box) == 3
    return box


def add_vacancies(atoms: Atoms, vacancies: float, seed: int) -> Atoms:
    if hasattr(atoms, "compute"):
        atoms = dask.delayed(add_vacancies)(atoms, vacancies, seed)

    np.random.seed(seed)
    mask = np.random.rand(len(atoms)) > vacancies
    atoms = atoms[mask]
    return atoms


def make_atoms(config: Optional[Config] = None):
    if config is None:
        config = parse_arguments(Config)
    box = validate_box(config.box)
    if isinstance(config.atoms, str):
        atoms = ase.io.read(config.atoms)
    else:
        atoms = config.atoms

    if config.shape == "disk":
        atoms = cut_disk(
            atoms,
            box,
            rotation_axis=config.rotation_axis,
            rotation_range=config.rotation_range,
        )
    elif config.shape == "ball":
        atoms = cut_ball(atoms, box)
    else:
        raise ValueError(f"Shape {config.shape} not supported")

    atoms = add_vacancies(atoms, config.vacancies, config.seed)
    return atoms


def run(config: Optional[Config] = None):
    if config is None:
        config = parse_arguments(Config)

    atoms = make_atoms(config)
    store_path = filename_from_config(config.store_path, config)
    save_atoms_to_zarr(store_path, atoms)


if __name__ == "__main__":
    run()
