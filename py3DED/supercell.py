import dataclasses
from numbers import Number

import ase.io
from ase import Atoms

from abtem import AtomsEnsemble
from abtem.atoms import euler_to_rotation
import numpy as np
from py3ded.config import BaseConfig, parse_arguments
from abtem.atoms import is_cell_orthogonal
import dask

from py3ded.io import save_atoms_to_zarr, strip_to_filename, filename_from_config

from abtem.core.axes import NonLinearAxis


@dataclasses.dataclass
class SupercellConfig(BaseConfig):
    atoms: str = "structures/si.cif"
    shape: str = "disk"
    box: str | tuple[float, float, float] = "100,100,1000"
    rotation_axis: float = 0.0
    store_path: str = "structures/{atoms}_{shape}.zarr"
    store_path_keep_only_name: bool = True


def validate_box(box: str | tuple[float, float, float]):
    if isinstance(box, str):
        box = np.array(tuple(float(d) for d in box.split(",")))
    else:
        box = np.array(box)
    assert len(box) == 3
    return box


def rotate_positions(positions, center, ai=0, aj=0, ak=0, axes="zxz"):
    R = euler_to_rotation(ai, aj, ak, axes)
    positions = np.dot((positions - center), R.T) + center
    return positions


def rotate_atoms(atoms: Atoms, center="COU", ai=0, aj=0, ak=0, axes="zxz"):

    if center == "COU":
        center = atoms.cell.sum(0) / 2
    elif isinstance(center, str):
        raise NotImplementedError

    positions = rotate_positions(atoms.positions, center, ai, aj, ak, axes)

    atoms = atoms.copy()
    atoms.positions[:] = positions
    return atoms


def min_circumscribed_radius(box):
    return np.linalg.norm(box) / 2


def transform_positions(basis, positions):
    return np.linalg.solve(basis.T, positions.T).T


def rotate_positions(positions, center, ai=0, aj=0, ak=0, axes="zxz"):
    R = euler_to_rotation(ai, aj, ak, axes)
    positions = np.dot((positions - center), R.T) + center
    return positions


def cube_points(L):
    if isinstance(L, Number):
        L = (L, L, L)

    x, y, z = np.mgrid[0 : L[0] + 1 : L[0], 0 : L[1] + 1 : L[1], 0 : L[2] + 1 : L[2]]
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return points


def repeated_lattice(cell, reps):
    xyz = tuple(np.linspace(0, n, n + 1) for n in reps)
    x, y, z = np.meshgrid(*xyz, indexing="ij")
    lattice = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    lattice = transform_positions(np.linalg.inv(cell), lattice)
    return lattice


def mask_box(positions, box):
    mask = (
        (positions[:, 0] >= 0)
        & (positions[:, 0] < box[0])
        & (positions[:, 1] >= 0)
        & (positions[:, 1] < box[1])
        & (positions[:, 2] >= 0)
        & (positions[:, 2] < box[2])
    )
    return mask


def crop_to_cell(atoms):

    if not is_cell_orthogonal(atoms):
        raise NotImplementedError

    box = np.diag(atoms.cell)

    return atoms[mask_box(atoms.positions, box)]


def center_to_box(positions, box):
    positions = positions - positions.mean(0) + box / 2
    return positions


def make_lattice_disk(cell, box, rotation_axis: float = 0.0):
    rotation_axis = np.rad2deg(rotation_axis)
    box = np.array(box)

    width = np.linalg.norm(box[:2]) + 2 * np.linalg.norm(cell)
    height = np.linalg.norm(box) + np.linalg.norm(cell)

    large_box = np.array((width, height, height))
    large_cube = cube_points(large_box)

    rotated_cell = rotate_positions(cell, (0, 0, 0), rotation_axis)

    transformed_cube = transform_positions(rotated_cell, large_cube)
    reps = np.ceil(transformed_cube.ptp(0)).astype(int)

    lattice = repeated_lattice(cell, reps)
    lattice = rotate_positions(lattice, (0, 0, 0), -rotation_axis)

    lattice = center_to_box(lattice, large_box)
    lattice = lattice[mask_box(lattice, large_box)]

    lattice = lattice - lattice.mean(0)
    lattice = lattice[np.linalg.norm(lattice[:, 1:], axis=-1) < height / 2]

    shift = cell.sum(0) / 2
    lattice = center_to_box(lattice, box) - shift
    lattice = rotate_positions(lattice, box / 2, rotation_axis)
    return lattice


def make_lattice_ball(cell, box):
    box = np.array(box)

    radius = min_circumscribed_radius(box)

    margin = np.linalg.norm(cell.sum(0)) / 2
    L = 2 * (radius + margin)

    cube = cube_points(L)

    transformed_cube = transform_positions(cell, cube)
    reps = np.ceil(transformed_cube.ptp(0)).astype(int)

    lattice = repeated_lattice(cell, reps)
    
    lattice -= lattice.mean(0)
    d = np.linalg.norm(lattice, axis=-1)
    lattice = lattice[d < radius]

    shift = cell.sum(0) / 2
    lattice = center_to_box(lattice, box) - shift
    return lattice


def make_perfect_supercell(atoms, box, shape, **kwargs):

    if shape == "ball":
        lattice = make_lattice_ball(atoms.cell, box)
    elif shape == "disk":
        lattice = make_lattice_disk(atoms.cell, box, **kwargs)
    else:
        raise ValueError(f"Shape {shape} not supported")

    positions = (lattice[:, None] + atoms.positions[None]).reshape(-1, 3)
    numbers = np.repeat(atoms.numbers, len(lattice))
    atoms = ase.Atoms(numbers, positions, cell=box, pbc=True)
    return atoms


def _rotate_and_crop_to_cell(atoms, rotation, rotation_axis=0.0):
    rotated_atoms = rotate_atoms(
        atoms,
        center="COU",
        ai=-rotation_axis,
        aj=rotation,
        ak=rotation_axis,
        axes="zxz",
    )
    cropped_atoms = crop_to_cell(rotated_atoms)
    return cropped_atoms


def make_rotated_atoms_ensemble(atoms, rotations):
    func = dask.delayed(_rotate_and_crop_to_cell)

    trajectory = [func(atoms, rotation) for rotation in rotations]

    axis_metadata = NonLinearAxis(label="x_rotation", units="deg", values=rotations)

    ensemble = AtomsEnsemble(
        trajectory, ensemble_mean=False, ensemble_axes_metadata=axis_metadata
    )
    return ensemble


def make_atoms(config=None):
    if config is None:
        config = parse_arguments(SupercellConfig)
    box = validate_box(config.box)
    atoms = ase.io.read(config.atoms)
    shape = config.shape
    kwargs = {"rotation_axis": config.rotation_axis}
    atoms = make_perfect_supercell(atoms, box, shape, **kwargs)
    return atoms


def main(config=None):
    if config is None:
        config = parse_arguments(SupercellConfig)
    box = validate_box(config.box)
    atoms = ase.io.read(config.atoms)
    shape = config.shape
    kwargs = {"rotation_axis": config.rotation_axis}

    atoms = make_perfect_supercell(atoms, box, shape, **kwargs)

    #if config.store_path_keep_only_name:
    #    config.atoms = strip_to_filename(config.atoms)

    store_path = filename_from_config(config.store_path, config)
    save_atoms_to_zarr(store_path, atoms)


if __name__ == "__main__":
    main()
