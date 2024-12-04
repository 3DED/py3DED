from __future__ import annotations

import dataclasses
from typing import Optional

import ase.io
import dask
import numpy as np
from abtem.atoms import euler_to_rotation, is_cell_orthogonal
from ase import Atoms
from ase.cell import Cell
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

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
    """

    atoms: str | Atoms = "structures/si.cif"
    shape: str = "disk"
    box: str | tuple[float, float, float] = "100,100,1000"
    rotation_axes: str = "zxz"
    rotations: tuple[float, float, float] = (0.0, 0.0, 0.0)
    store_path: str = "structures/{atoms}_{shape}.zarr"
    vacancies: float = 0.0
    seed: int = 1337


def validate_box(box: str | tuple[float, float, float]):
    if isinstance(box, str):
        box = np.array(tuple(float(d) for d in box.split(",")))
    else:
        box = np.array(box)
    assert len(box) == 3
    return box


def rotate_positions(
    positions: NDArray[np.float64],
    center: NDArray[np.float64],
    ai: float = 0,
    aj: float = 0,
    ak: float = 0,
    axes: str = "zxz",
) -> NDArray[np.float64]:
    R = euler_to_rotation(ai, aj, ak, axes)
    positions = np.dot((positions - center), R.T) + center
    return positions


def rotate_atoms(
    atoms: Atoms,
    center: str | np.ndarray = "COU",
    ai: float = 0,
    aj: float = 0,
    ak: float = 0,
    axes: str = "zxz",
) -> Atoms:
    if center == "COU":
        center = atoms.cell.sum(0) / 2
    elif isinstance(center, str):
        raise NotImplementedError

    positions = rotate_positions(atoms.positions, center, ai, aj, ak, axes)

    atoms = atoms.copy()
    atoms.positions[:] = positions
    return atoms


def min_circumscribed_radius(box: NDArray[np.float64]) -> float:
    return np.linalg.norm(box) / 2


def transform_positions(
    basis: NDArray[np.float64], positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.linalg.solve(basis.T, positions.T).T


def cube_points(L: float | tuple[float, float, float]) -> NDArray[np.float64]:
    if isinstance(L, float):
        L = (L, L, L)

    points = np.array(
        [
            [0, 0, 0],
            [L[0], 0, 0],
            [0, L[1], 0],
            [0, 0, L[2]],
            [L[0], L[2], 0],
            [L[0], 0, L[2]],
            [0, L[1], L[2]],
            [L[0], L[1], L[2]],
        ],
        dtype=np.float64,
    )
    return points


def repeated_lattice(
    cell: NDArray[np.float64], reps: tuple[int, int, int]
) -> NDArray[np.float64]:
    xyz = tuple(np.linspace(0, n, n + 1) for n in reps)
    x, y, z = np.meshgrid(*xyz, indexing="ij")
    lattice = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    lattice = transform_positions(np.linalg.inv(cell), lattice)
    return lattice


def mask_box(
    positions: NDArray[np.float64], box: NDArray[np.float64]
) -> NDArray[np.bool_]:
    mask = (
        (positions[:, 0] >= 0)
        & (positions[:, 0] < box[0])
        & (positions[:, 1] >= 0)
        & (positions[:, 1] < box[1])
        & (positions[:, 2] >= 0)
        & (positions[:, 2] < box[2])
    )
    return mask


def crop_atoms_to_cell(atoms: Atoms) -> Atoms:
    if not is_cell_orthogonal(atoms):
        raise NotImplementedError

    box = np.diag(atoms.cell)
    return atoms[mask_box(atoms.positions, box)]


def center_of_positions_to_center_of_box(
    positions: NDArray[np.float64], box: NDArray[np.float64]
) -> NDArray[np.float64]:
    positions = positions - positions.mean(0) + box / 2
    return positions


def make_lattice_disk(
    cell: Cell,
    box: tuple[float, float, float],
    rotation_axes: str = "zxz",
    rotations=(0.0, 0.0, 0.0),
):
    rotations = tuple(-np.deg2rad(a) for a in rotations)
    # rotation_axis = -np.deg2rad(rotation_axis)

    box = np.array(box)

    width = np.linalg.norm(box[:2]) + 2 * np.linalg.norm(cell)
    height = np.linalg.norm(box) + np.linalg.norm(cell)

    large_box = np.array((width, height, height))
    large_cube = cube_points(large_box)

    rotated_cell = rotate_positions(
        cell,
        center=(0, 0, 0),
        ai=rotations[0],
        aj=rotations[1],
        ak=rotations[2],
        axes=rotation_axes,
    )

    transformed_cube = transform_positions(rotated_cell, large_cube)
    reps = np.ceil(np.ptp(transformed_cube, 0)).astype(int)

    lattice = repeated_lattice(cell, reps)
    lattice = rotate_positions(
        lattice,
        center=(0, 0, 0),
        ai=-rotations[0],
        aj=-rotations[1],
        ak=-rotations[2],
        axes=rotation_axes,
    )

    lattice = center_of_positions_to_center_of_box(lattice, large_box)
    lattice = lattice[mask_box(lattice, large_box)]

    lattice = lattice - lattice.mean(0)
    lattice = lattice[np.linalg.norm(lattice[:, 1:], axis=-1) < height / 2]

    shift = cell.sum(0) / 2
    lattice = center_of_positions_to_center_of_box(lattice, box)
    lattice = rotate_positions(
        lattice,
        center=tuple(L / 2 for L in box),
        ai=rotations[0],
        aj=rotations[1],
        ak=rotations[2],
        axes=rotation_axes,
    )
    lattice = lattice - shift
    return lattice


def make_lattice_ball(cell, box):
    box = np.array(box)

    radius = min_circumscribed_radius(box)

    margin = np.linalg.norm(cell.sum(0)) / 2
    L = 2 * (radius + margin)

    cube = cube_points(L)

    transformed_cube = transform_positions(cell, cube)
    reps = np.ceil(np.ptp(transformed_cube, 0)).astype(int)

    lattice = repeated_lattice(cell, reps)

    lattice -= lattice.mean(0)
    d = np.linalg.norm(lattice, axis=-1)
    lattice = lattice[d < radius]

    shift = cell.sum(0) / 2
    lattice = center_of_positions_to_center_of_box(lattice, box) - shift
    return lattice


def make_perfect_supercell(atoms, box, shape, **kwargs):
    if shape == "ball":
        lattice = make_lattice_ball(atoms.cell, box)
    elif shape == "disk":
        lattice = make_lattice_disk(atoms.cell, box, **kwargs)
    else:
        raise ValueError(f"Shape {shape} not supported")

    positions = (lattice[:, None] + atoms.positions[None]).reshape(-1, 3)
    numbers = (np.ones_like(lattice[:, 0])[:, None] * atoms.numbers[None]).reshape(-1)
    # numbers = np.repeat(atoms.numbers, len(lattice))

    positions = rotate_positions(
        positions,
        center=tuple(L / 2 for L in box),
        ai=kwargs["rotations"][0],
        aj=kwargs["rotations"][1],
        ak=kwargs["rotations"][2],
        axes=kwargs["rotation_axes"],
    )

    atoms = ase.Atoms(numbers, positions, cell=box, pbc=True)

    return atoms


# def _rotate_and_crop_to_cell(atoms, rotation, rotation_axis: float = 0.0):
#     rotated_atoms = rotate_atoms(
#         atoms,
#         center="COU",
#         ai=-rotation_axis,
#         aj=rotation,
#         ak=rotation_axis,
#         axes="zxz",
#     )
#     cropped_atoms = crop_atoms_to_cell(rotated_atoms)
#     return cropped_atoms


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
    shape = config.shape
    # kwargs = {"rotation_axis": config.rotation_axis}
    kwargs = {"rotation_axes": config.rotation_axes, "rotations": config.rotations}
    atoms = make_perfect_supercell(atoms, box, shape, **kwargs)
    atoms = add_vacancies(atoms, config.vacancies, config.seed)
    return atoms


def run(config: Optional[Config] = None):
    if config is None:
        config = parse_arguments(Config)

    atoms = make_atoms(config)
    store_path = filename_from_config(config.store_path, config)
    save_atoms_to_zarr(store_path, atoms)


def rotate_cube_to_face_point(initial_point, target_direction):
    """
    Rotate a cube such that the initial_point faces the target_direction.

    Parameters:
    initial_point (np.ndarray): The initial position of the vertex to rotate.
    target_direction (np.ndarray): The target direction to face.

    Returns:
    np.ndarray: Euler angles (alpha, beta, gamma) in radians.
    """
    # Normalize the vectors
    initial_point = initial_point / np.linalg.norm(initial_point)
    target_direction = target_direction / np.linalg.norm(target_direction)

    # Calculate the rotation matrix
    v = np.cross(initial_point, target_direction)
    c = np.dot(initial_point, target_direction)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + k + k @ k * ((1 - c) / (s**2))

    # Convert the rotation matrix to Euler angles
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler("xyz", degrees=False)

    return euler_angles


# # Example usage
# initial_point = np.array([1, 1, 1])  # Initial vertex position
# target_direction = np.array([0, 0, 1])  # Target direction

# euler_angles = rotate_cube_to_face_point(initial_point, target_direction)
# print(f"Euler angles (radians): {euler_angles}")

# # Verify the rotation
# rotated_point = np.dot(R.from_euler("xy", euler_angles[:-1]).as_matrix(), initial_point)
# print(f"Rotated point: {rotated_point}")


if __name__ == "__main__":
    run()
