import abtem
import dask
import numpy as np
from abtem.core.axes import NonLinearAxis
from abtem.inelastic.phonons import AtomsEnsemble
from ase import Atoms


def rotate_atoms(atoms: Atoms, angle: float) -> Atoms:
    """Rotate atoms and cell around the x-axis by a given angle."""
    atoms = atoms.copy()
    atoms.rotate("x", angle / np.pi * 180.0, rotate_cell=True)
    return atoms


def cut_rotated_atoms(
    atoms: Atoms,
    angle: float,
    thickness: float,
    repetitions: tuple[int, int],
    margin: float = 0.0,
) -> Atoms:
    """
    Cut out a cuboid of atoms rotated around the x-axis by a given angle.

    Parameters
    ----------
    atoms : Atoms
        Unit cell of the crystal from which the cuboid is cut from.
    angle : float
        Rotation around the x-axis [deg].
    thickness : float
        Thickness along the z-axis [Å].
    repetitions : two int
        The number of repetitions along x and y.
    margin : float
        Margin added along y (the non-periodic direction) [Å].

    Returns
    -------
    atoms : Atoms
        Cuboid of atoms.
    """

    atoms = rotate_atoms(atoms, angle)

    box = (
        atoms.cell.lengths()[0] * repetitions[0],
        atoms.cell.lengths()[1] * repetitions[1],
        thickness,
    )

    atoms = abtem.atoms.cut_cell(
        atoms,
        cell=box,
    )

    atoms = atoms[atoms.positions[:, 2] < atoms.cell[2, 2] - margin]
    atoms = atoms[atoms.positions[:, 2] > margin]
    atoms.center(axis=2)

    return atoms


def rotated_atoms_ensemble(
    atoms: Atoms,
    angles: np.ndarray,
    thickness: float,
    repetitions: tuple[int, int],
    margin: float,
):
    """
    Create multiple cuboids of atoms at different rotations as an ensemble.

    atoms : Atoms
        Unit cell of the crystal from which the cuboids are cut from.
    angles : list of float
        Rotations around the x-axis [deg].
    thickness : float
        Thickness along the z-axis [Å].
    repetitions : two int
        The number of repetitions along x and y.
    margin : float
        Margin added along y (the non-periodic direction) [Å].

    Returns
    -------
    atoms_ensemble : AtomsEnsemble
        Ensemble of rotated atoms.
    """

    func = dask.delayed(cut_rotated_atoms)

    trajectory = [
        func(atoms, angle, thickness, repetitions, margin) for angle in angles
    ]

    axis_metadata = NonLinearAxis(label="x_rotation", units="deg", values=angles)

    ensemble = AtomsEnsemble(
        trajectory, ensemble_mean=False, ensemble_axes_metadata=axis_metadata
    )
    return ensemble
