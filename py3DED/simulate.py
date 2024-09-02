"""
This script can run simulations of 3D electron diffraction, utilizing both multislice and Bloch wave methods.

Users can specify parameters such as run mode, number of workers and unit cell, by modifying the ConfigSingleAxis
object or specifying them using command line arguments.

Usage
-----
This script can be run from the command line with the following format:::

    python simulate_single_axis.py --run_mode "ms" --num_workers 8 --vacancies 0.01

"""

import dataclasses
import dask

import abtem.parametrizations
import abtem.potentials.iam
import abtem.waves
import ase
import numpy as np
from abtem import AtomsEnsemble
from abtem.atoms import euler_to_rotation
from abtem.bloch import BlochWaves, StructureFactor
from abtem.core.energy import energy2wavelength
from ase import Atoms

from py3ded.config import BaseConfig, parse_arguments
from py3ded.detector import WindowedPixelatedDetector
from py3ded.io import read_atoms_from_zarr
from py3ded.supercell import make_rotated_atoms_ensemble


@dataclasses.dataclass
class Config(BaseConfig):
    run_mode: str = "ms"

    # Computations
    # ------------
    precision: str = "float32"
    fft: str = "fftw"
    task_progress: bool = True
    device: str = "cpu"
    enable_mps: bool = False
    num_workers: int = 8
    store_path: str = "output/{supercell}_{run_mode}.zarr"
    scheduler: str = "threads"

    # Experiment
    # ----------
    energy: float = 200e3
    unit_cell: str = "structures/si.cif"
    supercell: str = "structures/si_disk_rot5.0.zarr"
    thermal_sigma: str = "Si,0.078"
    # Rotation
    rotation_axis: float = 0.0  # TODO: not implemented
    rotation_min: float = 0.0
    rotation_max: float = 45.0
    rotation_steps: int = 451
    # Vacancies
    seed: int = 1337
    num_runs: int = 1  # TODO: not implemented
    vacancies: float = 0.0

    # Multislice
    # ----------
    slice_thickness: float = 1.0
    exit_planes: int = 4
    sampling: float = 0.04
    projection: str = "infinite"
    g_max_store: float = 4.0
    integration_radius: float = 0.01
    margin: float = 10.

    # Bloch wave
    # ----------
    sg_max: float = 0.3
    g_max: float = 16.0
    centering: str = "F"
    use_wave_eq: bool = True


def get_store_path(config: Config):
    """
    Get the formatted store path based on the values present in the configuration object.

    Parameters
    ----------
    config : ConfigSingleAxis, required
        The configuration object used to determine the store path.

    Returns
    -------
    str
        The formatted store path based on the values in the configuration object.
    """
    config_dict = {k: v for k, v in config.__dict__.items()}

    config_dict["unit_cell"] = config.unit_cell.split("/")[-1].split(".")[0]

    return config.store_path.format(**config_dict)


def parse_thermal_sigma(thermal_sigma: str | dict) -> dict:
    """
    Parse thermal sigma values represented as either a dictionary or a string.

    This function takes an input of either a dictionary or a string. If the input is
    a string, it assumes that the string contains comma-separated keys and values.
    The keys and values are parsed into a dictionary where the keys are the original
    keys and the values are converted to float.

    Parameters
    ----------
    thermal_sigma : str | dict
        The thermal sigma to be parsed. It can be either a dictionary, or a string of
        comma-separated keys and values.

    Returns
    -------
    dict
        The parsed thermal sigma as a dictionary.
    """
    if isinstance(thermal_sigma, str):
        keys = thermal_sigma.split(",")[::2]
        values = thermal_sigma.split(",")[1::2]
        thermal_sigma = {key: float(value) for key, value in zip(keys, values)}

    return thermal_sigma


def make_potential(atoms_ensemble, config):
    thermal_sigma = parse_thermal_sigma(config.thermal_sigma)

    parametrization = abtem.parametrizations.LobatoParametrization(
        sigmas={key: value * np.sqrt(3) for key, value in thermal_sigma.items()}
    )

    potential = abtem.potentials.iam.Potential(
        atoms_ensemble,
        sampling=config.sampling,
        exit_planes=config.exit_planes,
        slice_thickness=config.slice_thickness,
        parametrization=parametrization,
        projection=config.projection,
    )
    return potential


def get_x_angles(config: Config):
    angles = np.linspace(
        config.rotation_min, config.rotation_max, config.rotation_steps
    )
    angles = angles / 180 * np.pi
    return angles


def make_structure_factor(atoms, config: Config):
    thermal_sigma = parse_thermal_sigma(config.thermal_sigma)

    structure_factor = StructureFactor(
        atoms,
        g_max=config.g_max,
        parametrization="lobato",
        centering=config.centering,
        thermal_sigma=thermal_sigma,
    )

    return structure_factor


def set_abtem_config(config: Config):
    abtem.config.set(
        {
            "precision": config.precision,
            "device": config.device,
            "fft": config.fft,
            "diagnostics.task_progress": config.task_progress,
            "diagnostics.progress_bar": "tqdm",
            "enable_mps": config.enable_mps,
        }
    )


def add_vacancies(atoms, vacancies: float, seed: int):
    if hasattr(atoms, "compute"):
        atoms = dask.delayed(add_vacancies)(atoms, vacancies, seed)

    np.random.seed(seed)
    mask = np.random.rand(len(atoms)) > vacancies
    atoms = atoms[mask]
    return atoms


def get_atoms_ensemble(config: Config):
    supercell = read_atoms_from_zarr(config.supercell, lazy=True)


    angles = get_x_angles(config)
    atoms_ensemble = make_rotated_atoms_ensemble(supercell, angles)
    atoms = ase.io.read(config.unit_cell)
    return atoms_ensemble, atoms


def setup_multislice(
    config: Config,
):
    angles = get_x_angles(config)
    
    atoms_ensemble, atoms = get_atoms_ensemble(config)

    pw = abtem.waves.PlaneWave(energy=config.energy)

    potential = make_potential(atoms_ensemble, config)

    max_angle = config.g_max_store * energy2wavelength(config.energy) * 1e3

    detector = WindowedPixelatedDetector(
        max_angle=max_angle * 1.2, to_cpu=True, reciprocal_space=True, margin=config.margin
    )

    diffraction = pw.multislice(potential=potential, detectors=detector)
    
    orientation_matrices = np.array(
        [euler_to_rotation(angle, 0, 0, axes="xzx") for angle in angles]
    )[:, None]

    diffraction_indexed = (
        diffraction.to_cpu()
        .index_diffraction_spots(
            cell=atoms,
            orientation_matrices=orientation_matrices,
            centering=config.centering,
            sg_max=config.sg_max,
            g_max=config.g_max / 2,
            radius=config.integration_radius,
        )
        .crop(k_max=config.g_max_store)
    )

    return diffraction_indexed


def setup_bloch_wave(
    config: Config,
):
    angles = get_x_angles(config)

    atoms_ensemble, atoms = get_atoms_ensemble(config)

    structure_factor = make_structure_factor(atoms, config)

    bloch_waves = BlochWaves(
        structure_factor=structure_factor,
        energy=config.energy,
        sg_max=config.sg_max,
        use_wave_eq=config.use_wave_eq,
    )

    rotated = bloch_waves.rotate("x", angles)

    potential = make_potential(atoms_ensemble, config)

    diffraction_bw = (
        rotated.calculate_diffraction_patterns(
            thicknesses=potential.exit_thicknesses,  # tol=1e-12
        )
        .to_cpu()
        .crop(k_max=config.g_max_store)
    )

    return diffraction_bw


def run(config, save_to_disk=True):
    set_abtem_config(config)

    if config.run_mode == "ms":
        output = setup_multislice(config)
    else:
        output = setup_bloch_wave(config)

    if save_to_disk:
        output.to_zarr(
            get_store_path(config),
            num_workers=config.num_workers,
        )

    return output


def main():
    config = parse_arguments()

    angles = get_x_angles(config)

    abtem.config.set(
        {
            "precision": config.precision,
            "device": config.device,
            "fft": config.fft,
            "diagnostics.task_progress": config.task_progress,
            "diagnostics.progress_bar": "tqdm",
            "enable_mps": config.enable_mps,
        }
    )

    atoms_ensemble, atoms = make_atoms(config)

    if config.run_mode == "ms":
        run_multislice(angles, atoms_ensemble, atoms, config)
    elif config.run_mode == "bw":
        run_bloch_wave(angles, atoms_ensemble, atoms, config)
    else:
        raise ValueError("Invalid run mode")


if __name__ == "__main__":
    main()
