"""
This script can run simulations of 3D electron diffraction, utilizing both multislice and Bloch wave methods.

Users can specify parameters such as run mode, number of workers and unit cell, by modifying the ConfigSingleAxis
object or specifying them using command line arguments.

Usage
-----
This script can be run from the command line with the following format:::

    python simulate.py --run_mode "ms" --num_workers 8 --vacancies 0.01

"""

import dataclasses

import abtem.parametrizations
import abtem.potentials.iam
import abtem.waves
import ase
import numpy as np
from abtem.bloch import BlochWaves, StructureFactor
from abtem.core.energy import energy2wavelength
from abtem.rotation_series import (
    rotated_atoms_ensemble,
    rotation_series_orientation_matrices,
)

from py3DED.config import BaseConfig, parse_arguments
from py3DED.detector import WindowedPixelatedDetector
from py3DED.io import read_atoms_from_zarr


@dataclasses.dataclass
class Config(BaseConfig):
    """
    Configuration object for Bloch wave and multislice simulations.

    Parameters
    ----------
    run_mode : str, optional
        The simulation mode to be used. Either "ms" for multislice or "bw" for Bloch wave.
    precision : str, optional
        The precision to be used for computations. Either "float32" or "float64".
    fft : str, optional
        The FFT backend to be used. Either "fftw" or "numpy".
    task_progress : bool, optional
        Display the progress of each task in addition to the progress of the whole computation.
    device : str, optional
        The device to be used for computations. Either "cpu" or "gpu".
    enable_mps : bool, optional
        Enable Metal Performance Shaders (MPS) for GPU computations on Apple Silicon.
    num_workers : int, optional
        The number of workers to be used for parallel computations.
    store_path : str, optional
        The path to store the output data. The path can contain placeholders for configuration values
        that will be replaced with the actual values. Placeholders should be enclosed in curly brackets.
    scheduler : str, optional
        The scheduler to be used for parallel computations. Either "threads" or "processes".
    energy : float, optional
        The energy of the electrons in the simulation in eV.
    unit_cell : str or Atoms
        The unit cell to be used in the simulation. Can be a path to a CIF file or an ASE Atoms object.
    supercell : str, optional
        The supercell to be used in the simulation. Can be a path to a Zarr file or an ASE Atoms object.
        The supercell may be generated using the atoms module.
    thermal_sigma : dict or str, optional
        The thermal sigma values to be used in the simulation. Can be a dictionary or a string of comma-separated
        key-value pairs.
    rotation_axis : float, optional
        The rotation axis to be used in the simulation in degrees.
    rotation_min : float, optional
        The minimum rotation angle to be used in the simulation in degrees.
    rotation_max : float, optional
        The maximum rotation angle to be used in the simulation in degrees.
    rotation_steps : int, optional
        The number of rotation steps to be used in the simulation.
    slice_thickness : float, optional
        The slice thickness to be used in the multislice simulation in Angstroms.
    exit_planes : int, optional
        Save the simulation results every this many exit planes. Bloch wave results are saved at the corresponding
        exit plane depths.
    sampling : float, optional
        The real space sampling to be used in the multislice simulation in Angstroms.
    projection : str, optional
        The potential projection integrals to be used in the multislice simulation. Either "infinite" or "finite".
    g_max_store : float, optional
        The maximum g value to be stored in both multislice and Bloch wave simulations.
    integration_radius : float, optional
        The integration radius to be used in to calculate the intensity of the diffraction spots in multislice
        simulations in reciprocal Angstroms.
    window_func : str, optional
        The window function for the method. Should be one of the window functions available in
        `scipy.signal.windows`. Default is 'hann'.
    margin : float, optional
        The cropping margin to be applied to the (real-space) wave functions in the WindowedPixelatedDetector
        in the multislice simulation in Angstroms.
    sg_max : float, optional
        The maximum excitation error to be included in the Bloch wave simulations in reciprocal Angstroms.
    g_max : float, optional
        The maximum g value to be used in the Bloch wave simulation in reciprocal Angstroms.
    centering : {'P', 'I', 'A', 'B', 'C', 'F'}, optional
        The crystal centering to be used in the Bloch wave simulations. Either "P", "A", "B", "C", "I" or "F".
    use_wave_eq : bool, optional
        Use the version of the Bloch wave simulation derived from the wave equation.
    occupancy : float, optional
        The occupancy as a fraction used in the Bloch wave simulations. This does NOT affect the multislice simulations.
    """

    run_mode: str = "ms"

    # Computations
    # ------------
    precision: str = "float32"
    fft: str = "fftw"
    task_progress: bool = True
    device: str = "cpu"
    enable_mps: bool = False
    num_workers: int = 8
    store_path: str = ""
    scheduler: str = "threads"

    # Experiment
    # ----------
    energy: float = 200e3
    unit_cell: str = ""
    supercell: str = ""
    thermal_sigma: dict | str = "Si,0.078"
    # Rotation
    rotation_axis: float = 0.0
    rotation_min: float = 0.0
    rotation_max: float = 45.0
    rotation_steps: int = 5

    # Multislice
    # ----------
    slice_thickness: float = 1.0
    exit_planes: int = 4
    sampling: float = 0.04
    projection: str = "infinite"
    g_max_store: float = 4.0
    integration_radius: float = 0.01
    margin: float = 0.0
    window_func: str = "hann"

    # Bloch wave
    # ----------
    sg_max: float = 0.3
    g_max: float = 16.0
    centering: str = "F"
    use_wave_eq: bool = True
    occupancy: float = 1.0


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

    if isinstance(config_dict["unit_cell"], str):
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
    elif not isinstance(thermal_sigma, dict):
        raise ValueError(
            f"Thermal sigma must be a dictionary or a string, not {type(thermal_sigma)}"
        )

    return thermal_sigma


def make_potential(config):
    atoms_ensemble, atoms = get_atoms_ensemble(config)

    thermal_sigma = parse_thermal_sigma(config.thermal_sigma)

    if isinstance(config.thermal_sigma, float):
        parametrization = "lobato"
    else:
        parametrization = abtem.parametrizations.LobatoParametrization(
            sigmas={key: value for key, value in thermal_sigma.items()}
            #sigmas={key: value * np.sqrt(3) for key, value in thermal_sigma.items()}
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
    
    # In abTEM, the structure factor is calculated as the complex conjugate
    # of the structure factor that is used in crystallography.
    # This inconsistency will be resolved in future versions.
    # This is a temporary workaround:    
       
    atoms.set_positions( -atoms.positions )
    
    structure_factor = StructureFactor(
        atoms,
        g_max=config.g_max,
        parametrization="lobato",
        centering=config.centering,
        thermal_sigma=thermal_sigma,
        occupancy=config.occupancy,
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


def get_atoms_ensemble(config: Config):
    supercell = read_atoms_from_zarr(config.supercell, lazy=True).compute()
    angles_deg = np.rad2deg(get_x_angles(config))
    atoms_ensemble = rotated_atoms_ensemble(
        supercell, angles_deg, rotation_axis=config.rotation_axis
    )
    if isinstance(config.unit_cell, str):
        atoms = ase.io.read(config.unit_cell)
    else:
        atoms = config.unit_cell
    return atoms_ensemble, atoms


def setup_multislice(config: Config, return_exit_wave=False):
    atoms_ensemble, atoms = get_atoms_ensemble(config)

    pw = abtem.waves.PlaneWave(energy=config.energy)

    potential = make_potential(config)

    max_angle = config.g_max_store * energy2wavelength(config.energy) * 1e3

    detector = WindowedPixelatedDetector(
        max_angle=max_angle * 1.2,
        to_cpu=True,
        reciprocal_space=True,
        margin=config.margin,
        window_func=config.window_func,
    )

    diffraction = pw.multislice(potential=potential, detectors=detector)

    angles_deg = np.rad2deg(get_x_angles(config))
    orientation_matrices = rotation_series_orientation_matrices(
        angles_deg, rotation_axis=config.rotation_axis
    )[:, None]

    diffraction_indexed = diffraction.to_cpu().index_diffraction_spots(
        cell=atoms,
        orientation_matrices=orientation_matrices,
        centering=config.centering,
        sg_max=config.sg_max,
        g_max=config.g_max / 2,
        radius=config.integration_radius,
    )

    cropped = diffraction_indexed.crop(k_max=config.g_max_store)

    return cropped[:, 1:]


def get_bloch_waves(config: Config):
    atoms_ensemble, atoms = get_atoms_ensemble(config)

    structure_factor = make_structure_factor(atoms, config)

    bloch_waves = BlochWaves(
        structure_factor=structure_factor,
        energy=config.energy,
        sg_max=config.sg_max,
        use_wave_eq=config.use_wave_eq,
    )
    return bloch_waves


def setup_bloch_wave(
    config: Config,
):
    angles = get_x_angles(config)

    bloch_waves = get_bloch_waves(config)

    rotation_axis = np.deg2rad(config.rotation_axis)

    rotation_axis = rotation_axis.reshape(
        1,
    )

    rotation_axis = rotation_axis.repeat(angles.size, axis=0)

    # print(angles)

    all_angles = np.stack((rotation_axis, angles, -rotation_axis), axis=-1)

    rotated = bloch_waves.rotate("zxz", all_angles, degrees=False)
    # rotated = bloch_waves.rotate(
    #    "z", rotation_axis, "x", angles, "z", -rotation_axis, degrees=False
    # )
    potential = make_potential(config)

    diffraction_bw = (
        rotated.calculate_diffraction_patterns(
            thicknesses=potential.exit_thicknesses,  # tol=1e-12
        )
        .to_cpu()
        .crop(k_max=config.g_max_store)
    )

    # Stored in degrees, matching the "deg" units on this same axis in
    # get_atoms_ensemble()'s multislice-side AtomsEnsemble (abtem.rotation_series
    # .rotated_atoms_ensemble) -- both used to store radians under a "deg" label
    # (harmless on its own, since nothing read the values back), but now that
    # the multislice side stores actual degrees, leaving this one in radians
    # would make the two sides' x_rotation coordinates fail to align.
    diffraction_bw.axes_metadata[0].label = "x_rotation"
    diffraction_bw.axes_metadata[0].values = tuple(np.rad2deg(angles))

    return diffraction_bw[:, 1:]


def run(config=None, save_to_disk=True):
    if config is None:
        config = parse_arguments(Config)

    set_abtem_config(config)

    if config.run_mode == "ms":
        output = setup_multislice(config)
    elif config.run_mode == "bw":
        output = setup_bloch_wave(config)
    else:
        raise ValueError(f"Run mode {config.run_mode} not supported")

    if save_to_disk:
        output.to_zarr(
            get_store_path(config),
            num_workers=config.num_workers,
        )

    return output


if __name__ == "__main__":
    run(config=None, save_to_disk=True)
