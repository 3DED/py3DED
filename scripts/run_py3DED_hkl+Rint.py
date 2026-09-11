# -*- coding: utf-8 -*-
"""
Unified program to process diffraction data from zarr files and calculate:
1. Integrated intensities (calculated once)
2. HKL files for specified thicknesses
3. R_int values across all thicknesses
4. Output file with (thickness, R_int) pairs
"""

import numpy as np
import abtem
from abtem.bloch.utils import excitation_errors as calc_Sg
from scipy import integrate
from pathlib import Path
from ase.io import cif
from sys import argv as call_arguments
import json

def calculate_integrated_intensities(file, g_max=2.0):
    """
    Calculate integrated intensities from zarr diffraction data.
    
    Parameters
    ----------
    file : str or Path
        Path to the zarr file containing diffraction data

    g_max : float, optional
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'integrated_intensities': shape (N_thicknesses, N_reflections)
        - 'excitation_errors': shape (N_orientations, N_reflections)
        - 'miller_indices': shape (N_reflections, 3)
        - 'thicknesses': array of thickness values
    """
    file = Path(file)
    
    # Load zarr file
    dsz = abtem.from_zarr(str(file))
    
    # Calculate length of g vectors for all reflections
    # take first orientation as representative
    g = np.linalg.norm( dsz.positions[0,0], axis=-1)
    
    # resolution filter mask
    hkl_mask = (g < g_max)

    print("Resolution filter g_max: Using", np.count_nonzero(hkl_mask), "out of", len(g), "reflections")
    
    # Load and compute intensities
    intensities = dsz.intensities[:,:,hkl_mask].compute()

    # No rescale needed here: WindowedPixelatedDetector (py3DED/detector.py) is
    # now abTEM's own class, which auto-renormalizes MS intensities by the
    # actual window's power gain at write time -- applying (8/3)**2 on top
    # unconditionally (as this used to, regardless of which window_func was
    # actually used) would double-count that correction.

    # Get electron kinetic energy from metadata
    energy = dsz.metadata['energy']
    print("Energy in metadata:", energy, "eV")
    
    # Calculate excitation errors for all orientations and reflections
    excitation_errors = calc_Sg(
        dsz.positions[:,:,hkl_mask],
        energy=energy,
        use_wave_eq=True
    )
    
    # Get thickness values
    thicknesses = np.array(dsz.ensemble_axes_metadata[1].values)
    
    # Integrate rocking curves using Simpson integration
    integrated_intensities_data = np.abs(
        integrate.simpson(intensities, excitation_errors, axis=0)
    )
    
    return {
        'integrated_intensities': integrated_intensities_data,
        'excitation_errors': excitation_errors,
        'miller_indices': dsz.miller_indices[hkl_mask],
        'thicknesses': thicknesses,
        #'dsz': dsz,
    }


def write_hkl_files(zarr_file, integrated_data, hkl_thicknesses=None):
    """
    Write hkl files for specified thicknesses.
    
    Parameters
    ----------
    zarr_file : str or Path
        Path to the zarr file (for naming output files)
    integrated_data : dict
        Output from calculate_integrated_intensities()
    hkl_thicknesses : list of float, optional
        List of thickness values for which to write hkl files.
        Default is [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    """
    if hkl_thicknesses is None:
        hkl_thicknesses = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    zarr_file = Path(zarr_file)
    base_name = zarr_file.stem
    
    integrated_intensities_data = integrated_data['integrated_intensities']
    excitation_errors = integrated_data['excitation_errors']
    miller_indices = integrated_data['miller_indices']
    thicknesses = integrated_data['thicknesses']
    
    for target_thickness in hkl_thicknesses:
        # Find the closest thickness in the dataset
        thickness_idx = np.argmin(np.abs(thicknesses - target_thickness))
        closest_thickness = thicknesses[thickness_idx]
        
        # Identify reflections that pass the Ewald sphere criterion
        # Use thickness-dependent rocking curve width criterion
        rc_width_min = 2 * 2 / closest_thickness

        # This also filters out the 000 reflection
        passing_reflections = (
            (np.min(excitation_errors, axis=0) < -rc_width_min / 2) &
            (np.max(excitation_errors, axis=0) > rc_width_min / 2)
        )[0]
        
        # Get miller indices and intensities of passing reflections
        passing_hkl = miller_indices[passing_reflections]
        I_thickness = integrated_intensities_data[thickness_idx, passing_reflections]
        
        # Create output filename and write hkl file
        output_file = zarr_file.parent / f"{base_name}_{closest_thickness:.0f}.hkl"
        _write_hkl_file(output_file, passing_hkl, I_thickness)
        print(f"Written {output_file.name} with {np.count_nonzero(passing_reflections)} reflections.")


def calculate_rint(integrated_data, cif_file, add_inversion=True):
    """
    Calculate R_int values across all thicknesses using symmetry equivalence.
    
    The set of passing reflections is determined separately for each thickness
    based on the thickness-dependent rocking curve criterion.
    The first minima of the oscillating rocking curve are at excitation errors
    of -1/t and +1/t. The distance between the minima is 2 * 1/t
    The distance between 2nd order minima is 4 * 1/t.
    Here, the weaker filtering criterion is used.
    rc_width_min = 2 * 1 / thickness
    
    Parameters
    ----------
    integrated_data : dict
        Output from calculate_integrated_intensities()
    cif_file : str or Path
        Path to CIF file containing crystal structure and symmetry information
    add_inversion: True | False
        Add inversion center if point group is non-centrosymmetric
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'rint_values': array of R_int values for each thickness
        - 'thicknesses': array of thickness values
        - 'N_hkl': number of reflections used for R_int calculation
    """

    cif_file = Path(cif_file)
    
    integrated_intensities_data = integrated_data['integrated_intensities']
    excitation_errors = integrated_data['excitation_errors']
    miller_indices = integrated_data['miller_indices']
    thicknesses = integrated_data['thicknesses']
    
    # Calculate min/max excitation errors across orientations
    sg_min = np.min(excitation_errors, axis=0)  # shape: (N_reflections,)
    sg_max = np.max(excitation_errors, axis=0)  # shape: (N_reflections,)
    
    # Calculate thickness-dependent rocking curve criterion
    # The first minima of the oscillating rocking curve are at excitation errors of -1/t and +1/t
    # The distance between the minima is 2 * 1/t
    # The distance between 2nd order minima is 4 * 1/t
    rc_width_min = 2 * 1 / thicknesses  # shape: (N_thicknesses,)
    
    # Create mask with shape (N_thicknesses, N_reflections)
    # Each row t contains True where reflections pass for thickness t
    mask = ((sg_min[np.newaxis, :] < -rc_width_min[:, np.newaxis] / 2) &
            (sg_max[np.newaxis, :] > +rc_width_min[:, np.newaxis] / 2))
    
    mask = mask[0] # shape (N_t, N_hkl)
    
    # Use union of all passing reflections for symmetry analysis
    # This ensures we have a consistent set of reflection indices across all thicknesses
    passing_reflections_union = np.any(mask, axis=0)
    
    # Get miller indices for all reflections that pass at least once
    passing_hkl = miller_indices[passing_reflections_union]
    
    print(f"\n{np.count_nonzero(passing_reflections_union)} reflections pass at least once across thicknesses")
    
    # Load crystal symmetry from CIF file (read_cif does not work with Path objects
    asecif = cif.read_cif(str(cif_file))
    SG = asecif.info['spacegroup']
    rotations, translations = SG.get_op()
    
    inversion = np.eye(3)*(-1)
    
    # add center of inversion so that Friedel pairs are merged
    # Effectively, we limit the point group to a Laue class
    if add_inversion and not np.any( np.all(rotations == inversion, axis=(-2,-1)) ):
        print("Using Laue class to decide which reflections are related by symmetry.")
        # There is no center of inversion and it should be added
        # Add center of inversion, add all symmetry elements multiplied by inversion
        rotations = np.vstack(
            (rotations,
            np.matmul( rotations, inversion ).astype(int))
        )

    rotations_reciprocal = np.matrix_transpose(
        np.linalg.inv(rotations)
    ).astype(int)

    # make sure that list of symmetry operations is unique
    # and keep order
    _, unique_ids = np.unique(rotations_reciprocal, axis=0, return_index=True)
    rotations_reciprocal = rotations_reciprocal[ unique_ids.sort() ][0]
    
    for m in rotations_reciprocal:
        print(m)
        
    # Apply symmetry operations to miller indices to find equivalent reflections
    hkl_sym = (rotations_reciprocal @ passing_hkl.T).transpose(2, 0, 1)
    print(f"Symmetry analysis: {hkl_sym.shape[0]} reflections, {hkl_sym.shape[1]} symmetry operations")
    
    # Group reflections by symmetry equivalence
    A = np.ascontiguousarray(hkl_sym[:, 0])
    A_view = A.view([('x', A.dtype), ('y', A.dtype), ('z', A.dtype)])
    
    # B is a concatenated array of all reflections and their sym.equiv. partners
    B = np.ascontiguousarray(hkl_sym.reshape(-1, 3))
    B_view = B.view([('x', B.dtype), ('y', B.dtype), ('z', B.dtype)])
    
    # intersect1d returns (common_elements, indicesA, indicesB)
    _, idx_A, idx_B = np.intersect1d(A_view, B_view, return_indices=True)
    
    K = hkl_sym.shape[1] # number of symmetry elements
    origin = idx_B // K  # assign IDs < len(A), i.e., map reflections to their origin
    
    group_ids = np.full(len(A), fill_value=0)
    
    # unique group_ids > max_index
    group_ids = np.arange( len(A) ) + len(A)
    
    # indices i defined in idx_A
    # for each i, take minimum of (group_ids[ix_A][i] vs origin[i])
    np.minimum.at(group_ids, idx_A, origin)
    group_ids = group_ids.astype(int)
    
    # Preselect groups, which have at least two elements.
    # Determine unique groups with MORE THAN 1 element.
    # This will ignore 'lone' reflections without any other symmetrically equivalent reflection in the data set.
    groups_with_singletons, counts = np.unique( group_ids, return_counts=True)
    groups = groups_with_singletons[ counts > 1 ]
    
    print(f"There are {len(groups)} groups which contain at least 2 reflections that are related by symmetry.")
    
    # Calculate R_int for each thickness
    # Following Rmerge definition in https://journals.iucr.org/d/issues/2013/07/00/ba5190/
    # For each thickness, use only the reflections that pass the thickness-dependent criterion
    R_int = np.zeros(len(thicknesses))
    N_hkl = np.zeros(len(thicknesses), dtype=int)
    
    for t_idx in range(len(thicknesses)):
        # Determine which reflections pass at this thickness
        passing_at_t = mask[t_idx, passing_reflections_union]  # shape: (N_passing,)
        
        # Get integrated intensities for this thickness and the reflections that pass
        I_t = integrated_intensities_data[t_idx, passing_reflections_union][passing_at_t]
        group_ids_t = group_ids[passing_at_t]
        
        # Calculate R_int using only reflections that pass at this thickness
        numerator = 0.0
        denominator = 0.0
        N = 0
        
        for g in np.unique(group_ids_t):
            selector = (group_ids_t == g)
            
            if np.count_nonzero(selector) > 1: # ignore singletons
                I_group_mean = I_t[selector].mean()
                I_group_sum = I_t[selector].sum()
                I_group_deltas = np.sum(
                    np.abs(I_group_mean - I_t[selector])
                )
                
                numerator += I_group_deltas
                denominator += I_group_sum
                N += 1
        
        if denominator > 0:
            R_int[t_idx] = numerator / denominator
                
        # Print progress
        N_hkl[t_idx] = N #np.count_nonzero(passing_at_t)
    
    return {
        'rint_values': R_int,
        'thicknesses': thicknesses,
        'N_hkl': N_hkl
    }


def write_rint_file(output_file, thicknesses, rint_values, N_hkl):
    """
    Write R_int values to a file with two columns: thickness and R_int.
    
    Parameters
    ----------
    output_file : str or Path
        Output file path
    thicknesses : ndarray
        Array of thickness values (in Angstrom)
    rint_values : ndarray
        Array of R_int values (in fraction, will be converted to percentage)
    """
    output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Thickness (Å)    R_int (%)     N_hkl_groups\n")
        
        # Write data
        for thickness, rint, N in zip(thicknesses, rint_values, N_hkl):
            f.write(f"{thickness:15.2f} {rint*100:12.1f} {N:9}\n")
    
    print(f"Written {output_file.name}")
    print("In the 3rd column, the number of reflection groups used to calculate R_int is given.")
    print("This number depends on the thickness because reflections with incomple rocking curve profiles are filtered out.")


def _write_hkl_file(filepath, miller_indices, intensities):
    """
    Write hkl file in the standard format.
    
    Parameters
    ----------
    filepath : Path or str
        Output file path
    miller_indices : ndarray
        Array of shape (N_reflections, 3) containing h, k, l values
    intensities : ndarray
        Array of shape (N_reflections,) containing integrated intensities
    """
    filepath = Path(filepath)
    intensitiesE8 = intensities * 1e8
    
    with open(filepath, 'w') as f:
        for (h, k, l), intensity in zip(miller_indices, intensitiesE8):
            f.write(f"{int(h):4d}{int(k):4d}{int(l):4d}{intensity:12.4f}{intensity**0.5:12.4f}\n")


def main(zarr_file, cif_file, hkl_thicknesses=None):
    """
    Main function to process diffraction data and generate all outputs.
    
    Parameters
    ----------
    zarr_file : str or Path
        Path to the zarr file
    cif_file : str or Path
        Path to the CIF file
    hkl_thicknesses : list of float, optional
        Thicknesses for which to write hkl files
    """
    zarr_file = Path(zarr_file)
    cif_file = Path(cif_file)
    
    print("=" * 70)
    print("CALCULATING INTEGRATED INTENSITIES")
    print("=" * 70)
    integrated_data = calculate_integrated_intensities(zarr_file)
    print(f"Calculated integrated intensities for {len(integrated_data['thicknesses'])} thicknesses")
    print(f"Thickness range: {integrated_data['thicknesses'][0]:.1f} - {integrated_data['thicknesses'][-1]:.1f} Å")
    
    print("\n" + "=" * 70)
    print("WRITING HKL FILES")
    print("=" * 70)
    write_hkl_files(zarr_file, integrated_data, hkl_thicknesses)
    
    print("\n" + "=" * 70)
    print("CALCULATING R_INT")
    print("=" * 70)
    rint_data = calculate_rint(integrated_data, cif_file)
    
    print("\n" + "=" * 70)
    print("WRITING R_INT OUTPUT FILE")
    print("=" * 70)
    output_rint_file = zarr_file.parent / f"{zarr_file.stem}_rint.txt"
    write_rint_file(output_rint_file, rint_data['thicknesses'], rint_data['rint_values'], rint_data['N_hkl'])
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':    
    file_zarr = None
    file_cif = None

    # ZARR file
    if len(call_arguments) == 2 and call_arguments[1].endswith(".zarr"):
        file_zarr = Path(call_arguments[1])
        if not file_zarr.exists():
            file_zarr = None

    if not file_zarr:
        print("""Usage: python run_py3DED_merged.py <zarr_file>
        'settings.json' and a CIF file expected in the folder
        """)
        input()
        raise ValueError("No *.zarr file provided.")

    print("Intensity data file:", file_zarr)
        
    # CIF file
    file_settings = file_zarr.parent / "settings.json"

    try:
        content = file_settings.read_text()
        settings_sim = json.loads(content)
        file_cif = file_zarr.parent / Path(settings_sim['cif_file']).name
    except:
        file_cif = list( file_zarr.parent.glob("*.cif") )
        if file_cif:
            file_cif = file_cif[0]

    if not file_cif:
        print("CIF file cannot be found. Rint will not be calculated.")
    else:
        print("Rint will be based on symmetry found in", file_cif)
    
    main(file_zarr, file_cif)
