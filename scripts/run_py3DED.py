'''
missing:
 - automatic centering
 - support R centering, Robv/Rrev: abtem.bloch.utils.py / get_reflection_condition
 - option: supply exising supercell instead of cif file
 - better separate MS and BW setup: do not create supercell for BW
 - name: "cif", "json"
 
# MS-only parameters:
box_size_x
box_size_y
sampling
projection

# BW-only parameters:
g_max


For BW, only the product of "slice_thickness" x "exit_planes" matters.

Thermal sigmas: σ = sqrt(U) = sqrt(B / (8 pi ** 2))


'''
import abtem
from ase.io import read as ase_read
from py3DED import atoms, simulate
from py3DED.io import open_zarr_group, read_atoms_from_zarr

import zarr
import numpy as np
import scipy

from sys import argv as call_arguments, version as python_version

from pathlib import Path
from datetime import datetime
from time import time
from json import dump, load

def sep():
    print('- '*42)
    
print('''
                         ad888888b,  88888888ba,    88888888888  88888888ba,    
                        d8"     "88  88      `"8b   88           88      `"8b   
                                a8P  88        `8b  88           88        `8b  
8b,dPPYba,   8b       d8     aad8"   88         88  88aaaaa      88         88  
88P'    "8a  `8b     d8'     ""Y8,   88         88  88"""""      88         88  
88       d8   `8b   d8'         "8b  88         8P  88           88         8P  
88b,   ,a8"    `8b,d8'  Y8,     a88  88      .a8P   88           88      .a8P   
88`YbbdP"'       Y88'    "Y888888P'  88888888Y"'    88888888888  88888888Y"'    
88               d8'                                                            
88              d8'

by Jacob Madsen, Małgorzata Katarzyna Cabaj, Paul Klar 
''') # https://patorjk.com/software/taag/#p=display&f=Univers&t=py3DED

#####
##### Input file as argument
#####

print(datetime.now().strftime('%Y%m%d-%H%M%S'))

# get parameters from external settings file (json format)
if len(call_arguments) > 1 and call_arguments[1].endswith('.json'):
    with open(call_arguments[1], 'r') as fh:
        setup_simulation = load(fh)
        s = setup_simulation
else:
    print('No input file provided.')
    print('Example call:')
    print('python run_py3DED.py "path/to/settings.json"')
    print()
    print('''
{"cif_file": "data/unit_cells/Si_CollCode51688.cif",
"output_folder": "results",
"transform_cell": false,
"thermal_sigmas": 0.1,
"centering": "F",
"box_size_x": 100,
"box_size_y": 100,
"thickness": 1000,
"mode": "ms+bw",
"energy": 200000,
"rotation_axis_orientation": 25.0,
"rotation_min": 0.0,
"rotation_max": 45.0,
"rotation_steps": 2,
"slice_thickness": 0.8,
"exit_planes": 4,
"integration_radius": 0.02,
"sg_max": 0.2,
"g_max_store": 4.0,
"sampling": 0.04,
"fft": "fftw",
"window_func": "hann",
"g_max": 8.0,
"use_wave_eq": true,
"device": "gpu",
"num_workers": 1,
"scheduler": "threads",
"precision": "float32"}
''')
    input('... press enter to exit.')
    
    
#
### OUTPUT FOLDER, pathlib
#    

# cif_file and output_folder are resolved relative to the config file's own
# directory (not the shell's cwd at invocation) when given as relative paths,
# so this script works the same regardless of where it's launched from.
config_dir = Path(call_arguments[1]).resolve().parent

cif_file = Path(s['cif_file'])
if not cif_file.is_absolute():
    cif_file = config_dir / cif_file

output_folder = Path(s['output_folder'])
if not output_folder.is_absolute():
    output_folder = config_dir / output_folder

if 'ms' not in s['mode'].lower():
    s['box_size_x'] = 1
    s['box_size_y'] = 1
    print("""NO multislice calculation. Current implemention still requires to write out a superstructure, even though Bloch wave does not use it. 
Here, a dummy superstructure with x = y = 1 Å is used.""")

# define output subfolder
output_subfolder = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{cif_file.stem}_" + \
                   f"{s['box_size_x']:.0f}x{s['box_size_y']:.0f}x{s['thickness']:.0f}_" + \
                   f"{s['rotation_steps']:.0f}"
if s['transform_cell'] is not False:
    output_subfolder += 'T'

output_base = output_folder / output_subfolder

output_base.mkdir(parents=True)

#
### LOGGING
#

logfile = output_base / 'py3DED.log'
    
# manipulate print ... is this a bad idea?
from builtins import print as ppp

fhlog = open(logfile, 'w')
def print(*args, **kwargs):
    # normal output
    ppp(*args, **kwargs)
    # log to file
    ppp(*args, file=fhlog, **kwargs)
    return None

def sep():
    print('- '*42)
    

# print parameters
print(call_arguments[1])
print(python_version)
for p in abtem, np, zarr, : # package version numbers
    if hasattr(p, "__version__") and hasattr(p, "__name__"):
        print(p.__name__, p.__version__)
        
print(datetime.now().strftime('%Y%m%d-%H%M%S'))
print('All files will be written to:')
print(output_base.resolve())
sep()

print( Path(call_arguments[1]).resolve() )
for k,v in setup_simulation.items():
    print(k,':',v)
    
# check setup
# ...this could be extended...
if s['rotation_steps'] == 1 and ( s['rotation_min'] != s['rotation_max'] ):
    print('!!! Simulate 1 orientation. Ignoring "rotation_max"')
    s['rotation_max'] = s['rotation_min']
    
if 'bw' in s['mode'].lower() and ( s['g_max'] < s['g_max_store'] ):
    print('!!! g_max is smaller than g_max_store.')
    #input('Hit any key to continue...')

sep()

#####
##### Generate supercell
#####

# setup structure
structure_UC = ase_read( cif_file, format='cif') #ase.io.read

if s['transform_cell']:
    structure_UC.set_cell( 
        np.reshape( s['transform_cell'], (3,3) ), # shape=
        scale_atoms=True)

# output unit cell info
print( structure_UC.symbols.get_chemical_formula() )
print( '\n'.join( [ f'{a:7}{e:10.5f}' for a, e in zip('a b c alpha beta gamma'.split(), structure_UC.cell.cellpar()) ] ) )

structure_config = atoms.Config()
structure_config.atoms = structure_UC
structure_config.store_path = str(output_base / 'supercell.zarr')
structure_config.rotation_axis = s['rotation_axis_orientation']
structure_config.box = (s['box_size_x'],
                        s['box_size_y'],
                        s['thickness'])
structure_config.vacancies = s['vacancies']

print()

sep()
t0 = time()
print('>>> Generating supercell ... ')

atoms.run(structure_config) # <-- this may take a couple of seconds
print(f'DONE. {time()-t0:.1f} seconds')
print(structure_config.store_path)

''' # working with zarr
zarr_file_supercell = zarr.open( structure_config.store_path, mode='a')
t = zarr_file_supercell
for k,v in t.attrs.items(): # attributes
    print(k, v)
t.atomic_positions[0,:] # data, (x,y,z) of first atom
t.unit_cell[:] # unit cell size
'''

# write attributes / meta data
print('Writing meta data as zarray attributes.')
zarr_file_supercell = open_zarr_group(structure_config.store_path, mode='a')

zarr_file_supercell.attrs['config'] = str(structure_config)
zarr_file_supercell.attrs['cif_file'] = str(s['cif_file'])
zarr_file_supercell.attrs['output_path'] = structure_config.store_path
zarr_file_supercell.attrs['reading_instruction'] = 'from py3DED.io import read_atoms_from_zarr; read_atoms_from_zarr( path_to_this_zarr_file )'
zarr_file_supercell.attrs['rotation_axis_orientation'] = f"{s['rotation_axis_orientation']:.3f} degrees from Cartesian x-axis towards y-axis."

for k in 'box_size_x', 'box_size_y', 'thickness':
    zarr_file_supercell.attrs[k] = f'{s[k]:.3f} Angstrom'

# item access, not attribute access: zarr 3.x (required by current abTEM) dropped
# the v2 sugar that let a dataset be read as zarr_file_supercell.atom
N_atoms = zarr_file_supercell['atom'].shape[0]
zarr_file_supercell.attrs['N_atoms'] = N_atoms

print('Zarray structure:')
for name, content in zarr_file_supercell.arrays():
    print(f'|__ {name:20} {content.dtype.name:8}', content.shape)

zarr_file_supercell.store.close()  # flushes a .zip store's central directory
del zarr_file_supercell

sep()

#####
##### Prepare MS / BW
#####

# default configuration
run_config = simulate.Config()

# update configuration
for k in s.keys():
    if hasattr(run_config, k):
        setattr(run_config, k, s[k])

run_config.unit_cell = structure_config.atoms
run_config.supercell = structure_config.store_path        

# rotation axis
run_config.rotation_axis = s['rotation_axis_orientation']

# occupancy / vacancies
if s['vacancies'] > 0.0:
    run_config.occupancy = 1.0 - s['vacancies']

# displacement parameters (thermal_sigmas)
if isinstance(s['thermal_sigmas'], float):
    run_config.thermal_sigma = { symbol: s['thermal_sigmas'] for symbol in structure_UC.symbols.species()}
elif isinstance(s['thermal_sigmas'], dict):
    run_config.thermal_sigma = s['thermal_sigmas']
else:
    print('!!! thermal_sigmas not recognised. Using default value of 0.1 Å.')
    run_config.thermal_sigma = { symbol: 0.1 for symbol in structure_UC.symbols.species() }
    input('Hit any key to continue...')

# dump settings file
settings_file = output_base / 'settings.json'
s['run_config'] = str( run_config )
s['structure_config'] = str( structure_config )
with open(settings_file, 'w') as fh:
    dump(s, fh, indent=2)
    print('Settings written to:', settings_file)

# copy CIF file
destination = output_base / cif_file.name
destination.write_bytes( cif_file.read_bytes() )
print('CIF file copied.')

sep()
print(run_config)
sep()

#####
##### Run MS
#####
if 'ms' in s['mode'].lower():
    sep()
    output_file = output_base.resolve() / 'ms.zarr'
    print('Output_file:', output_file)
    print('>>> Run multislice calculations ...')
    t0 = time()
    
    with run_config.ctx(run_mode="ms", 
                    num_workers=s['num_workers'],
                    store_path=str(output_file)):
        output_ms = simulate.run(run_config, save_to_disk=True) # this takes a while ...

    duration = time()-t0
    print(f'DONE. {duration/60:.2f} minutes')

    # write meta data
    zarr_file_ms = open_zarr_group(output_file, mode='a')
    for e in dir(run_config):
        if not e.startswith('_') and e not in ('ctx', 'unit_cell'):
            zarr_file_ms.attrs[e] = getattr(run_config, e)
    zarr_file_ms.attrs['setup_dict'] = str(s)
    zarr_file_ms.attrs['runtime_seconds'] = duration

    # print out basic info
    print('Zarray structure:')
    for name, content in zarr_file_ms.arrays():
        print(f'|__ {name:20} {content.dtype.name:8}', content.shape)
    zarr_file_ms.store.close()  # flushes a .zip store's central directory


#####
##### Run BW
#####
if 'bw' in s['mode'].lower():
    sep()
    output_file = output_base.resolve() / 'bw.zarr'
    print('Output_file:', output_file)
    print('>>> Run Bloch wave calculations ...')
    t0 = time()
    
    with run_config.ctx(run_mode="bw", 
                    num_workers=s['num_workers'],
                    store_path=str(output_file)):
        output_bw = simulate.run(run_config, save_to_disk=True) # this takes a while ...
        
    duration = time()-t0
    print(f'DONE. {duration/60:.2f} minutes')

    # write meta data
    zarr_file_bw = open_zarr_group(output_file, mode='a')
    for e in dir(run_config):
        if not e.startswith('_') and e not in ('ctx', 'unit_cell'):
            zarr_file_bw.attrs[e] = getattr(run_config, e)
    zarr_file_bw.attrs['setup_dict'] = str(s)
    zarr_file_bw.attrs['runtime_seconds'] = duration

    # print out basic info
    print('Zarray structure:')
    for name, content in zarr_file_bw.arrays():
        print(f'|__ {name:20} {content.dtype.name:8}', content.shape)
    zarr_file_bw.store.close()  # flushes a .zip store's central directory

# z = zarr.open( output_file, mode='r')
# hkl = z.attrs['kwargs0']['miller_indices']
# alpha = z.attrs['kwargs0']['ensemble_axes_metadata'][0]['values']
# thickness = z.attrs['kwargs0']['ensemble_axes_metadata'][1]['values']
# datapoint = z.array0[0,0,0]


##########################################################################
sep()
print(datetime.now().strftime('%Y%m%d-%H%M%S'))
sep()

# close logfie
fhlog.close()

# reset print if this script is called from within another script (e.g. notebook,, %run  ...)
from builtins import print