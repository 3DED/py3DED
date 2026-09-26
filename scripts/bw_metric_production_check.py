#!/usr/bin/env python
"""Production-scale check of abTEM's Bloch-wave metric fix: old vs new BW
against an existing multislice result.

The fix (abTEM c52f5ba3, "Bloch waves: apply the metric consistently for each
form of the equation") only changes Bloch waves, so the multislice side does
not need recomputing. Given a finished run_py3DED.py output folder (with
settings.json, supercell.zarr.zip, the CIF file and ms.zarr.zip), this script
reruns ONLY the Bloch-wave part with exactly that run's configuration, twice:

    bw_newmetric.zarr.zip   the installed (fixed) abTEM
    bw_oldmetric.zarr.zip   the same, with the pre-fix code paths patched in
                            (verbatim copies, from compare_bloch_metric.py)

and compares both against the run's ms.zarr.zip with the same R_z as
plot_rz_vs_energy.py / run_py3DED_compare2.py (rotation-integrated intensities,
000 excluded, g_max_limit and per-energy tolerance filter). Single-energy and
energy-ensemble runs both work.

It writes, into the run folder:
    bw_newmetric.zarr.zip, bw_oldmetric.zarr.zip   (reused if present; --overwrite)
    R_z_metric_check.png / .pdf                    R_z(MS, BW) vs thickness, per
                                                   energy: old dashed, new solid
    R_z_metric_check.csv                           the curves

Needs the fixed abTEM (abtem.bloch.dynamical._metric present) and the run's MS
data computed with the per-energy indexing fix (abTEM 686fd0ee or later) for
energy ensembles.

Usage, on the GPU workstation, from the py3DED repo root:
    python scripts/bw_metric_production_check.py \\
        examples/Si_kinetic_energy/results_energy_ensemble/<run folder>

Options:
    --propagator paraxial|exact   override the run's propagator (BW side only;
                                  compare against an MS run made with the same)
    --sg-max, --g-max             override the run's Bloch-wave beam set
    --device, --num-workers       override the run's settings
    --overwrite                   recompute BW stores that already exist
    --only new|old                compute just one variant
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import abtem  # noqa: E402
import abtem.bloch.dynamical as dyn  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import xarray  # noqa: E402
from ase.io import read as ase_read  # noqa: E402

from compare_bloch_metric import (  # noqa: E402
    old_dynamical_scattering,
    old_structure_matrix,
)
from plot_rz_vs_energy import bragg_r_z, find_store  # noqa: E402
from py3DED import simulate  # noqa: E402


def run_config_from_folder(folder: Path, args) -> simulate.Config:
    """Rebuild the run's simulate.Config the way run_py3DED.py built it."""
    s = json.loads((folder / "settings.json").read_text())
    config = simulate.Config()
    for k, v in s.items():
        if hasattr(config, k):
            setattr(config, k, v)

    cif = folder / Path(s["cif_file"]).name
    unit_cell = ase_read(cif, format="cif")
    if s.get("transform_cell"):
        unit_cell.set_cell(np.reshape(s["transform_cell"], (3, 3)), scale_atoms=True)
    config.unit_cell = unit_cell
    supercell = find_store(folder, "supercell")
    if supercell is None:
        raise FileNotFoundError(f"no supercell store in {folder}")
    config.supercell = str(supercell)
    config.rotation_axis = s["rotation_axis_orientation"]
    if s.get("vacancies", 0.0) > 0.0:
        config.occupancy = 1.0 - s["vacancies"]
    sigmas = s["thermal_sigmas"]
    if isinstance(sigmas, (int, float)):
        sigmas = {sym: float(sigmas) for sym in unit_cell.symbols.species()}
    config.thermal_sigma = sigmas

    for name in ("propagator", "sg_max", "g_max", "device", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    config.run_mode = "bw"
    return config


class old_metric_code:
    """Temporarily swap the pre-fix Bloch-wave code paths into abTEM."""

    def __enter__(self):
        self._saved = (dyn.calculate_structure_matrix, dyn.calculate_dynamical_scattering)

        def structure_matrix(structure_factor, hkl, hkl_selected, cell, energy, gpts,
                             use_wave_eq=False):
            return old_structure_matrix(structure_factor, hkl, hkl_selected, cell,
                                        energy, gpts, use_wave_eq)

        def dynamical_scattering(structure_matrix, hkl, cell, energy, thicknesses,
                                 use_wave_eq=False):
            out = old_dynamical_scattering(structure_matrix, hkl, cell, energy,
                                           np.atleast_1d(thicknesses))
            return out if np.ndim(thicknesses) else out[0]

        dyn.calculate_structure_matrix = structure_matrix
        dyn.calculate_dynamical_scattering = dynamical_scattering
        return self

    def __exit__(self, *exc):
        dyn.calculate_structure_matrix, dyn.calculate_dynamical_scattering = self._saved


def compute_bw(config, store: Path, old: bool, overwrite: bool):
    if store.exists() and not overwrite:
        print(f"  {store.name} exists, reusing (--overwrite to recompute)")
        return
    if store.exists():
        store.unlink()
    print(f"  computing {store.name} ({'old' if old else 'new'} metric code) ...", flush=True)
    simulate.set_abtem_config(config)
    with config.ctx(store_path=str(store)):
        if old:
            with old_metric_code():
                simulate.run(config, save_to_disk=True)
        else:
            simulate.run(config, save_to_disk=True)


def load(store: Path, g_max_limit: float):
    data = abtem.from_zarr(str(store)).to_data_array()
    if "Energy" not in data.dims:
        energy = abtem.from_zarr(str(store)).metadata.get("energy")
        data = data.expand_dims(Energy=[float(energy)])
    return data.sel(hkl=(data["k"] < g_max_limit))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_folder")
    p.add_argument("--propagator", choices=("paraxial", "exact"))
    p.add_argument("--sg-max", dest="sg_max", type=float)
    p.add_argument("--g-max", dest="g_max", type=float)
    p.add_argument("--device")
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--g-max-limit", type=float, default=2.0)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", choices=("new", "old"))
    args = p.parse_args()

    if not hasattr(dyn, "_metric"):
        raise SystemExit("installed abTEM lacks the Bloch-wave metric fix (c52f5ba3)")

    folder = Path(args.run_folder).resolve()
    ms_store = find_store(folder, "ms")
    if ms_store is None:
        raise FileNotFoundError(f"no ms store in {folder}")

    config = run_config_from_folder(folder, args)
    print(f"{folder.name}: energy {config.energy}, {config.rotation_steps} rotations, "
          f"sg_max {config.sg_max}, g_max {config.g_max}, propagator {config.propagator}, "
          f"use_wave_eq -> {simulate.propagator_settings(config)[1]}, device {config.device}")

    stores = {v: folder / f"bw_{v}metric.zarr.zip" for v in ("new", "old")}
    for variant, store in stores.items():
        if args.only in (None, variant):
            compute_bw(config, store, old=(variant == "old"), overwrite=args.overwrite)

    print("Loading and comparing ...", flush=True)
    ms = load(ms_store, args.g_max_limit)
    bw = {v: load(s, args.g_max_limit) for v, s in stores.items() if s.exists()}
    aligned = xarray.align(ms, *bw.values(), join="inner")
    ms = aligned[0].compute()
    bw = {v: a.compute() for v, a in zip(bw, aligned[1:])}

    # One tolerance mask for both variants, from the new BW (as bragg_r_z would
    # build it): otherwise reflections right at the threshold can be kept for
    # one variant and dropped for the other, and R_z then measures that filter
    # difference rather than the metric fix.
    reference = bw["new"] if "new" in bw else next(iter(bw.values()))
    keep = reference.mean(dim="z").max(dim="x_rotation") > args.tolerance
    curves = {
        v: bragg_r_z(b.where(keep), ms.where(keep), tolerance=None)
        for v, b in bw.items()
    }

    energies = sorted(float(e) for e in next(iter(curves.values()))["Energy"].data)
    print(f"\n{'energy':>8} " + "".join(f"{'R_z ' + v + ' mean':>16}{'max':>8}" for v in curves)
          + ("   R(BW old, BW new)" if len(bw) == 2 else ""))
    for e in energies:
        line = f"{e/1e3:6.0f}keV "
        for v, c in curves.items():
            r = c.sel(Energy=e)
            line += f"{float(r.mean()):16.3f}{float(r.max()):8.3f}"
        if len(bw) == 2:
            o, n = xarray.align(bw["old"].sel(Energy=e), bw["new"].sel(Energy=e), join="inner")
            o, n = o.drop_sel(hkl="0 0 0").compute(), n.drop_sel(hkl="0 0 0").compute()
            line += f"   {100 * float(np.abs(o - n).sum() / n.sum()):8.3f} %"
        print(line)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    rows = []
    for i, e in enumerate(energies):
        color = cmap(i / max(len(energies) - 1, 1))
        for v, style in (("new", "-"), ("old", "--")):
            if v not in curves:
                continue
            c = curves[v].sel(Energy=e).sortby("z")
            ax.plot(c["z"], c, style, color=color,
                    label=f"{e/1e3:.0f} keV, {v} metric")
            rows += [(e, v, float(z), float(r)) for z, r in zip(c["z"].data, c.data)]
    ax.set_xlabel("thickness (Å)")
    ax.set_ylabel("$R_z$ (%)")
    ax.set_title(f"MS vs BW, Bloch-wave metric: new (solid) vs old (dashed)\n{folder.name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"R_z_metric_check.{ext}", dpi=200)
    with open(folder / "R_z_metric_check.csv", "w") as fh:
        fh.write("energy_eV,metric,z_A,R_z_percent\n")
        fh.writelines(f"{e:.0f},{v},{z:.3f},{r:.5f}\n" for e, v, z, r in rows)
    print(f"\nSaved {folder / 'R_z_metric_check.png'} (+ .pdf, .csv)")


if __name__ == "__main__":
    main()
