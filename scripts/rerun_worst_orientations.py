#!/usr/bin/env python
"""Rerun the orientations with the worst frame-by-frame R factor (R_z_alpha),
with the multislice and several Bloch-wave variants, to see what explains them.

The question: for some orientations of the article's runs, R_z_alpha (the
Bragg R factor of a single frame, run_py3DED_compare2.py's
Bragg_R_z_alpha.csv) exceeds 10 %. Those runs used use_wave_eq=True for Bloch
waves and the first-order Fresnel propagator for multislice -- the same
(paraxial) equation -- but abTEM's Bloch waves then also applied the metric
M = (1 + g_z/K)^-1/2, which belongs only to the standard (use_wave_eq=False)
form (fixed in abTEM c52f5ba3). This script separates the two candidate
causes, the excitation-error form and the metric, frame by frame.

The source is either

  - a finished run folder (settings.json, the CIF file, supercell, ms and bw
    stores; V2026.7 run folders work), or
  - the run's run_py3DED.py json, when that folder no longer exists. A new
    folder is made as run_py3DED.py would (settings.json, CIF, supercell --
    generated with today's code), and the article's BW is recomputed for the
    whole rotation series (cheap; it gives the tolerance mask). Without the
    article's MS, the orientations come from --angles, or from the article's
    Bragg_R_z_alpha.csv with --alpha-csv. Pass the new folder, not the json,
    on later calls, so nothing is regenerated.

The script

  1. ranks the orientations by R_z_alpha (largest R over z > --zmin), from
     the run's own ms vs bw stores or from --alpha-csv, and picks the
     --worst N of them, plus --controls M well-behaved ones for comparison
     (or takes --angles explicitly);
  2. reruns just those orientations, from the run's own supercell:

       ms_paraxial    multislice, FourierMultislice(order=1)  (the article's MS)
       bw_old_waveeq  use_wave_eq=True, pre-fix metric code   (the article's BW)
       bw_new_waveeq  use_wave_eq=True, fixed metric (M = 1)  -- metric fix only
       bw_old_std     use_wave_eq=False, pre-fix metric code  -- excitation error
       bw_new_std     use_wave_eq=False, fixed metric             change only
     and with --exact also
       ms_exact       FourierMultislice(order="exact")
       bw_new_exact   use_wave_eq="exact"

  3. prints R_z_alpha (max over z > zmin, and mean over z) per orientation for
     each BW variant against the matching MS, plus two reproduction checks:
     bw_old_waveeq vs the run's bw store and ms_paraxial vs the run's ms store
     (both should be small; if not, the rerun does not reproduce the run, and
     the rest is not comparable). From a json there is no run MS, and the run
     BW is the same code as bw_old_waveeq, so these checks say nothing; check
     instead that "bw_old_waveeq vs ms_paraxial" reproduces the article's
     numbers for those orientations.

How to read it: if bw_new_waveeq vs ms_paraxial is much better than
bw_old_waveeq vs ms_paraxial, the metric was the problem. If bw_*_std agrees
worse than bw_new_waveeq, the excitation-error form was right all along.

Reflections: g < --g-max-limit, tolerance mask taken from the run's FULL bw
store (as compare2 does, over all orientations), 000 excluded; unscaled
intensities, as compare2's R_z_alpha.

Outputs into <run folder>/worst_orientations/: one store per variant (reused
if present; --overwrite), R_z_alpha.csv (all curves), R_z_alpha_summary.csv
and R_z_alpha.png/.pdf (one panel per orientation).

Needs the fixed abTEM (abTEM 31bd1c75 or later, #384) installed. Run on the
GPU workstation, from the py3DED repo root:

    python scripts/rerun_worst_orientations.py <run folder>
    python scripts/rerun_worst_orientations.py <run folder> --worst 4 --controls 2 --exact
    python scripts/rerun_worst_orientations.py <run folder> --angles 12.5 37.0
    python scripts/rerun_worst_orientations.py <run folder> --select-only
    python scripts/rerun_worst_orientations.py <run json> --alpha-csv Bragg_R_z_alpha.csv
    python scripts/rerun_worst_orientations.py <run json> --angles 12.5 37.0

Options:
    --precision      default float64 (the run's own setting with "run")
    --reuse-ms       compare against the run's ms store instead of rerunning MS
    --only           compute only these variants (comparisons need their pairs)
    --device, --num-workers override the run's settings
"""
import argparse
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

from bw_metric_production_check import (  # noqa: E402
    old_metric_code,
    run_config_from_folder,
)
from plot_rz_vs_energy import find_store  # noqa: E402
from py3DED import simulate  # noqa: E402

# name: (run_mode, propagator, use_wave_eq, old metric code)
VARIANTS = {
    "ms_paraxial": ("ms", "paraxial", True, False),
    "bw_old_waveeq": ("bw", "paraxial", True, True),
    "bw_new_waveeq": ("bw", "paraxial", True, False),
    "bw_old_std": ("bw", "paraxial", False, True),
    "bw_new_std": ("bw", "paraxial", False, False),
    "ms_exact": ("ms", "exact", True, False),
    "bw_new_exact": ("bw", "exact", True, False),
}
EXACT_ONLY = ("ms_exact", "bw_new_exact")

# (label, BW variant, MS variant): the R_z_alpha comparisons reported
COMPARISONS = [
    ("run: bw vs ms", "run_bw", "run_ms"),
    ("check: bw_old_waveeq vs run bw", "bw_old_waveeq", "run_bw"),
    ("check: ms_paraxial vs run ms", "ms_paraxial", "run_ms"),
    ("bw_old_waveeq vs ms_paraxial", "bw_old_waveeq", "ms_paraxial"),
    ("bw_new_waveeq vs ms_paraxial", "bw_new_waveeq", "ms_paraxial"),
    ("bw_old_std vs ms_paraxial", "bw_old_std", "ms_paraxial"),
    ("bw_new_std vs ms_paraxial", "bw_new_std", "ms_paraxial"),
    ("bw_new_exact vs ms_exact", "bw_new_exact", "ms_exact"),
]


def load(store: Path, expected_deg: np.ndarray, g_max_limit: float):
    """The store as (x_rotation [deg], Energy, z, hkl), g < g_max_limit, with
    coordinates rounded so stores from different code versions align."""
    data = abtem.from_zarr(str(store)).to_data_array()
    if "Energy" not in data.dims:
        energy = abtem.from_zarr(str(store)).metadata.get("energy")
        data = data.expand_dims(Energy=[float(energy)], axis=1)
    rot = np.asarray(data["x_rotation"], dtype=float)
    # V2026.7 stores kept radians under a "deg" label
    if not np.allclose(rot, expected_deg[: len(rot)]) and np.allclose(
        np.rad2deg(rot), expected_deg[: len(rot)]
    ):
        rot = np.rad2deg(rot)
    data = data.assign_coords(
        x_rotation=np.round(rot, 4),
        z=np.round(np.asarray(data["z"], dtype=float), 3),
    )
    return data.sel(hkl=(data["k"] < g_max_limit))


def r_z_alpha(bw, ms):
    """Bragg R factor of single frames (%), per (x_rotation, Energy, z), 000
    excluded -- compare2's R_frames. Masked (NaN) reflections are skipped."""
    bw, ms = xarray.align(bw, ms, join="inner")
    bw = bw.drop_sel(hkl="0 0 0", errors="ignore")
    ms = ms.drop_sel(hkl="0 0 0", errors="ignore")
    return 100 * np.abs(bw - ms).sum(dim="hkl") / bw.sum(dim="hkl")


def summarize(r, zmin):
    deep = r.sel(z=r["z"] > zmin) if (r["z"] > zmin).any() else r
    return deep.max(dim="z"), r.mean(dim="z")


def ranked(run_r, zmin):
    """The run's R_z_alpha, max over z > zmin (and energies), worst first."""
    worst = summarize(run_r, zmin)[0].max(dim="Energy")
    return worst.sortby(worst, ascending=False)


def pick_angles(worst, args):
    angles = list(worst["x_rotation"].data[: args.worst])
    if args.controls:
        angles += list(worst["x_rotation"].data[::-1][: args.controls])
    return np.array(sorted(set(float(a) for a in angles)))


def choose_angles(worst, all_deg, args):
    """--angles, or the --worst/--controls picks from the ranking; prints it."""
    if args.angles:
        angles = np.round(np.array(sorted(args.angles)), 4)
        missing = [a for a in angles if a not in all_deg]
        if missing:
            raise SystemExit(f"{missing} are not orientations of this run: {all_deg}")
    else:
        angles = pick_angles(worst, args)
    if worst is not None:
        print(f"\nRun's R_z_alpha, max over z > {args.zmin:g} Å, worst first:")
        for a, r in zip(worst["x_rotation"].data[:12], worst.data[:12]):
            print(f"  {a:9.3f} deg  {r:7.2f} %{'   <- rerun' if a in angles else ''}")
    return angles


def select(data, labels):
    return data.sel(hkl=data["hkl"].isin(labels).data)


def compute(name, base_config, angles_deg, store: Path, args):
    if store.exists() and not args.overwrite:
        print(f"  {store.name} exists, reusing (--overwrite to recompute)")
        return
    if store.exists():
        import shutil

        shutil.rmtree(store) if store.is_dir() else store.unlink()
    run_mode, propagator, use_wave_eq, old = VARIANTS[name]
    print(f"  computing {name} ({run_mode}, propagator {propagator}, "
          f"use_wave_eq {use_wave_eq}, {'pre-fix' if old else 'fixed'} metric) ...",
          flush=True)
    config = base_config
    simulate.set_abtem_config(config)
    angles = np.deg2rad(angles_deg)
    saved = simulate.get_x_angles
    simulate.get_x_angles = lambda config: angles  # every caller looks it up here
    try:
        with config.ctx(run_mode=run_mode, propagator=propagator,
                        use_wave_eq=use_wave_eq, store_path=str(store)):
            if old:
                with old_metric_code():
                    simulate.run(config, save_to_disk=True)
            else:
                simulate.run(config, save_to_disk=True)
    finally:
        simulate.get_x_angles = saved


def prepare_from_config(json_path: Path, output_root):
    """A run folder from a run_py3DED.py json, as run_py3DED.py builds it --
    settings.json, the CIF file and the supercell -- but with no MS or BW run."""
    from datetime import datetime
    from json import dump, load as load_json

    from ase.io import read as ase_read

    from py3DED import atoms

    s = load_json(open(json_path))
    s.setdefault("transform_cell", False)
    s.setdefault("vacancies", 0.0)
    base = json_path.resolve().parent  # relative paths as run_py3DED.py
    cif = Path(s["cif_file"])
    cif = cif if cif.is_absolute() else base / cif
    root = Path(output_root or s["output_folder"])
    root = root if root.is_absolute() else base / root
    folder = root / (f"{datetime.now():%Y%m%d-%H%M%S}_{cif.stem}_"
                     f"{s['box_size_x']:.0f}x{s['box_size_y']:.0f}x{s['thickness']:.0f}_"
                     f"{s['rotation_steps']:.0f}_worst")
    folder.mkdir(parents=True)

    unit_cell = ase_read(cif, format="cif")
    if s["transform_cell"]:
        unit_cell.set_cell(np.reshape(s["transform_cell"], (3, 3)), scale_atoms=True)
    structure = atoms.Config()
    structure.atoms = unit_cell
    structure.store_path = str(folder / "supercell.zarr.zip")
    structure.rotation_axis = s["rotation_axis_orientation"]
    structure.box = (s["box_size_x"], s["box_size_y"], s["thickness"])
    structure.vacancies = s["vacancies"]
    structure.rotation_range = (s["rotation_min"], s["rotation_max"])
    print(f"Generating the supercell in {folder} ...", flush=True)
    atoms.run(structure)

    (folder / cif.name).write_bytes(cif.read_bytes())
    with open(folder / "settings.json", "w") as fh:
        dump(s, fh, indent=2)
    return folder


def worst_from_csv(path, all_deg, zmin):
    """The run's R_z_alpha ranking from its Bragg_R_z_alpha.csv
    (run_py3DED_compare2.py): rows alpha [deg], columns z [Å]."""
    import pandas as pd

    table = pd.read_csv(path, index_col=0, skipinitialspace=True)
    z = table.columns.astype(float)
    r = (table.loc[:, z > zmin] if (z > zmin).any() else table).max(axis=1)
    alpha = np.asarray(r.index, dtype=float)
    if np.abs(alpha).max() > 1.5 * np.abs(all_deg).max() > 0:
        alpha = np.deg2rad(alpha)  # compare2 converted degrees as if radians
    nearest = np.abs(all_deg[None] - alpha[:, None]).argmin(axis=1)
    if np.abs(all_deg[nearest] - alpha).max() > 1e-2:
        raise SystemExit(f"the orientations in {path} are not this run's: {all_deg}")
    worst = xarray.DataArray(r.to_numpy(), dims="x_rotation",
                             coords={"x_rotation": all_deg[nearest]})
    return worst.sortby(worst, ascending=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="run folder, or a run_py3DED.py json config")
    p.add_argument("--alpha-csv", help="the run's Bragg_R_z_alpha.csv, to pick the worst "
                   "orientations when the folder has no ms store")
    p.add_argument("--output-folder", help="for a json source: where to put the new run "
                   "folder (default: the json's output_folder)")
    p.add_argument("--worst", type=int, default=5)
    p.add_argument("--controls", type=int, default=2)
    p.add_argument("--angles", type=float, nargs="+", help="orientations [deg] to rerun")
    p.add_argument("--exact", action="store_true", help="also the exact-propagator pair")
    p.add_argument("--reuse-ms", action="store_true")
    p.add_argument("--only", nargs="+", choices=list(VARIANTS))
    p.add_argument("--precision", default="float64", help='"float32", "float64" or "run"')
    p.add_argument("--device")
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--g-max-limit", type=float, default=2.0)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--zmin", type=float, default=20.0, help="R max taken over z > zmin [Å]")
    p.add_argument("--select-only", action="store_true", help="just list the worst orientations")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not hasattr(dyn, "_metric"):
        raise SystemExit("installed abTEM lacks the Bloch-wave metric fix (c52f5ba3)")

    source = Path(args.source).resolve()
    if source.suffix == ".json":
        if not (args.angles or args.alpha_csv):
            raise SystemExit("from a json, give --angles or --alpha-csv: without the "
                             "article's MS there is nothing to rank orientations by")
        folder = prepare_from_config(source, args.output_folder)
        print(f"Next time, pass the folder instead of the json: {folder}")
    else:
        folder = source
    run_ms, run_bw = find_store(folder, "ms"), find_store(folder, "bw")

    # the propagator/sg_max/... overrides of run_config_from_folder stay unset
    for name in ("propagator", "sg_max", "g_max"):
        setattr(args, name, None)
    config = run_config_from_folder(folder, args)
    if args.precision != "run":
        config.precision = args.precision
    all_deg = np.round(np.rad2deg(simulate.get_x_angles(config)), 4)

    print(f"{folder.name}: energy {config.energy}, {len(all_deg)} orientations, "
          f"sg_max {config.sg_max}, g_max {config.g_max}, precision {config.precision}, "
          f"device {config.device}")
    if run_ms is None:
        worst = worst_from_csv(args.alpha_csv, all_deg, args.zmin) if args.alpha_csv else None
        if worst is None and not args.angles:
            raise SystemExit(f"no ms store in {folder}: give --angles or --alpha-csv")
        angles = choose_angles(worst, all_deg, args)
        if args.select_only:
            return
        if run_bw is None:
            # The article's BW over the whole series: cheap, and it gives the
            # tolerance mask over all orientations, as compare2 builds it.
            run_bw = folder / "bw.zarr.zip"
            print("No bw store: computing the article's BW (use_wave_eq=True, pre-fix "
                  "metric) for all orientations", flush=True)
            compute("bw_old_waveeq", config, all_deg, run_bw, args)

    print("Reading the run's stores ...", flush=True)
    full_bw = load(run_bw, all_deg, args.g_max_limit).compute()
    # tolerance mask over the FULL rotation series, from BW, as compare2
    keep = full_bw.mean(dim="z").max(dim="x_rotation").max(dim="Energy") > args.tolerance
    print(f"  {int(keep.sum())} of {keep.size} reflections (g < {args.g_max_limit}) "
          f"above tolerance {args.tolerance:g}")
    labels = keep["hkl"].data[keep.data]
    full_bw = select(full_bw, labels)
    if run_ms is not None:
        full_ms = select(load(run_ms, all_deg, args.g_max_limit).compute(), labels)
        worst = ranked(r_z_alpha(full_bw, full_ms), args.zmin)
        angles = choose_angles(worst, all_deg, args)
        if args.select_only:
            return

    out = folder / "worst_orientations"
    out.mkdir(exist_ok=True)
    names = [n for n in VARIANTS if args.exact or n not in EXACT_ONLY]
    if args.reuse_ms:
        names = [n for n in names if n != "ms_paraxial"]
    if args.only:
        names = [n for n in names if n in args.only]
    stores = {n: out / f"{n}.zarr.zip" for n in VARIANTS}
    angle_tag = out / "angles.txt"
    if angle_tag.exists() and not args.overwrite:
        previous = np.loadtxt(angle_tag, ndmin=1)
        if not np.array_equal(np.round(previous, 4), angles):
            raise SystemExit(f"{out} holds other orientations ({previous}); "
                             "use --overwrite or move it")
    np.savetxt(angle_tag, angles, fmt="%.4f")

    print(f"\nRerunning {len(angles)} orientations: {', '.join(f'{a:g}' for a in angles)} deg")
    for n in names:
        compute(n, config, angles, stores[n], args)

    print("\nLoading and comparing ...", flush=True)
    data = {"run_bw": full_bw.sel(x_rotation=angles)}
    if run_ms is not None:
        data["run_ms"] = full_ms.sel(x_rotation=angles)
    for n, store in stores.items():  # also ones left by an earlier call
        if store.exists():
            data[n] = select(load(store, angles, args.g_max_limit), labels).compute()
    if args.reuse_ms:
        data["ms_paraxial"] = data["run_ms"]

    curves = {}
    for label, a, b in COMPARISONS:
        if a in data and b in data and not (args.reuse_ms and "ms_paraxial vs run ms" in label):
            curves[label] = r_z_alpha(data[a], data[b])

    energies = list(next(iter(curves.values()))["Energy"].data)
    rows, summary = [], []
    for e in energies:
        print(f"\nEnergy {float(e)/1e3:.0f} keV.  R_z_alpha (%): max over z > {args.zmin:g} Å "
              f"/ mean over z")
        print(f"{'':34}" + "".join(f"{a:>14.3f}" for a in angles))
        for label, r in curves.items():
            mx, mean = summarize(r.sel(Energy=e), args.zmin)
            print(f"{label:34}" + "".join(
                f"{float(mx.sel(x_rotation=a)):7.2f}/{float(mean.sel(x_rotation=a)):<6.2f}"
                for a in angles))
            for a in angles:
                summary.append((float(e), label, a, float(mx.sel(x_rotation=a)),
                                float(mean.sel(x_rotation=a))))
                c = r.sel(Energy=e, x_rotation=a)
                rows += [(float(e), label, a, float(z), float(v))
                         for z, v in zip(c["z"].data, c.data)]

    with open(out / "R_z_alpha_summary.csv", "w") as fh:
        fh.write("energy_eV,comparison,alpha_deg,R_max_z_gt_zmin_percent,R_mean_percent\n")
        fh.writelines(f"{e:.0f},{l},{a:.4f},{m:.4f},{n:.4f}\n" for e, l, a, m, n in summary)
    with open(out / "R_z_alpha.csv", "w") as fh:
        fh.write("energy_eV,comparison,alpha_deg,z_A,R_percent\n")
        fh.writelines(f"{e:.0f},{l},{a:.4f},{z:.3f},{v:.5f}\n" for e, l, a, z, v in rows)

    styles = {
        "run: bw vs ms": dict(color="k", lw=2.5, alpha=0.4),
        "check: bw_old_waveeq vs run bw": dict(color="0.5", ls=":"),
        "check: ms_paraxial vs run ms": dict(color="0.5", ls="-."),
        "bw_old_waveeq vs ms_paraxial": dict(color="C3"),
        "bw_new_waveeq vs ms_paraxial": dict(color="C0"),
        "bw_old_std vs ms_paraxial": dict(color="C1", ls="--"),
        "bw_new_std vs ms_paraxial": dict(color="C2", ls="--"),
        "bw_new_exact vs ms_exact": dict(color="C4"),
    }
    for e in energies:
        ncol = min(len(angles), 4)
        nrow = -(-len(angles) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow + 1.0),
                                 squeeze=False, sharex=True)
        for ax, a in zip(axes.flat, angles):
            for label, r in curves.items():
                c = r.sel(Energy=e, x_rotation=a).sortby("z")
                ax.plot(c["z"], c, label=label, **styles[label])
            run = "" if worst is None else f"  (run: {float(worst.sel(x_rotation=a)):.1f} %)"
            ax.set_title(f"α = {a:g}°{run}", fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes.flat[len(angles):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("thickness (Å)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$R_{z,\alpha}$ (%)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(f"{folder.name}, {float(e)/1e3:.0f} keV", fontsize=10)
        # the bottom 1 in holds the legend
        fig.tight_layout(rect=(0, 1.0 / (3.2 * nrow + 1.0), 1, 1))
        suffix = "" if len(energies) == 1 else f"_{float(e)/1e3:.0f}keV"
        for ext in ("png", "pdf"):
            fig.savefig(out / f"R_z_alpha{suffix}.{ext}", dpi=200)
        plt.close(fig)
    print(f"\nSaved R_z_alpha.png/.pdf, R_z_alpha.csv, R_z_alpha_summary.csv in {out}")


if __name__ == "__main__":
    main()
