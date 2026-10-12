#!/usr/bin/env python
"""Compare MS vs. BW for a single py3DED run whose config.energy was a list
(abTEM's energy ensemble), plotting the Bragg R factor of integrated
intensities as a function of thickness -- R_z, the same metric as
run_py3DED_compare2.py's R_z.pdf -- with one curve per energy on a shared
plot, instead of one plot per energy.

Usage:
    python plot_rz_vs_energy.py <result_folder> [--g_max_limit VALUE] [--tolerance VALUE]

result_folder can either be one run's own output directory (directly
containing ms.zarr/bw.zarr, or their .zip forms), or the output_folder a
config points at (e.g. examples/Si_kinetic_energy/results_energy_ensemble/)
-- each run of run_py3DED.py creates its own timestamped subfolder there, so
this script looks one level down for a subfolder with both an ms and a bw
store if result_folder itself doesn't have them directly. Both stores must
carry an "Energy" dimension, which abtem.from_zarr(...).to_data_array()
exposes automatically.
"""
import argparse
from pathlib import Path

import abtem
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray


def find_store(folder: Path, stem: str):
    for name in (f"{stem}.zarr", f"{stem}.zarr.zip", f"{stem}.zip"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def find_run_folder(result_folder: Path) -> Path:
    """Return the folder that actually has both an ms and a bw store: either
    result_folder itself, or -- if not -- whichever of its immediate
    subfolders does. run_py3DED.py names each run's subfolder with a leading
    YYYYMMDD-HHMMSS timestamp, so subfolders sort chronologically; if more
    than one qualifies, the most recent one is used."""
    if find_store(result_folder, "ms") and find_store(result_folder, "bw"):
        return result_folder

    candidates = sorted(
        f for f in result_folder.iterdir()
        if f.is_dir() and find_store(f, "ms") and find_store(f, "bw")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run with both an ms and a bw store found directly under "
            f"{result_folder}, nor in any of its immediate subfolders."
        )
    if len(candidates) > 1:
        print(
            f"Found {len(candidates)} runs under {result_folder}: "
            f"{', '.join(c.name for c in candidates)}"
        )
        print(f"Using the most recent: {candidates[-1].name}")
    return candidates[-1]


def bragg_r_z(data_bw, data_ms, tolerance=1e-6):
    """Bragg R factor of rotation-integrated intensities (excl. 000), one per
    (energy, thickness) -- per energy, identical to run_py3DED_compare2.py's
    R_z for a single-energy run with scale=False and no averaging_depth.

    The tolerance filter is applied per energy, as compare2 would when run on
    each energy on its own: a reflection that is weak at one energy but not
    at another only counts where it is above tolerance."""
    if tolerance:
        # strongest intensity along alpha, averaged by z, using BW as reference
        max_bw = data_bw.mean(dim="z").max(dim="x_rotation")  # (Energy, hkl)
        keep = max_bw > tolerance
        any_keep = keep.any(dim="Energy").data
        data_bw = data_bw.sel(hkl=any_keep)
        data_ms = data_ms.sel(hkl=any_keep)
        keep = keep.sel(hkl=any_keep)
    else:
        keep = True

    int_bw = data_bw.sum(dim="x_rotation").where(keep)
    int_ms = data_ms.sum(dim="x_rotation").where(keep)

    # masked-out (NaN) reflections are skipped by the hkl sums
    delta = np.abs(int_bw - int_ms)
    return 100 * delta.drop_sel(hkl="0 0 0").sum(dim="hkl") / int_bw.drop_sel(
        hkl="0 0 0"
    ).sum(dim="hkl")


def main(result_folder, g_max_limit=2.0, tolerance=1e-6):
    result_folder = find_run_folder(Path(result_folder))

    ms_store = find_store(result_folder, "ms")
    bw_store = find_store(result_folder, "bw")
    if ms_store is None or bw_store is None:
        raise FileNotFoundError(
            f"Need both an ms and a bw store under {result_folder}"
        )

    print(f"Reading MS data from {ms_store} ...")
    data_ms = abtem.from_zarr(str(ms_store)).to_data_array()
    print(f"Reading BW data from {bw_store} ...")
    data_bw = abtem.from_zarr(str(bw_store)).to_data_array()

    for name, data in (("MS", data_ms), ("BW", data_bw)):
        if "Energy" not in data.dims:
            raise ValueError(
                f"{name} store has no 'Energy' dimension -- this script needs a "
                "run whose config.energy was a list (abTEM's energy ensemble)."
            )

    if set(data_bw["hkl"].data) != set(data_ms["hkl"].data):
        print("hkl sets differ between MS and BW -- aligning on their intersection.")
        data_bw, data_ms = xarray.align(data_bw, data_ms, join="inner")

    if g_max_limit:
        data_bw = data_bw.sel(hkl=(data_bw["k"] < g_max_limit))
        data_ms = data_ms.sel(hkl=(data_ms["k"] < g_max_limit))

    print("Loading data to memory ...")
    data_bw = data_bw.compute()
    data_ms = data_ms.compute()

    R_z = bragg_r_z(data_bw, data_ms, tolerance)

    energies = sorted(float(e) for e in R_z["Energy"].data)

    print(f"{'energy (keV)':>14} {'R_z mean':>10} {'R_z max':>10}")
    for e in energies:
        curve = R_z.sel(Energy=e)
        print(f"{e / 1e3:14.0f} {float(curve.mean()):10.3f} {float(curve.max()):10.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(energies):
        curve = R_z.sel(Energy=e).sortby("z")
        color = cmap(i / max(len(energies) - 1, 1))
        ax.plot(curve["z"], curve, label=f"{e / 1e3:.0f} keV", color=color)

    ax.set_xlabel("thickness (Å)")
    ax.set_ylabel("$R_z$ (%)")
    ax.set_title("Bragg R factor (MS vs. BW) vs. thickness, per energy")
    ax.legend(title="energy")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(result_folder / f"R_z_vs_energy.{ext}", dpi=200)
    print(f"\nSaved plot to {result_folder / 'R_z_vs_energy.png'} (and matching .pdf)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_folder")
    parser.add_argument(
        "--g_max_limit", type=float, default=2.0,
        help="ignore reflections with k > this value (default: 2.0)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6,
        help="ignore reflections with peak BW intensity below this value (default: 1e-6)",
    )
    args = parser.parse_args()
    main(args.result_folder, args.g_max_limit, args.tolerance)
