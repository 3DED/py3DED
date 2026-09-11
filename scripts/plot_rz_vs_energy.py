#!/usr/bin/env python
"""Compare MS vs. BW for a single py3DED run whose config.energy was a list
(abTEM's energy ensemble), plotting the Bragg R factor of integrated
intensities as a function of thickness -- R_z, the same metric as
run_py3DED_compare2.py's R_z.pdf -- with one curve per energy on a shared
plot, instead of one plot per energy.

Usage:
    python plot_rz_vs_energy.py <result_folder> [--g_max_limit VALUE] [--tolerance VALUE]

result_folder must contain ms.zarr and bw.zarr (or their .zip forms) written
from one run with an energy ensemble -- both must carry an "Energy"
dimension, which abtem.from_zarr(...).to_data_array() exposes automatically.
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


def main(result_folder, g_max_limit=2.0, tolerance=1e-6):
    result_folder = Path(result_folder)

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

    if tolerance:
        # strongest intensity along alpha, averaged by z, using BW as reference
        max_bw = data_bw.mean(dim="z").max(dim="x_rotation").max(dim="Energy")
        hkl_mask = max_bw.data > tolerance
        data_bw = data_bw.sel(hkl=hkl_mask)
        data_ms = data_ms.sel(hkl=hkl_mask)

    int_bw = data_bw.sum(dim="x_rotation")
    int_ms = data_ms.sum(dim="x_rotation")

    # Bragg R factor based on intensities, one R factor per (energy, thickness)
    delta = np.abs(int_bw - int_ms)
    R_z = 100 * delta.drop_sel(hkl="0 0 0").sum(dim="hkl") / int_bw.drop_sel(
        hkl="0 0 0"
    ).sum(dim="hkl")

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
