#!/usr/bin/env python
"""Plot convergence vs. a swept parameter, for any convergence_tests folder
(Si_MS_boxsize, Si_BW_gmax, Si_BW_Sgmax, or one of Si_MS_slice_thickness's
subfolders), using two metrics that don't depend on symmetry grouping or a
complete rotation range (unlike R_int, which is contaminated by a limited
single-axis rotation series always sampling different symmetry-equivalent
reflections through different, non-comparable portions of their own rocking
curve):

1. ASI (average scattered intensity), parameter value p, normalized to a
   reference value p0 (the largest value found, unless --reference is
   given):

     ASI_p = < sum_h I_h(z, alpha, p) > / < sum_h I_h(z, alpha, p0) >
           = [(1 / (N_z * N_alpha)) * sum_z sum_alpha sum_h I_h(z, alpha, p)]
             / [same, evaluated at p0]

   "Scattered" excludes the 000 reflection. Converges to 1.0 at p0 by
   construction.

2. RL2 (relative L2 norm) against the same reference value p0:

     RL2_p = sqrt(
         mean_{z,alpha,h} [I_h(p, alpha, z) - I_h(p0, alpha, z)]^2
         / mean_{z,alpha,h} I_h(p0, alpha, z)^2
     )

   Summed over reflections h common to both p's and p0's datasets (also
   excluding 000), matched by Miller index rather than assumed array order.
   Converges to 0.0 at p0 by construction.

Both metrics are computed from whichever of ms.zarr/bw.zarr a given result
folder actually has (preferring ms if both are present) -- the MS-only
Si_MS_boxsize/Si_MS_slice_thickness sweeps and the BW-only Si_BW_gmax/
Si_BW_Sgmax sweeps all work the same way here.

Usage:
    python plot_ms_convergence.py <results_dir> [--parameter KEY] [--reference VALUE]

results_dir is the directory containing one subfolder per run (each with its
own settings.json and ms/bw store, in either zip or plain-directory form).
--parameter is the settings.json key that was swept (default: box_size_x;
use g_max, sg_max, or slice_thickness for the other convergence_tests
folders).
"""
import argparse
import json
from pathlib import Path

import abtem
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_store(folder: Path):
    """Return (path, kind) for whichever of ms/bw this folder has, preferring
    ms if both are present ("kind" is "ms" or "bw"), or (None, None)."""
    for stem in ("ms", "bw"):
        for name in (f"{stem}.zarr", f"{stem}.zarr.zip", f"{stem}.zip"):
            candidate = folder / name
            if candidate.exists():
                return candidate, stem
    return None, None


def load_run(folder: Path, parameter: str):
    """Return (param_value, kind, miller_indices, intensities, folder_name)
    for one result folder's ms/bw store, or None if it can't be loaded.

    kind           : "ms" or "bw", whichever store was actually used
    miller_indices : (N_h, 3) int array, eager (this is tiny -- just labels)
    intensities    : (N_orientation, N_z, N_h) dask array, kept LAZY -- the
                     caller decides what to reduce/slice before ever calling
                     .compute(), so dask can do it chunk-by-chunk without
                     materializing the full array in memory regardless of how
                     large N_h turns out to be.
    """
    settings_file = folder / "settings.json"
    if not settings_file.exists():
        return None
    try:
        settings = json.loads(settings_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    param_value = settings.get(parameter)
    if param_value is None:
        return None

    store, kind = find_store(folder)
    if store is None:
        return None

    try:
        dsz = abtem.from_zarr(str(store))
        miller_indices = np.asarray(dsz.miller_indices)
        intensities = dsz.intensities  # stays a dask array -- no .compute()
    except Exception as exc:
        print(f"Warning: failed to load {store}: {exc}")
        return None

    return float(param_value), kind, miller_indices, intensities, folder.name


def _to_scalar(x):
    """.compute() if x is dask-backed (has that method), else pass through;
    either way return a plain float. This is the only place a full reduction
    or an already-sliced-down subset actually gets materialized."""
    if hasattr(x, "compute"):
        x = x.compute()
    return float(np.asarray(x))


def exclude_000(miller_indices, intensities):
    # boolean mask is on the (tiny, eager) miller_indices array; slicing the
    # (lazy) intensities array by it stays lazy.
    keep = ~np.all(miller_indices == 0, axis=1)
    return miller_indices[keep], intensities[:, :, keep]


def compute_asi(intensities):
    # sum over h, then mean over (orientation, z) -- stays a dask graph
    # until this single final reduction is computed, so the full
    # (orientation, z, hkl) array is never materialized at once.
    return _to_scalar(intensities.sum(axis=-1).mean())


def align_by_hkl(mi_a, intensities_a, mi_b, intensities_b):
    """Restrict both (lazy) intensity arrays to hkl present in both
    datasets, in a common order, and materialize only that (typically much
    smaller than either full dataset) common subset.
    Returns (I_a_aligned, I_b_aligned, n_common) as plain numpy arrays."""
    map_a = {tuple(h): i for i, h in enumerate(mi_a.tolist())}
    map_b = {tuple(h): i for i, h in enumerate(mi_b.tolist())}
    common = sorted(set(map_a) & set(map_b))
    idx_a = [map_a[h] for h in common]
    idx_b = [map_b[h] for h in common]
    sliced_a = intensities_a[:, :, idx_a]
    sliced_b = intensities_b[:, :, idx_b]
    if hasattr(sliced_a, "compute"):
        sliced_a = sliced_a.compute()
    if hasattr(sliced_b, "compute"):
        sliced_b = sliced_b.compute()
    return np.asarray(sliced_a), np.asarray(sliced_b), len(common)


def compute_rl2(intensities_p, intensities_ref):
    diff2_mean = np.mean((intensities_p - intensities_ref) ** 2)
    ref2_mean = np.mean(intensities_ref ** 2)
    return float(np.sqrt(diff2_mean / ref2_mean))


def main(results_dir, parameter="box_size_x", reference_value=None):
    results_dir = Path(results_dir)

    candidates = [
        f for f in sorted(results_dir.iterdir())
        if f.is_dir() and f.name != "analysis_logs"
    ]
    runs = []
    for i, folder in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] loading {folder.name} ...", end=" ", flush=True)
        loaded = load_run(folder, parameter)
        if loaded is None:
            print("skipped (no settings.json/ms/bw store, or "
                  f"no '{parameter}' key)", flush=True)
            continue
        print(f"ok ({loaded[1]}, N_h={loaded[2].shape[0]})", flush=True)
        runs.append(loaded)

    if not runs:
        print(f"No usable ms/bw stores found under {results_dir}")
        return

    kinds = {kind for _, kind, _, _, _ in runs}
    if len(kinds) > 1:
        print(f"Warning: mixing ms and bw stores across runs ({kinds}) -- "
              "each folder used whichever it had, preferring ms.")

    by_value = {}
    for param_value, kind, mi, intensities, name in runs:
        by_value.setdefault(param_value, []).append((kind, mi, intensities, name))
    for param_value, entries in by_value.items():
        if len(entries) > 1:
            names = ", ".join(e[3] for e in entries)
            print(f"Warning: {len(entries)} folders share {parameter}={param_value}"
                  f": {names} -- using the first (alphabetically-earliest "
                  "folder name).")

    values = sorted(by_value)
    data = {v: by_value[v][0] for v in values}  # v -> (kind, mi, I, name)
    data = {
        v: (kind, *exclude_000(mi, intensities), name)
        for v, (kind, mi, intensities, name) in data.items()
    }

    if reference_value is None:
        reference_value = max(values)
    elif reference_value not in data:
        raise ValueError(
            f"--reference {reference_value} not found among {parameter} values {values}"
        )
    _, mi_ref, I_ref, name_ref = data[reference_value]
    print(f"Using {parameter}={reference_value} ({name_ref}) as reference "
          "for both ASI and RL2\n")

    # ASI is normalized by its own value at the reference, so it converges
    # to 1.0 there by construction -- matching how RL2 converges to 0 there.
    print("Computing ASI ...", flush=True)
    asi_raw = {}
    for v, (kind, mi, intensities, name) in data.items():
        asi_raw[v] = compute_asi(intensities)
        print(f"  {parameter}={v:.4g}: ASI_raw={asi_raw[v]:.6e}", flush=True)
    asi = {v: value / asi_raw[reference_value] for v, value in asi_raw.items()}

    print("Computing RL2 ...", flush=True)
    rl2 = {}
    for v, (kind, mi, intensities, name) in data.items():
        if v == reference_value:
            rl2[v] = 0.0
            continue
        I_aligned, I_ref_aligned, n_common = align_by_hkl(mi, intensities, mi_ref, I_ref)
        if n_common == 0:
            print(f"Warning: no common reflections between {parameter}={v} "
                  "and the reference; skipping RL2 for this value.")
            continue
        rl2[v] = compute_rl2(I_aligned, I_ref_aligned)
        print(f"  {parameter}={v:.4g}: n_common={n_common} RL2={rl2[v]:.4f}", flush=True)

    print(f"{parameter:>15} {'ASI (raw)':>14} {'ASI / ref':>10} {'RL2':>10}  folder")
    for v in values:
        rl2_str = f"{rl2[v]:.4f}" if v in rl2 else "n/a"
        print(f"{v:15.4g} {asi_raw[v]:14.6e} {asi[v]:10.4f} {rl2_str:>10}  {data[v][3]}")

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(values, [asi[v] for v in values], "o-", color="#2456a6")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label=f"reference ({reference_value:.4g})")
    ax.set_xlabel(parameter)
    ax.set_ylabel(f"ASI / ASI({reference_value:.4g})")
    ax.set_title(f"Average scattered intensity vs. {parameter}\n(normalized to reference value)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    rl2_values = [v for v in values if v in rl2 and rl2[v] > 0]
    ax.plot(rl2_values, [rl2[v] for v in rl2_values], "o-", color="#b8571b")
    ax.axvline(reference_value, color="gray", linestyle="--", linewidth=1,
               label=f"reference ({reference_value:.4g}, RL2=0, off this log scale)")
    if rl2_values:
        ax.set_yscale("log")
    else:
        print("Warning: every non-reference RL2 is exactly 0 (or there's "
              "only the reference point) -- leaving the RL2 axis linear "
              "since log-scale needs at least one positive value.")
    ax.set_xlabel(parameter)
    ax.set_ylabel("RL2 (log scale)")
    ax.set_title(f"RL2 vs. {parameter} (reference = {reference_value:.4g})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(results_dir / f"convergence_{parameter}.{ext}", dpi=200)
    print(f"\nSaved plot to {results_dir / f'convergence_{parameter}.png'} "
          f"(and matching .pdf)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument(
        "--parameter", default="box_size_x",
        help="settings.json key that was swept (default: box_size_x; use "
             "g_max, sg_max, or slice_thickness for the other "
             "convergence_tests folders)",
    )
    parser.add_argument(
        "--reference", type=float, default=None,
        help="parameter value to use as the reference; defaults to the "
             "largest value found",
    )
    args = parser.parse_args()
    main(args.results_dir, args.parameter, args.reference)
