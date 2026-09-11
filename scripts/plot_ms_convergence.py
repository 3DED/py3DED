#!/usr/bin/env python
"""Plot multislice convergence vs. box size for the Si_MS_boxsize sweep,
using two metrics that don't depend on symmetry grouping or a complete
rotation range (unlike R_int, which is contaminated by a limited single-axis
rotation series always sampling different symmetry-equivalent reflections
through different, non-comparable portions of their own rocking curve):

1. ASI (average scattered intensity), box size p:

     ASI_p = < sum_h I_h(z, alpha, p) >
           = (1 / (N_z * N_alpha)) * sum_z sum_alpha sum_h I_h(z, alpha, p)

   "Scattered" excludes the 000 reflection.

2. RL2 (relative L2 norm) against a reference box size p0 (the largest box
   size found, unless --reference is given):

     RL2_p = sqrt(
         mean_{z,alpha,h} [I_h(p, alpha, z) - I_h(p0, alpha, z)]^2
         / mean_{z,alpha,h} I_h(p0, alpha, z)^2
     )

   Summed over reflections h common to both p's and p0's datasets (also
   excluding 000), matched by Miller index rather than assumed array order.

Usage:
    python plot_ms_convergence.py <results_dir> [--reference BOX_SIZE]

results_dir is the directory containing one subfolder per run (each with its
own settings.json and ms.zarr/ms.zarr.zip/ms.zip).
"""
import argparse
import json
from pathlib import Path

import abtem
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_ms_store(folder: Path):
    for name in ("ms.zarr", "ms.zarr.zip", "ms.zip"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def load_run(folder: Path):
    """Return (box_size, miller_indices, intensities, folder_name) for one
    result folder's ms store, or None if it can't be loaded.

    miller_indices : (N_h, 3) int array
    intensities    : (N_orientation, N_z, N_h) float array
    """
    settings_file = folder / "settings.json"
    if not settings_file.exists():
        return None
    try:
        settings = json.loads(settings_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    box_size = settings.get("box_size_x")
    if box_size is None:
        return None

    ms_store = find_ms_store(folder)
    if ms_store is None:
        return None

    try:
        dsz = abtem.from_zarr(str(ms_store))
        miller_indices = np.asarray(dsz.miller_indices)
        intensities = np.asarray(dsz.intensities.compute())
    except Exception as exc:
        print(f"Warning: failed to load {ms_store}: {exc}")
        return None

    return box_size, miller_indices, intensities, folder.name


def exclude_000(miller_indices, intensities):
    keep = ~np.all(miller_indices == 0, axis=1)
    return miller_indices[keep], intensities[:, :, keep]


def compute_asi(intensities):
    # sum over h, then mean over (orientation, z)
    return float(intensities.sum(axis=-1).mean())


def align_by_hkl(mi_a, intensities_a, mi_b, intensities_b):
    """Restrict both intensity arrays to hkl present in both datasets, in a
    common order. Returns (I_a_aligned, I_b_aligned, n_common)."""
    map_a = {tuple(h): i for i, h in enumerate(mi_a.tolist())}
    map_b = {tuple(h): i for i, h in enumerate(mi_b.tolist())}
    common = sorted(set(map_a) & set(map_b))
    idx_a = [map_a[h] for h in common]
    idx_b = [map_b[h] for h in common]
    return intensities_a[:, :, idx_a], intensities_b[:, :, idx_b], len(common)


def compute_rl2(intensities_p, intensities_ref):
    diff2_mean = np.mean((intensities_p - intensities_ref) ** 2)
    ref2_mean = np.mean(intensities_ref ** 2)
    return float(np.sqrt(diff2_mean / ref2_mean))


def main(results_dir, reference_box_size=None):
    results_dir = Path(results_dir)

    candidates = [
        f for f in sorted(results_dir.iterdir())
        if f.is_dir() and f.name != "analysis_logs"
    ]
    runs = []
    for i, folder in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] loading {folder.name} ...", end=" ", flush=True)
        loaded = load_run(folder)
        if loaded is None:
            print("skipped (no settings.json/ms store)", flush=True)
            continue
        print(f"ok (N_h={loaded[1].shape[0]})", flush=True)
        runs.append(loaded)

    if not runs:
        print(f"No usable ms stores found under {results_dir}")
        return

    by_box_size = {}
    for box_size, mi, intensities, name in runs:
        by_box_size.setdefault(box_size, []).append((mi, intensities, name))
    for box_size, entries in by_box_size.items():
        if len(entries) > 1:
            names = ", ".join(e[2] for e in entries)
            print(f"Warning: {len(entries)} folders share box_size={box_size}"
                  f": {names} -- using the first (alphabetically-earliest "
                  "folder name).")

    box_sizes = sorted(by_box_size)
    data = {bs: by_box_size[bs][0] for bs in box_sizes}  # bs -> (mi, I, name)
    data = {
        bs: (*exclude_000(mi, intensities), name)
        for bs, (mi, intensities, name) in data.items()
    }

    if reference_box_size is None:
        reference_box_size = max(box_sizes)
    elif reference_box_size not in data:
        raise ValueError(
            f"--reference {reference_box_size} not found among box sizes {box_sizes}"
        )
    mi_ref, I_ref, name_ref = data[reference_box_size]
    print(f"Using box_size={reference_box_size} ({name_ref}) as reference "
          "for both ASI and RL2\n")

    # ASI is normalized by its own value at the reference box size, so it
    # converges to 1.0 there by construction -- matching how RL2 converges
    # to 0 there.
    print("Computing ASI ...", flush=True)
    asi_raw = {}
    for bs, (mi, intensities, name) in data.items():
        asi_raw[bs] = compute_asi(intensities)
        print(f"  box_size={bs:.2f}: ASI_raw={asi_raw[bs]:.6e}", flush=True)
    asi = {bs: value / asi_raw[reference_box_size] for bs, value in asi_raw.items()}

    print("Computing RL2 ...", flush=True)
    rl2 = {}
    for bs, (mi, intensities, name) in data.items():
        if bs == reference_box_size:
            rl2[bs] = 0.0
            continue
        I_aligned, I_ref_aligned, n_common = align_by_hkl(mi, intensities, mi_ref, I_ref)
        if n_common == 0:
            print(f"Warning: no common reflections between box_size={bs} "
                  "and the reference; skipping RL2 for this box size.")
            continue
        rl2[bs] = compute_rl2(I_aligned, I_ref_aligned)
        print(f"  box_size={bs:.2f}: n_common={n_common} RL2={rl2[bs]:.4f}", flush=True)

    print(f"{'box_size (Å)':>13} {'ASI (raw)':>14} {'ASI / ref':>10} {'RL2':>10}  folder")
    for bs in box_sizes:
        rl2_str = f"{rl2[bs]:.4f}" if bs in rl2 else "n/a"
        print(f"{bs:13.2f} {asi_raw[bs]:14.6e} {asi[bs]:10.4f} {rl2_str:>10}  {data[bs][2]}")

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(box_sizes, [asi[bs] for bs in box_sizes], "o-", color="#2456a6")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label=f"reference ({reference_box_size:.2f} Å)")
    ax.set_xlabel("box size (Å)")
    ax.set_ylabel(f"ASI / ASI({reference_box_size:.2f} Å)")
    ax.set_title("Average scattered intensity vs. box size\n(normalized to reference box size)")
    ax.set_ylim(0.9,1.1)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    rl2_box_sizes = [bs for bs in box_sizes if bs in rl2 and rl2[bs] > 0]
    ax.plot(rl2_box_sizes, [rl2[bs] for bs in rl2_box_sizes], "o-", color="#b8571b")
    ax.axvline(reference_box_size, color="gray", linestyle="--", linewidth=1,
               label=f"reference ({reference_box_size:.2f} Å, RL2=0, off this log scale)")
    ax.set_yscale("log")
    ax.set_xlabel("box size (Å)")
    ax.set_ylabel("RL2 (log scale)")
    ax.set_title(f"RL2 vs. box size (reference = {reference_box_size:.2f} Å)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(results_dir / f"ms_convergence_asi_rl2.{ext}", dpi=200)
    print(f"\nSaved plot to {results_dir / 'ms_convergence_asi_rl2.png'} "
          f"(and matching .pdf)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument(
        "--reference", type=float, default=None,
        help="box size (Å) to use as the RL2 reference; defaults to the "
             "largest box size found",
    )
    args = parser.parse_args()
    main(args.results_dir, args.reference)
