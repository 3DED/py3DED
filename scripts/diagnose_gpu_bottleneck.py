#!/usr/bin/env python
"""Isolate one candidate cause of poor GPU utilization at a time, on a small
fast config, with GPU stats logged automatically in the background instead
of needing to watch nvidia-smi live.

Runs a single small multislice calculation (same code paths as a real
py3DED run -- lazy potential, WindowedPixelatedDetector, optionally an
energy ensemble) while an `nvidia-smi --query-gpu=... -l 1` process logs to
a CSV file in the background. Prints timestamped checkpoints for graph
construction vs. the actual `.compute()`, then parses the GPU log against
those checkpoints and prints a summary: mean/min/max utilization overall and
per phase, idle-period count/total/longest streak, and a coarse ASCII
sparkline -- so the answer is a report, not something to eyeball live.

Only change ONE of --num-workers / --chunk-size-gpu / --to-cpu / --no-to-cpu
from its default per run, so each run actually isolates that one variable
instead of repeating the earlier confounded test.

Usage examples (run each while nothing else is on the GPU):
    # baseline: current defaults
    python diagnose_gpu_bottleneck.py --cif examples/Si_kinetic_energy/Si_CollCode51688.cif --label baseline

    # isolate to_cpu
    python diagnose_gpu_bottleneck.py --cif examples/Si_kinetic_energy/Si_CollCode51688.cif --no-to-cpu --label no_to_cpu

    # isolate num_workers / chunk-size-gpu reverted to abTEM defaults
    python diagnose_gpu_bottleneck.py --cif examples/Si_kinetic_energy/Si_CollCode51688.cif --num-workers 1 --chunk-size-gpu "512 MB" --label reverted_workers_chunks

Each run writes <label>_gpu_log.csv (raw nvidia-smi samples) and
<label>_report.txt (the summary) into --out-dir (default: current directory).
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def build_small_config(cif_file, box_size, thickness, rotation_steps, sampling,
                        slice_thickness, energies, device, num_workers, to_cpu,
                        exit_planes=10):
    import ase.io
    from abtem.atoms import cut_disk

    from py3DED import simulate

    atoms = ase.io.read(cif_file)
    box = (box_size, box_size, thickness)
    rotation_min, rotation_max = 0.0, 45.0

    disk = cut_disk(atoms, box, rotation_axis=0.0, rotation_range=(rotation_min, rotation_max))

    config = simulate.Config()
    config.unit_cell = atoms
    config.box_size_x, config.box_size_y = box_size, box_size
    config.rotation_axis = 0.0
    config.rotation_min = rotation_min
    config.rotation_max = rotation_max
    config.rotation_steps = rotation_steps
    config.slice_thickness = slice_thickness
    config.exit_planes = exit_planes
    config.sampling = sampling
    config.g_max_store = 2.0
    config.sg_max = 1.0
    config.g_max = 4.0
    config.integration_radius = 0.02
    config.device = device
    config.num_workers = num_workers
    config.precision = "float32"
    config.window_func = "hann"
    config.energy = energies if len(energies) > 1 else energies[0]
    config.to_cpu = to_cpu

    return config, disk, atoms


def start_gpu_logger(log_path: Path, interval_s: float = 0.2):
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "nvidia-smi not found on PATH -- this script must run on the GPU "
            "machine (europa), not here."
        )
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            f"-lms={int(interval_s * 1000)}",
        ],
        stdout=log_file,
        stderr=subprocess.DEVNULL,
    )
    return proc, log_file


def stop_gpu_logger(proc, log_file):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()


def parse_gpu_log(log_path: Path):
    """Return (timestamps as time.time()-comparable floats, gpu_util, mem_used_mb)."""
    import datetime

    rows = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            ts_str, util_gpu, util_mem, mem_used, mem_total = parts[:5]
            try:
                # nvidia-smi timestamp format: "2026/09/14 12:34:56.789"
                ts = datetime.datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f").timestamp()
                rows.append((ts, float(util_gpu), float(mem_used)))
            except ValueError:
                continue
    if not rows:
        return np.array([]), np.array([]), np.array([])
    ts, util, mem = zip(*rows)
    return np.array(ts), np.array(util), np.array(mem)


def sparkline(values, width=60):
    if len(values) == 0:
        return "(no data)"
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = 0.0, 100.0
    # downsample to `width` points by averaging chunks
    n = len(values)
    if n <= width:
        sampled = values
    else:
        edges = np.linspace(0, n, width + 1).astype(int)
        sampled = [values[edges[i]:edges[i + 1]].mean() if edges[i] < edges[i + 1] else values[edges[i]]
                   for i in range(width)]
    out = []
    for v in sampled:
        idx = int(np.clip((v - lo) / (hi - lo) * (len(blocks) - 1), 0, len(blocks) - 1))
        out.append(blocks[idx])
    return "".join(out)


def summarize(ts, util, mem, checkpoints, idle_threshold=10.0):
    lines = []
    if len(ts) == 0:
        lines.append("No GPU samples were captured -- the run may have finished "
                      "faster than the logging interval, or nvidia-smi produced "
                      "no parseable output.")
        return "\n".join(lines)

    t0 = ts[0]
    rel = ts - t0

    lines.append(f"Samples: {len(ts)} over {rel[-1]:.2f}s (~{len(ts)/max(rel[-1],1e-9):.1f} Hz)")
    lines.append(f"GPU util:  mean={util.mean():.1f}%  min={util.min():.1f}%  max={util.max():.1f}%  std={util.std():.1f}%")
    lines.append(f"GPU mem:   mean={mem.mean():.0f}MB  max={mem.max():.0f}MB")
    lines.append("")
    lines.append("Utilization over time (each block ~{:.2f}s):".format(rel[-1] / 60 if len(rel) else 0))
    lines.append(sparkline(util))
    lines.append("")

    # idle-period analysis
    is_idle = util < idle_threshold
    idle_runs = []
    run_start = None
    for i, flag in enumerate(is_idle):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            idle_runs.append((rel[run_start], rel[i - 1] - rel[run_start]))
            run_start = None
    if run_start is not None:
        idle_runs.append((rel[run_start], rel[-1] - rel[run_start]))

    total_idle = sum(d for _, d in idle_runs)
    lines.append(f"Idle periods (<{idle_threshold:.0f}% util): {len(idle_runs)}, "
                 f"total {total_idle:.2f}s ({100*total_idle/max(rel[-1],1e-9):.1f}% of run)")
    if idle_runs:
        longest = max(idle_runs, key=lambda r: r[1])
        lines.append(f"Longest idle streak: {longest[1]:.2f}s, starting at t={longest[0]:.2f}s")
        if len(idle_runs) <= 20:
            lines.append("All idle periods (start_s, duration_s):")
            for start, dur in idle_runs:
                lines.append(f"  t={start:6.2f}s  dur={dur:5.2f}s")
        else:
            lines.append(f"({len(idle_runs)} idle periods -- showing first 10)")
            for start, dur in idle_runs[:10]:
                lines.append(f"  t={start:6.2f}s  dur={dur:5.2f}s")

    lines.append("")
    lines.append("Checkpoints (relative to first GPU sample):")
    for name, cp_time in checkpoints:
        lines.append(f"  {name:30s} t={cp_time - t0:7.2f}s")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif", required=True, help="path to a CIF file (e.g. the Si_kinetic_energy one)")
    parser.add_argument("--box-size", type=float, default=40.0, help="box_size_x = box_size_y [A] (default: 40, small/fast)")
    parser.add_argument("--thickness", type=float, default=200.0, help="thickness [A] (default: 200)")
    parser.add_argument("--rotation-steps", type=int, default=8)
    parser.add_argument("--sampling", type=float, default=0.02)
    parser.add_argument("--slice-thickness", type=float, default=0.4)
    parser.add_argument("--exit-planes", type=int, default=10,
                         help="detector/output checkpoint every this many slices (default: 10)")
    parser.add_argument("--potential-chunk-size", default="auto",
                         help="abtem's potential.slice-chunk-size -- number of "
                              "potential slices built per memory-budgeted chunk "
                              "during generate_chunked_slices (default: 'auto', "
                              "sized from live free VRAM). Pass a large integer "
                              "(>= total slice count) to force a single chunk "
                              "and isolate this from exit_planes checkpointing.")
    parser.add_argument("--energies", type=float, nargs="+", default=[200e3],
                         help="one or more energies [eV] -- pass several for an energy ensemble")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size-gpu", default="512 MB")
    # Defaults to False to match simulate.Config's own default (root-caused
    # and fixed via this same script -- see commit 36a519d): pass --to-cpu
    # explicitly to reproduce the old, slower behavior for comparison.
    to_cpu_group = parser.add_mutually_exclusive_group()
    to_cpu_group.add_argument("--to-cpu", dest="to_cpu", action="store_true", default=False)
    to_cpu_group.add_argument("--no-to-cpu", dest="to_cpu", action="store_false")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--gpu-poll-interval", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.label}_gpu_log.csv"
    report_path = out_dir / f"{args.label}_report.txt"

    print(f"[{args.label}] settings: num_workers={args.num_workers} "
          f"chunk-size-gpu={args.chunk_size_gpu} to_cpu={args.to_cpu} "
          f"energies={args.energies}", flush=True)

    import abtem
    abtem.config.set({
        "device": args.device,
        "precision": "float32",
        "fft": "fftw",
        "diagnostics.task_progress": True,
        "diagnostics.progress_bar": "tqdm",
        "grid.round-to-fast-fft": True,
        "dask.chunk-size-gpu": args.chunk_size_gpu,
        "potential.slice-chunk-size": args.potential_chunk_size,
    })

    # Log whatever chunk size "auto" actually resolves to at real-run
    # conditions (live free VRAM, probe batch already resident) instead of
    # guessing from idle-period counts -- estimate_potential_chunk_size can
    # be called more than once per compute(), so log every call.
    from abtem.core import chunks as _abtem_chunks
    _orig_estimate_potential_chunk_size = _abtem_chunks.estimate_potential_chunk_size

    def _logging_estimate_potential_chunk_size(gpts, device="cpu", dtype=None):
        n = _orig_estimate_potential_chunk_size(gpts, device, dtype)
        print(f"[{args.label}] potential chunk size resolved: {n} slices "
              f"(gpts={gpts}, device={device})", flush=True)
        return n

    _abtem_chunks.estimate_potential_chunk_size = _logging_estimate_potential_chunk_size

    checkpoints = []
    proc, log_file = start_gpu_logger(log_path, args.gpu_poll_interval)
    try:
        checkpoints.append(("gpu_logger_started", time.time()))

        t0 = time.time()
        config, disk, atoms = build_small_config(
            args.cif, args.box_size, args.thickness, args.rotation_steps,
            args.sampling, args.slice_thickness, args.energies, args.device,
            args.num_workers, args.to_cpu, args.exit_planes,
        )
        checkpoints.append(("config_and_disk_built", time.time()))
        print(f"[{args.label}] disk atoms: {len(disk)}  (built in {time.time()-t0:.2f}s)", flush=True)

        from abtem.rotation_series import rotated_atoms_ensemble
        from py3DED import simulate
        import py3DED.simulate as simmod

        t1 = time.time()
        angles_deg = np.rad2deg(simmod.get_x_angles(config))
        atoms_ensemble = rotated_atoms_ensemble(disk, angles_deg, rotation_axis=config.rotation_axis)
        simmod.get_atoms_ensemble = lambda cfg: (atoms_ensemble, atoms)
        checkpoints.append(("atoms_ensemble_built_lazy", time.time()))
        print(f"[{args.label}] atoms_ensemble (lazy) built in {time.time()-t1:.3f}s", flush=True)

        t2 = time.time()
        lazy_result = simmod.setup_multislice(config)
        checkpoints.append(("multislice_graph_built_lazy", time.time()))
        print(f"[{args.label}] multislice graph (lazy) built in {time.time()-t2:.3f}s", flush=True)

        t3 = time.time()
        result = lazy_result.compute()
        checkpoints.append(("compute_finished", time.time()))
        print(f"[{args.label}] .compute() finished in {time.time()-t3:.2f}s "
              f"(total wall time {time.time()-t0:.2f}s)", flush=True)
        print(f"[{args.label}] result shape: {result.array.shape}", flush=True)

    finally:
        # give nvidia-smi one more poll interval to capture the tail, then stop
        time.sleep(args.gpu_poll_interval * 2)
        stop_gpu_logger(proc, log_file)

    ts, util, mem = parse_gpu_log(log_path)
    report = summarize(ts, util, mem, checkpoints)
    report_header = (
        f"=== {args.label} ===\n"
        f"num_workers={args.num_workers} chunk-size-gpu={args.chunk_size_gpu} "
        f"to_cpu={args.to_cpu} energies={args.energies}\n"
        f"box={args.box_size}x{args.box_size}x{args.thickness} "
        f"rotation_steps={args.rotation_steps} sampling={args.sampling} "
        f"exit_planes={args.exit_planes} "
        f"potential_chunk_size={args.potential_chunk_size}\n\n"
    )
    full_report = report_header + report
    report_path.write_text(full_report)
    print()
    print(full_report)
    print(f"\nRaw GPU log: {log_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
