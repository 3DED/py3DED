#!/usr/bin/env bash
# Batch-run every *.json config in a convergence_tests folder (e.g.
# examples/convergence_tests/Si_MS_boxsize, Si_BW_gmax, Si_BW_Sgmax, or one of
# Si_MS_slice_thickness's subfolders) through scripts/run_py3DED.py, one at a
# time, logging each run and continuing past individual failures instead of
# aborting the whole sweep. Non-recursive -- each of these folders (or
# subfolder, for Si_MS_slice_thickness) is its own independent sweep over one
# varying parameter, so run this once per folder rather than pointing it at a
# parent directory.
#
# Usage (from anywhere):
#   ./run_si_boxsize_sweep.sh [config_dir] [python-executable]
#
# config_dir defaults to examples/convergence_tests/Si_MS_boxsize relative to
# this script's own repo. The python executable defaults to "python" on PATH;
# pass e.g. a conda env's python explicitly if that's not the one with
# py3DED/abTEM installed:
#   ./run_si_boxsize_sweep.sh ../examples/convergence_tests/Si_BW_gmax /path/to/env/bin/python
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${1:-$REPO_ROOT/examples/convergence_tests/Si_MS_boxsize}"
SCRIPT="$REPO_ROOT/scripts/run_py3DED.py"
PYTHON="${2:-python}"

if [ ! -f "$SCRIPT" ]; then
    echo "Can't find run_py3DED.py at $SCRIPT -- adjust REPO_ROOT in this script if it lives elsewhere." >&2
    exit 1
fi

# Resolve to an absolute path *before* the cd below, so a relative config_dir
# (e.g. "../examples/convergence_tests/Si_BW_gmax") doesn't break LOG_DIR's
# meaning once the working directory changes.
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Config directory not found: $CONFIG_DIR" >&2
    exit 1
fi
CONFIG_DIR="$(cd "$CONFIG_DIR" && pwd)"

LOG_DIR="$CONFIG_DIR/sweep_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
echo -e "config\tstatus\tseconds" > "$SUMMARY"

# cif_file and output_folder in these configs are relative paths, resolved
# against the working directory run_py3DED.py is launched from -- not
# against the JSON file's own location -- so cd into the config directory
# first, matching how these configs are meant to be run.
cd "$CONFIG_DIR"

shopt -s nullglob
configs=(*.json)
shopt -u nullglob

if [ ${#configs[@]} -eq 0 ]; then
    echo "No .json configs found in $CONFIG_DIR" >&2
    exit 1
fi

echo "Found ${#configs[@]} configs in $CONFIG_DIR"
echo "Logs: $LOG_DIR"
echo

for config in "${configs[@]}"; do
    log_file="$LOG_DIR/${config%.json}.log"
    printf '=== %s ===\n' "$config"
    start=$(date +%s)

    # run_py3DED.py's output_base.mkdir() has no parents=True, so it fails if
    # the config's own output_folder (e.g. "results/Si_boxsize") doesn't
    # already exist -- create it up front rather than let every config fail
    # on that alone.
    output_folder=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['output_folder'])" "$config")
    mkdir -p "$output_folder"

    # </dev/null: none of these configs should hit an interactive prompt in
    # run_py3DED.py, but this makes sure a stray one fails fast instead of
    # hanging the whole sweep.
    if "$PYTHON" "$SCRIPT" "$config" > "$log_file" 2>&1 < /dev/null; then
        status="ok"
    else
        status="FAILED"
    fi

    elapsed=$(( $(date +%s) - start ))
    printf '%-30s %-8s %6ds  (log: %s)\n\n' "$config" "$status" "$elapsed" "$log_file"
    echo -e "${config}\t${status}\t${elapsed}" >> "$SUMMARY"
done

echo "Done."
echo
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
