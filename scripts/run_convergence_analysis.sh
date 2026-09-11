#!/usr/bin/env bash
# Run run_py3DED_hkl+Rint.py on every ms/bw zarr store under a results directory
# (one subfolder per box-size run, e.g. results/Si_boxsize/*_100x100x1000_16/),
# and run_py3DED_compare2.py on every subfolder that has BOTH ms and bw stores
# (most Si_MS_boxsize configs are MS-only, so compare2 will only apply to a
# few -- that's expected, not a bug). A folder with only supercell.zarr(.zip)
# -- the run hasn't reached, or crashed before, the multislice/Bloch-wave
# stage -- is reported separately and skipped, not counted as "MS-only".
#
# Handles a zipped ms.zarr.zip/bw.zarr.zip (or ms.zip/bw.zip) store, or a
# plain ms.zarr/bw.zarr directory store -- whichever is present.
#
# Usage:
#   ./run_convergence_analysis.sh [results_dir] [python-executable]
#
# results_dir defaults to examples/convergence_tests/Si_MS_boxsize/results/Si_boxsize
# relative to this script's own repo (i.e. scripts/../examples/...).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-$REPO_ROOT/examples/convergence_tests/Si_MS_boxsize/results/Si_boxsize}"
PYTHON="${2:-python}"
HKL_SCRIPT="$REPO_ROOT/scripts/run_py3DED_hkl+Rint.py"
COMPARE_SCRIPT="$REPO_ROOT/scripts/run_py3DED_compare2.py"

for f in "$HKL_SCRIPT" "$COMPARE_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "Can't find $f -- adjust REPO_ROOT in this script if it lives elsewhere." >&2
        exit 1
    fi
done

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory not found: $RESULTS_DIR" >&2
    exit 1
fi

# Print whichever of $1/<stem>.zarr (directory), $1/<stem>.zarr.zip, or
# $1/<stem>.zip exists first, for stem "ms" or "bw"; prints nothing if none do.
find_store() {
    local folder="$1" stem="$2"
    for candidate in "$folder/$stem.zarr" "$folder/$stem.zarr.zip" "$folder/$stem.zip"; do
        if [ -e "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
}

LOG_DIR="$RESULTS_DIR/analysis_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
echo -e "folder\thkl_ms\thkl_bw\tcompare2" > "$SUMMARY"

shopt -s nullglob
folders=("$RESULTS_DIR"/*/)
shopt -u nullglob

if [ ${#folders[@]} -eq 0 ]; then
    echo "No result subfolders found in $RESULTS_DIR" >&2
    exit 1
fi

echo "Found ${#folders[@]} entries in $RESULTS_DIR"
echo "Logs: $LOG_DIR"
echo

for folder in "${folders[@]}"; do
    folder="${folder%/}"
    name="$(basename "$folder")"
    [ "$name" = "analysis_logs" ] && continue

    echo "=== $name ==="
    hkl_ms_status="-"
    hkl_bw_status="-"
    compare_status="-"

    ms_store="$(find_store "$folder" ms)"
    bw_store="$(find_store "$folder" bw)"

    if [ -z "$ms_store" ] && [ -z "$bw_store" ]; then
        # Only supercell.zarr(.zip) so far -- the run hasn't reached (or
        # failed before) the multislice/Bloch-wave stage, nothing to
        # analyze yet. Distinct from the "MS-only config" case below.
        echo "  no ms/bw store yet (supercell only) -- skipping"
        echo -e "${name}\t-\t-\t-" >> "$SUMMARY"
        echo
        continue
    fi

    if [ -n "$ms_store" ]; then
        if "$PYTHON" "$HKL_SCRIPT" "$ms_store" > "$LOG_DIR/${name}_hkl_ms.log" 2>&1 < /dev/null; then
            hkl_ms_status="ok"
        else
            hkl_ms_status="FAILED"
        fi
        echo "  hkl+Rint ($(basename "$ms_store")): $hkl_ms_status"
    fi

    if [ -n "$bw_store" ]; then
        if "$PYTHON" "$HKL_SCRIPT" "$bw_store" > "$LOG_DIR/${name}_hkl_bw.log" 2>&1 < /dev/null; then
            hkl_bw_status="ok"
        else
            hkl_bw_status="FAILED"
        fi
        echo "  hkl+Rint ($(basename "$bw_store")): $hkl_bw_status"
    fi

    if [ -n "$ms_store" ] && [ -n "$bw_store" ]; then
        # Explicit two-file form (bw, then ms), not the single-folder form --
        # that one only ever looks for a literal "ms.zarr"/"bw.zarr" inside
        # the folder, so it won't find a .zip store.
        if "$PYTHON" "$COMPARE_SCRIPT" "$bw_store" "$ms_store" > "$LOG_DIR/${name}_compare2.log" 2>&1 < /dev/null; then
            compare_status="ok"
        else
            compare_status="FAILED"
        fi
        echo "  compare2 (ms+bw):   $compare_status"
    elif [ -n "$ms_store" ]; then
        echo "  compare2: skipped (no bw store -- MS-only config, or BW just hasn't finished yet)"
    else
        echo "  compare2: skipped (no ms store -- BW-only config, or MS just hasn't finished yet)"
    fi

    echo -e "${name}\t${hkl_ms_status}\t${hkl_bw_status}\t${compare_status}" >> "$SUMMARY"
    echo
done

echo "Done."
echo
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
