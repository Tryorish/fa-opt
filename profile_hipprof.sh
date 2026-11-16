#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode=${1:-all}
length=${2:-4096}
pair=${3:-prefetch}
optimized=0
case "$mode" in
    trace|pmc|all) ;;
    *) echo "Usage: bash profile_hipprof.sh [trace|pmc|all] [length] [prefetch|matrix|tuned|optimized]" >&2; exit 2 ;;
esac
case "$pair" in
    prefetch) q_tile=64; kv_tile=32; full=0; variants=(before_qk after_softmax sdpa) ;;
    matrix) q_tile=128; kv_tile=32; full=0; variants=(transposed matrix sdpa) ;;
    tuned) q_tile=128; kv_tile=64; full=1; variants=(tuned sdpa) ;;
    optimized) q_tile=128; kv_tile=64; full=1; optimized=1; variants=(optimized sdpa) ;;
    *) echo "Comparison must be prefetch, matrix, tuned or optimized" >&2; exit 2 ;;
esac
command -v hipprof >/dev/null || { echo "hipprof is not on PATH" >&2; exit 1; }
python_bin=$(python -c 'import sys; print(sys.executable)')
mkdir -p build/hipprof
run_dir=$(mktemp -d "$PWD/build/hipprof/run-XXXXXX")
echo "Reports (including failures): $run_dir"
echo "Workload Python: $python_bin"
hipprof -h > "$run_dir/hipprof-help.txt" 2>&1

run_capture() {
    local kind=$1 warmup=$2 repeats=$3 status=0
    local prefix="$output/$kind"
    local -a runner=()
    case "$kind" in
        trace) runner=(hipprof --hip-trace --output-type 0 -o "$prefix") ;;
        pmc) runner=(hipprof --pmc --pmc-type 3 -o "$prefix") ;;
    esac
    # hipprof may return zero after an application/export failure. Check the
    # application's own status/log and exported rows, not just the launcher.
    "${runner[@]}" "$python_bin" -u "$PWD/profile_capture.py" "$prefix.workload.log" \
        --impl "$impl" --length "$length" --warmup "$warmup" --repeats "$repeats" \
        2>&1 | tee "$prefix.log" || status=$?
    "$python_bin" - "$prefix" "$status" "$((warmup + repeats))" <<'PY'
import csv
from pathlib import Path
import re
import sys

prefix, status, calls = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
log = Path(str(prefix) + ".workload.log")
app_status = Path(str(prefix) + ".workload.status")
print(f"Workload log: {log}", flush=True)
content = log.read_text(errors="replace") if log.exists() else "<workload log missing>\n"
print(content, end="", flush=True)
errors = []
if status:
    errors.append(f"launcher/pipeline exit status {status}")
if not app_status.exists() or app_status.read_text().strip() != "0":
    errors.append(f"workload exit status missing or nonzero: {app_status}")
if not re.search(rf"^Completed {calls} calls;", content, re.M):
    errors.append(f"workload did not report completing {calls} calls")
launcher_log = Path(str(prefix) + ".log").read_text(errors="replace")
if re.search(r"has not valid trace data|merge pmc data failed|\[ERROR\]", launcher_log + content, re.I):
    errors.append("collection/export error reported in launcher or workload log")
if prefix.name != "preflight":
    pattern = prefix.name + ("*hipkernel*.csv" if prefix.name == "trace" else ".csv")
    exports = list(prefix.parent.glob(pattern))
    has_rows = False
    for export in exports:
        with export.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
            rows = (row for row in csv.reader(stream) if any(cell.strip() for cell in row))
            next(rows, None)  # Header alone does not prove any kernels were collected.
            has_rows |= next(rows, None) is not None
    if not has_rows:
        errors.append(f"no kernel data rows in {prefix.parent / pattern}")
if errors:
    raise SystemExit("Collection stopped:\n  " + "\n  ".join(errors))
print(f"Checked {prefix.name}: workload completed; "
      + ("no profiler used." if prefix.name == "preflight" else "kernel CSV rows present."))
PY
}

for variant in "${variants[@]}"; do
    output="$run_dir/$variant"
    mkdir -p "$output"
    impl=custom
    late=0
    matrix=0
    if [[ "$variant" == after_softmax ]]; then late=1; fi
    if [[ "$variant" == matrix || "$pair" == tuned || "$pair" == optimized ]]; then matrix=1; fi
    if [[ "$variant" == sdpa ]]; then impl=sdpa; fi
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$q_tile -DFA_BW_KV_TILE=$kv_tile -DFA_BW_MATRIX_V=$matrix -DFA_BW_FULL_KV=$full -DFA_BW_ASYNC_K=$optimized -DFA_BW_ASYNC_V=$optimized -DFA_BW_DEFER_DENOM=$optimized -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=$late -Rpass-analysis=kernel-resource-usage -Werror=inline-asm"

    # Build and validate custom variants outside the profiler. Fail before collecting bad results.
    if [[ "$impl" == custom ]]; then
        "$python_bin" -m unittest test_flash_attn -v 2>&1 | tee "$output/check.log"
        if [[ "$pair" != prefetch && "$mode" == all ]]; then
            # Save emitted opcodes for the selected configuration, outside hipprof.
            "$python_bin" diagnose.py --tiles "$kv_tile" 2>&1 | tee "$output/assembly-summary.log"
        fi
    fi
    # Exercise the exact workload once before injecting the profiler.
    run_capture preflight 0 1
    if [[ "$mode" == trace || "$mode" == all ]]; then
        run_capture trace 5 20
    fi
    if [[ "$mode" == pmc || "$mode" == all ]]; then
        run_capture pmc 0 3
    fi
done
tar -czf "${run_dir}.tar.gz" -C "$(dirname -- "$run_dir")" "$(basename -- "$run_dir")"
echo "Reports: $run_dir"
echo "Archive: ${run_dir}.tar.gz"
if [[ "$pair" != prefetch && ( "$mode" == pmc || "$mode" == all ) ]]; then
    "$python_bin" dump_hipprof.py "$run_dir" \
        --columns '^(Index|KernelName|grd|wgr|lds|scr|arch_vgpr|accum_vgpr|sgpr|wave_size|SQ_.*)$'
fi
