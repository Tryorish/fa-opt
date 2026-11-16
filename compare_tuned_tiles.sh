#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for arg in "$@"; do
    if [[ "$arg" == -h || "$arg" == --help ]]; then
        echo "Compare Q64/128/256 x C32/64 with matrix V and full KV."
        echo "Benchmark options follow; backend=flash and JSON paths are fixed."
        exec python bench.py --help
    fi
done

mkdir -p build/tuned_tiles
run_dir=$(mktemp -d build/tuned_tiles/run-XXXXXX)
echo "Results directory: $run_dir"
results=()
for rows in 64 128 256; do
    for keys in 32 64; do
        echo "Testing Q${rows} / C${keys} / matrix V / full KV / unconditional rescale"
        export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=$keys -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
        python -m unittest test_flash_attn -v
        result="$run_dir/results_v11_q${rows}_c${keys}.json"
        python bench.py "$@" --backend flash --json "$result"
        results+=("$result")
    done
done

# Only summarize this completed run, using the compiled metadata for tile names.
python - "${results[@]}" <<'PY'
import json
from pathlib import Path
import sys

configs = {}
for filename in sys.argv[1:]:
    report = json.loads(Path(filename).read_text())
    meta = report["metadata"]
    label = f"Q{meta['q_tile']}/C{meta['kv_tile']}"
    configs[label] = {
        tuple(row[key] for key in ("B", "H", "N", "M")): row
        for row in report["results"]
    }

shapes = list(next(iter(configs.values())))
if len(configs) != 6 or any(set(rows) != set(shapes) for rows in configs.values()):
    raise SystemExit("Expected six tile configurations with matching benchmark shapes.")

print("\nCustom kernel latency (us); all configurations passed correctness checks.")
print(f"{'B':>3} {'H':>3} {'N':>6} {'M':>6} "
      + " ".join(f"{label:>11}" for label in configs))
for shape in shapes:
    print(" ".join(f"{value:>{width}}" for value, width in zip(shape, (3, 3, 6, 6)))
          + " " + " ".join(f"{rows[shape]['custom_us']:11.2f}" for rows in configs.values()))

print("\nLowest measured time per shape, with that run's paired Flash SDPA:")
print(f"{'B':>3} {'H':>3} {'N':>6} {'M':>6} {'tile':>11} "
      f"{'BW us':>10} {'SDPA us':>10} {'speedup':>9}")
for shape in shapes:
    label = min(configs, key=lambda name: configs[name][shape]["custom_us"])
    row = configs[label][shape]
    print(" ".join(f"{value:>{width}}" for value, width in zip(shape, (3, 3, 6, 6)))
          + f" {label:>11} {row['custom_us']:10.2f} {row['sdpa_us']:10.2f} "
          + f"{row['sdpa_us'] / row['custom_us']:8.3f}x")
print("Close timings (about 1%) need repeat runs; this sweep does not change defaults.")
PY

echo "Saved all six reports in $run_dir"
