#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# A partially updated directory may still contain the D64-only benchmark.
# Check before compiling kernels or running the correctness suite.
bench_help=$(python bench.py --help)
if [[ "$bench_help" != *"--head-dim"* ]]; then
    echo "Outdated bench.py: missing --head-dim. Copy the current bench.py and compare_d128.sh to this directory." >&2
    exit 1
fi

mkdir -p build/d128
results=$(mktemp -d build/d128/run-XXXXXX)
echo "Results directory: $results"

# D128 doubles register payload and LDS bytes. Measure both tradeoffs;
# D64's winner is not assumed to remain the winner here.
for rows in 64 128; do
    for keys in 32 64; do
        echo "Testing B1 / H32 / D128 / Q${rows} / C${keys}"
        export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=$keys -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_ASYNC_V=1 -DFA_BW_ASYNC_K=1 -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_WAVE_RESCALE=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -DFA_BW_DEFER_DENOM=1 -DFA_BW_REDUCE4=0 -DFA_BW_DELAY_SCALE=1 -Rpass-analysis=kernel-resource-usage -Werror=inline-asm"
        python -m unittest test_head_dim -v
        python bench.py --lengths 4096 "$@" --batch 1 --heads 32 --head-dim 128 \
            --backend flash --json "$results/q${rows}_c${keys}.json"
    done
done

python - "$results" <<'PY'
import json
from pathlib import Path
import sys

print("\nD128 comparison: GPU graph timing by default; >1x means faster than Flash SDPA")
print("    Q     K      N      BW us    SDPA us   speedup")
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    data = json.loads(path.read_text())
    meta = data["metadata"]
    for row in data["results"]:
        print(f"{meta['q_tile']:5} {meta['kv_tile']:5} {row['N']:6} "
              f"{row['custom_us']:10.2f} {row['sdpa_us']:10.2f} {row['speedup']:8.3f}x")
PY
