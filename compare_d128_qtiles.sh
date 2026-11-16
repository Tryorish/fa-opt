#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

bench_help=$(python bench.py --help)
if [[ "$bench_help" != *"--head-dim"* ]]; then
    echo "Outdated bench.py: copy the current fa_bw directory before running D128." >&2
    exit 1
fi
python -c 'from test_flash_attn import check_probe_sources; check_probe_sources()'

mkdir -p build/d128_qtiles
results=$(mktemp -d build/d128_qtiles/run-XXXXXX)
echo "Results directory: $results"

# Q128/C32 is the measured D128 baseline. Larger Q reuses K/V across more
# rows, but may spill the larger D128 Q/output register payload.
for rows in 128 192 256; do
    echo "Testing B1 / H32 / D128 / Q${rows} / C32"
    export FA_BW_CXXFLAGS="-DFA_BW_HEAD_DIM=128 -DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_ASYNC_V=1 -DFA_BW_ASYNC_K=1 -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_WAVE_RESCALE=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -DFA_BW_DEFER_DENOM=1 -DFA_BW_REDUCE4=0 -DFA_BW_DELAY_SCALE=1 -Werror=inline-asm"
    # Force a fresh diagnostic build so cached libraries cannot hide the
    # resource report. HEAD_DIM is explicit here as diagnose.py builds directly.
    python diagnose.py --tiles 32 --full-only | tee "$results/q${rows}_resources.log"
    python -m unittest test_head_dim -v 2>&1 | tee "$results/q${rows}_tests.log"
    python bench.py --lengths 4096 "$@" --batch 1 --heads 32 --head-dim 128 \
        --backend flash --json "$results/q${rows}.json" | tee "$results/q${rows}_bench.log"
done

python - "$results" <<'PY'
import json
from pathlib import Path
import sys

print("\nD128 query-tile comparison (C32): >1x means faster than Flash SDPA")
print("    Q      N      BW us    SDPA us   speedup")
for path in sorted(Path(sys.argv[1]).glob("q*.json")):
    data = json.loads(path.read_text())
    for row in data["results"]:
        print(f"{data['metadata']['q_tile']:5} {row['N']:6} "
              f"{row['custom_us']:10.2f} {row['sdpa_us']:10.2f} {row['speedup']:8.3f}x")
print(f"Resource reports (VGPR/scratch/spills): {sys.argv[1]}/q*_resources.log")
PY
