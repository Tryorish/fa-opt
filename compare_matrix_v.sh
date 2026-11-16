#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Isolate V staging/reading, using the best measured N4096 tile.
for matrix in 0 1; do
    echo "Testing Q128 / C32 / MATRIX_V=${matrix}; compiler resources follow."
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=128 -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=$matrix -DFA_BW_FULL_KV=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
    python -m unittest test_flash_attn -v
    python bench.py --backend flash "$@" --json "results_v8_matrix${matrix}.json"
done
