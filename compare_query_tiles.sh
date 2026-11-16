#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Isolate Q-row reuse: fixed C32, native exp2, early prefetch, 16 KiB LDS.
for rows in 64 128 256; do
    echo "Testing Q${rows} / C32 (before_qk); compiler resources follow."
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=0 -DFA_BW_FULL_KV=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
    python -m unittest test_flash_attn -v
    python bench.py --backend flash "$@" --json "results_v6_q${rows}.json"
done
