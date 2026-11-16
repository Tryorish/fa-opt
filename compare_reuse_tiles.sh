#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Q128/Q256 already use one wave/SIMD with C32 on the measured compiler.
# Check whether C64 reduces loop overhead without spills or further occupancy loss.
for rows in 128 256; do
    for keys in 32 64; do
        echo "Testing Q${rows} / C${keys} (before_qk); compiler resources follow."
        export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=$keys -DFA_BW_MATRIX_V=0 -DFA_BW_FULL_KV=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
        python -m unittest test_flash_attn -v
        python bench.py --backend flash "$@" --json "results_v7_q${rows}_c${keys}.json"
    done
done
