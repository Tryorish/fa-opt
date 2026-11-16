#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Isolate conditional rescaling, retaining the verified full-KV/V-matrix path.
for conditional in 0 1; do
    echo "Testing Q128 / C32 / matrix V / full KV / CONDITIONAL_RESCALE=${conditional}"
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=128 -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_CONDITIONAL_RESCALE=$conditional -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
    python -m unittest test_flash_attn -v
    python bench.py --backend flash "$@" --json "results_v10_rescale${conditional}.json"
done
