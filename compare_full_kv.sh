#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Only specialize complete KV tiles; tails keep the generic kernel.
for full in 0 1; do
    echo "Testing Q128 / C32 / matrix V / FULL_KV=${full}"
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=128 -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=$full -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
    python -m unittest test_flash_attn -v
    if [[ "$full" == 1 ]]; then
        # Both template instantiations appear separately in the summary.
        python diagnose.py --tiles 32
    fi
    python bench.py --backend flash "$@" --json "results_v9_fullkv${full}.json"
done
