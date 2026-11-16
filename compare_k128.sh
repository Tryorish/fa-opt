#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Compare C64/C128 at two query sizes. Keep the validated math/copy choices.
for rows in 64 128; do
    for keys in 64 128; do
        echo "Testing Q${rows} / C${keys} / ASYNC_K=1 / ASYNC_V=1 / DEFER_DENOM=1"
        export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=$keys -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_ASYNC_V=1 -DFA_BW_ASYNC_K=1 -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -DFA_BW_DEFER_DENOM=1 -Rpass-analysis=kernel-resource-usage -Werror=inline-asm"
        python -m unittest test_flash_attn -v
        python bench.py "$@" --backend flash --json "results_v15_q${rows}_c${keys}.json"
    done
done
