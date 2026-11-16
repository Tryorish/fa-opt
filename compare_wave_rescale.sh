#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Compare the wave-uniform vote against unconditional rescaling.
for rows in 128 256; do
    for uniform in 0 1; do
        echo "Testing Q${rows} / C64 / ASYNC_K=1 / ASYNC_V=1 / DEFER_DENOM=1 / WAVE_RESCALE=${uniform}"
        export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=64 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_ASYNC_V=1 -DFA_BW_ASYNC_K=1 -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_WAVE_RESCALE=$uniform -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -DFA_BW_DEFER_DENOM=1 -DFA_BW_REDUCE4=0 -Rpass-analysis=kernel-resource-usage -Werror=inline-asm"
        python -m unittest test_flash_attn -v
        python bench.py "$@" --backend flash --json "results_v18_q${rows}_wave_rescale_${uniform}.json"
    done
done
