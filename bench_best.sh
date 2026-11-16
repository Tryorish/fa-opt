#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# )); then
    echo "Usage: bash bench_best.sh (fixed B1/H16, N1024/2048/4096, Graph timing)" >&2
    exit 2
fi
mkdir -p build/best
results_dir=$(mktemp -d build/best/run-XXXXXX)
echo "Results: $results_dir"

# Reproduce the measured winners. bench.py validates outputs before timing.
for rows in 128 256; do
    if (( rows == 128 )); then
        lengths=(1024 4096)
    else
        lengths=(2048)
    fi
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=64 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_ASYNC_V=1 -DFA_BW_ASYNC_K=1 -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_WAVE_RESCALE=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -DFA_BW_DEFER_DENOM=1 -DFA_BW_REDUCE4=0 -DFA_BW_DELAY_SCALE=1 -Rpass-analysis=kernel-resource-usage -Werror=inline-asm"
    python bench.py --batch 1 --heads 16 --lengths "${lengths[@]}" --backend flash \
        --timing graph --warmup 10 --inner 20 --repeats 10 --json "$results_dir/q${rows}.json"
done
