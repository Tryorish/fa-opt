#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Change only V's global-to-LDS path; preserve the measured Q128/C64 control.
for direct in 0 1; do
    echo "Testing Q128 / C64 / matrix V / full KV / ASYNC_V=${direct}"
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=128 -DFA_BW_KV_TILE=64 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_ASYNC_V=$direct -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
    python -m unittest test_flash_attn -v
    python bench.py "$@" --backend flash --json "results_v12_async_v${direct}.json"
done
