#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Keep C32 and native exp2 fixed. Compiler remarks report VGPRs and occupancy.
for late in 0 1; do
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=64 -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=0 -DFA_BW_FULL_KV=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=$late -Rpass-analysis=kernel-resource-usage"
    python -m unittest test_flash_attn -v
    python bench.py "$@" --json "results_v5_prefetch${late}.json"
done
