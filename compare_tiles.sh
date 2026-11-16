#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Fresh processes for each build configuration. Keep native exp2 enabled.
for tile in 32 64; do
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=64 -DFA_BW_KV_TILE=$tile -DFA_BW_MATRIX_V=0 -DFA_BW_FULL_KV=0 -DFA_BW_NATIVE_EXP2=1"
    python -m unittest test_flash_attn -v
    python bench.py "$@" --json "results_v4_c${tile}.json"
done
