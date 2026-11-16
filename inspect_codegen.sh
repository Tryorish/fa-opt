#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Current best Q128/Q256, plus the slower Q128 wave-branch experiment.
# Compile only; all output is pasteable and artifacts remain on this machine.
for config in '128 0' '128 1' '256 0'; do
    read -r rows uniform <<< "$config"
    echo "Inspecting Q${rows} / C64 / WAVE_RESCALE=${uniform}"
    export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=$rows -DFA_BW_KV_TILE=64 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=1 -DFA_BW_ASYNC_V=1 -DFA_BW_ASYNC_K=1 -DFA_BW_CONDITIONAL_RESCALE=0 -DFA_BW_WAVE_RESCALE=$uniform -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -DFA_BW_DEFER_DENOM=1 -DFA_BW_REDUCE4=0 -Werror=inline-asm"
    python diagnose.py --tiles 64 --details --full-only
done
