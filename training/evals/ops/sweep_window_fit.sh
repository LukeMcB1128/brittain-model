#!/usr/bin/env bash
# How much of the trajectory half survives at each window / tool-mode?
#
# At a 2,048 window an 883-token tool block leaves ~1,100 tokens for the whole
# conversation and the target, and an agent step may not fit in that at all.
# The alternatives are a cheaper declaration (names only, 181 tokens) or a
# larger window, which costs GPU memory. This measures rather than guesses.
set -u
source "$HOME/venv-vllm/bin/activate"
EVALS=/home/lukeb/brittain4/evals

run() {
    window="$1"
    mode="$2"
    echo "=== window ${window} / tool-mode ${mode} ==="
    python3 "$EVALS/build_mix.py" --tool-mode "$mode" >/dev/null 2>&1
    # The render exits non-zero when it keeps no tool blocks, which is the
    # answer here rather than an error, so the failure text is kept.
    python3 -u "$EVALS/train_lora.py" --prepare-only --window "$window" 2>&1 \
        | grep -aE 'examples \(|tool blocks|sequence length|no rendered example'
    echo
}

run 2048 names_desc
run 2048 names
run 3072 names_desc
run 4096 names_desc
