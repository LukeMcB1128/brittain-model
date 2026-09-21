#!/usr/bin/env bash
# Remap every run 2 checkpoint into the multimodal module namespace.
#
# The served w4a16 checkpoint loads as Qwen3_5ForConditionalGeneration, whose
# decoder lives under model.language_model.layers.N, while training saw
# Qwen3_5ForCausalLM with model.layers.N. An adapter keyed the training way
# loads without error and matches zero modules -- it changes nothing at all,
# and reads as a training failure rather than a plumbing one. That cost this
# project a whole scoring pass once.
set -e
source "$HOME/venv-vllm/bin/activate"
for ckpt in /home/lukeb/brittain4/adapters/run2/step-[0-9]*; do
    case "$ckpt" in
        *-mm) continue ;;
    esac
    echo "=== $(basename "$ckpt") ==="
    python3 /home/lukeb/brittain4/evals/remap_adapter.py --adapter "$ckpt"
done
echo
echo "checkpoints now present:"
ls -1 /home/lukeb/brittain4/adapters/run2/
