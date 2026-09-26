#!/usr/bin/env bash
# Serve BRITTAIN-4 with an API key, in the FOREGROUND.
#
# The key comes from BRITTAIN_API_KEY in the environment and is never written
# into the repo. Generate one once and keep it in your shell profile:
#   export BRITTAIN_API_KEY="$(openssl rand -hex 24)"
#
# Do not background this: WSL reaps the process group when the invoking shell
# returns, so a backgrounded `vllm serve` reaches "Application startup
# complete" and is killed a moment later -- a log that reads like success and a
# port that answers nothing.
#
# --reasoning-parser moves the thinking trace out of message.content and into
# reasoning_content. Without it the trace renders as visible chat: a plain
# "hello" came back as "The user is greeting me. I should respond in a
# friendly, natural way" followed by the real reply. Brittain Code already
# reads delta.reasoning_content on its openai transport, so this is the fix
# rather than switching thinking off -- the reasoning is doing useful work.
#
# --allowed-origins is deliberately NOT "*" here. A wildcard plus a public
# tunnel means any page on the internet can call this endpoint from a visitor's
# browser. Set SITE_ORIGIN to the real site before going public.
source "$HOME/venv-vllm/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0
pkill -9 -f "VLLM::Engine""Core" 2>/dev/null
sleep 2

# Read the key here rather than taking it through the caller's environment:
# passing $(cat ...) through PowerShell -> wsl -> bash gets interpolated by the
# outermost shell, which resolved it against the Windows filesystem and failed.
if [ -z "${BRITTAIN_API_KEY:-}" ] && [ -f "$HOME/.brittain4_key" ]; then
    BRITTAIN_API_KEY=$(cat "$HOME/.brittain4_key")
fi
if [ -z "${BRITTAIN_API_KEY:-}" ]; then
    echo "No API key. Set BRITTAIN_API_KEY or create ~/.brittain4_key." >&2
    echo "  openssl rand -hex 24 > ~/.brittain4_key && chmod 600 ~/.brittain4_key" >&2
    exit 1
fi

ORIGINS="${SITE_ORIGIN:-*}"
if [ "$ORIGINS" = "*" ]; then
    ORIGIN_ARG='["*"]'
    echo "WARNING: CORS is open to any origin. Set SITE_ORIGIN before this is public."
else
    ORIGIN_ARG="[\"$ORIGINS\"]"
fi

export VLLM_API_KEY="$BRITTAIN_API_KEY"

# vLLM registers /v1/load_lora_adapter and /v1/unload_lora_adapter only
# when this is set. Without it --enable-lora still serves adapters named at
# startup, but the runtime endpoints 404 against a server that otherwise
# looks perfectly healthy -- which cost a whole scoring pass.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

# Adapters are loaded at startup, not registered by hand afterwards. The site
# asks for one by name, and a restart that came back with only the base left it
# asking for an adapter that was not there -- for seven hours, the night of run
# 5. The first --lora-modules entry is production; the rest are rollbacks.
# Changing the served adapter means changing it here AND DEFAULT_MODEL in
# site/server/gateway.js, then deploying the site.
#
# Port 11435 belongs to the recorder, not vLLM: it passes every request through
# to vLLM on 127.0.0.1:11436 and keeps a copy of tunnel traffic in
# ~/brittain4/records (see scripts/inference/recorder.py). The tunnel, the site,
# Brittain Code and the eval scripts all keep using 11435. vLLM binds to
# loopback so nothing can reach it around the recorder. RECORDER=off serves
# vLLM directly on 11435, as before.
#
# The recorder restarts itself if it dies -- it is on the path of every reply --
# and goes when vLLM does. No `exec` below, or the trap would never run.
if [ "${RECORDER:-on}" = "off" ]; then
    VLLM_HOST=0.0.0.0; VLLM_PORT=11435
else
    VLLM_HOST=127.0.0.1; VLLM_PORT=11436
    RECORDER_PY="${RECORDER_PY:-/mnt/c/Coding/brittain-model/scripts/inference/recorder.py}"
    pkill -f "[r]ecorder.py --port 11435" 2>/dev/null
    (
        while true; do
            python "$RECORDER_PY" --port 11435 --upstream "http://127.0.0.1:$VLLM_PORT" \
                --records "$HOME/brittain4/records"
            echo "recorder exited ($?); restarting" >&2
            sleep 1
        done
    ) &
    RECORDER_LOOP=$!
    trap 'kill $RECORDER_LOOP 2>/dev/null; pkill -f "[r]ecorder.py --port 11435"' EXIT
fi

vllm serve \
    --model /home/lukeb/brittain4/models/brittain4-base-w4a16 \
    --served-model-name brittain4 \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --attention-backend TRITON_ATTN \
    --enable-lora \
    --max-lora-rank 32 \
    --lora-modules \
        run5b-step-0124=/home/lukeb/brittain4/adapters/run5b/step-0124-mm \
        run4c-step-0116=/home/lukeb/brittain4/adapters/run4c/step-0116-mm \
        run3-step-0116=/home/lukeb/brittain4/adapters/run3/step-0116-mm \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser "${REASONING_PARSER:-qwen3}" \
    --chat-template /home/lukeb/brittain4/chat_template_brittain4.jinja \
    --allowed-origins "$ORIGIN_ARG"
