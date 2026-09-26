#!/usr/bin/env bash
# Run 5a then 5b, unattended: build, train, remap, serve, score -- then leave
# production serving whatever serve.sh loads, whatever happened.
#
# Lives in ~/brittain4/evals, not a session scratchpad: the scratchpad is
# cleared between sessions, and that is what killed the first run of
# score_all.sh. The whole body is one function called on the last line, so
# bash has parsed all of it before running any of it -- a script edited or
# deleted mid-run cannot change what this does.
#
# Production is down while a run trains; the GPU cannot serve and train at
# once. The EXIT trap brings it back with the served adapter registered, and
# the script ends by waiting on the server, because a server backgrounded
# inside a WSL session dies when that session's last process exits.

main() {
set -o pipefail   # not -u: venv activate scripts touch unset variables
REPO=/mnt/c/Coding/brittain-model
E="$HOME/brittain4/evals"; D="$HOME/brittain4/data"
A="$HOME/brittain4/adapters"; RES="$HOME/brittain4/results"
LOG="$RES/overnight_run5.log"
KEY=$(tr -d '\r\n' < "$HOME/.brittain4_key")
SERVER_PID=""
mkdir -p "$RES"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

healthy() { curl -sf -m 3 -o /dev/null http://localhost:11435/health; }

register() {  # name, adapter dir
    curl -s -o /dev/null -w "%{http_code}" -m 60 \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -X POST http://localhost:11435/v1/load_lora_adapter \
        -d "{\"lora_name\": \"$1\", \"lora_path\": \"$2\"}"
}

start_server() {
    healthy && return 0
    bash "$HOME/brittain4/serve.sh" >> "$RES/serve_overnight.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 90); do healthy && break; sleep 10; done
    healthy || { say "SERVER DID NOT COME UP"; return 1; }
    # serve.sh loads the production adapter and its rollbacks at startup.
    # This script used to register run3-step-0116 here as "production" --
    # a name the site had stopped asking for the day before, which left the
    # site pointing at an unloaded adapter for seven hours. Never hardcode it.
    say "server up"
}

stop_server() {
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null
    SERVER_PID=""; sleep 8
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    say "server stopped; gpu in use ${used} MiB"
    [ "$used" -lt 2000 ]
}

production() {
    # Whatever state the night ended in, leave the site able to answer.
    start_server || true
    say "production: server up with the adapters serve.sh loads"
}
trap 'say "exiting; restoring production"; production' EXIT

mirror() {
    for f in build_mix.py build_run4_sft.py run_general_eval.py run_bs_eval.py \
             served_generate.py eval_defects.py audit_contamination.py jev.py; do
        tr -d '\r' < "$REPO/training/evals/$f" > "$E/$f"
    done
}

last_step() { ls -d "$1"/step-[0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -1; }

train_run() {  # run name, behaviour-builder flags
    local run="$1"; shift
    say "==== $run: data"
    ( cd "$REPO/training/evals" && python3 build_run4_sft.py "$@" --out "$D/${run}_sft.jsonl" ) >> "$LOG" 2>&1 \
        || { say "$run: behaviour build refused"; return 1; }
    ( cd "$REPO/training/evals" && python3 audit_contamination.py --heldout --train "$D/${run}_sft.jsonl" ) >> "$LOG" 2>&1 \
        || { say "$run: held-out probes contaminated -- refusing to train"; return 1; }
    ( cd "$E" && python3 build_mix.py --general 500 --trajectory 300 --brittainscript 0 \
        --tool-mode names --exclude-general 'wildjailbreak|wildguard|coconot' \
        --behaviour "${run}_sft.jsonl" --out "$D/train_mix_${run}.jsonl" ) >> "$LOG" 2>&1 \
        || { say "$run: mix refused"; return 1; }
    say "$run: $(grep -E '^total:' "$LOG" | tail -1)"
    [ -e "$A/$run" ] && { say "$run: $A/$run exists -- refusing to overwrite"; return 1; }
    stop_server || { say "$run: GPU still held -- refusing to train"; return 1; }
    say "==== $run: training"
    ( source "$HOME/venv/bin/activate" && cd "$E" && python3 train_lora.py \
        --mix "$D/train_mix_${run}.jsonl" --out "$A/$run" --save-every 25 ) \
        > "$A/${run}-train.log" 2>&1 || { say "$run: training failed"; return 1; }
    say "$run: $(grep 'done in' "$A/${run}-train.log")"
    say "==== $run: remap"
    ( source "$HOME/venv/bin/activate" && cd "$E" && for d in "$A/$run"/step-[0-9][0-9][0-9][0-9]; do
        python3 remap_adapter.py --adapter "$d" > /dev/null 2>&1; done
      python3 - "$A/$run" <<'PY'
import glob, sys
from safetensors.torch import load_file
bad = 0
for p in sorted(glob.glob(sys.argv[1] + "/step-*-mm/adapter_model.safetensors")):
    keys = list(load_file(p).keys())
    ok = sum("language_model" in k for k in keys) == len(keys) == 256
    bad += not ok
    print(p.split("/")[-2], "ok" if ok else "NOT REMAPPED")
raise SystemExit(1 if bad else 0)
PY
    ) >> "$LOG" 2>&1 || { say "$run: remap check failed"; return 1; }
    start_server || return 1
}

score() {  # served name
    local m="$1"
    say "---- scoring $m"
    source "$HOME/venv-vllm/bin/activate"
    cd "$E"
    [ -f "$RES/general_$m.json" ] || python3 run_general_eval.py --model "$m" --server "$m" \
        --eval "$D/eval_general.jsonl" --out "$RES/general_$m.json" --label "$m" >> "$LOG" 2>&1
    [ -f "$RES/code_$m.json" ] || python3 run_general_eval.py --model "$m" --server "$m" \
        --eval "$D/eval_code.jsonl" --out "$RES/code_$m.json" --label "$m" >> "$LOG" 2>&1
    [ -f "$RES/bs_$m.json" ] || python3 run_bs_eval.py --model "$m" --server "$m" \
        --eval "$D/eval_bs.jsonl" --out "$RES/bs_$m.json" --label "$m" >> "$LOG" 2>&1
    deactivate
    if [ "$m" != "brittain4" ] && [ ! -f "$RES/heldout_$m.json" ]; then
        ( cd "$REPO/training/evals" && python3 eval_defects.py --heldout --samples 24 \
            --model "$m" --out "$RES/heldout_$m.json" ) >> "$LOG" 2>&1
    fi
    say "$m scored"
}

register_run() {  # run name -> registers the final checkpoint, echoes its served name
    local d; d=$(last_step "$A/$1")
    local name="$1-$(basename "$d")"
    say "$name -> $(register "$name" "$d-mm")"
    SERVED="$name"
}

# ---------------------------------------------------------------- the night
say "######## overnight run 5 starting"
mirror
# Measured through the server earlier today; 493 of 494 agree with offline.
[ -f "$RES/general_brittain4.json" ] || cp "$RES/general_brittain4_served.json" "$RES/general_brittain4.json"
start_server || exit 1
register run4b-step-0114 "$A/run4b/step-0114-mm" > /dev/null
register run4c-step-0116 "$A/run4c/step-0116-mm" > /dev/null

# Run 5a: run 4c's mix with the Tulu safety subsets removed. Nothing else.
if train_run run5a; then
    register run4b-step-0114 "$A/run4b/step-0114-mm" > /dev/null
    register run4c-step-0116 "$A/run4c/step-0116-mm" > /dev/null
    register_run run5a
    # The code eval with power, on everything, so HumanEval can be attributed.
    # 5a first, so the result that matters lands before the comparisons.
    for m in "$SERVED" brittain4 run3-step-0116 run4b-step-0114 run4c-step-0116; do score "$m"; done
fi

# Run 5b: 5a plus the fabrication set. Dial unchanged, so 5a against 5b reads
# the fabrication data alone.
mirror
if train_run run5b --figures; then
    register_run run5b
    score "$SERVED"
fi

say "######## overnight run 5 done"
trap - EXIT
production
# Keep this session alive, and with it the server.
if [ -n "$SERVER_PID" ]; then wait "$SERVER_PID"; else while healthy; do sleep 300; done; fi
}
main "$@"
