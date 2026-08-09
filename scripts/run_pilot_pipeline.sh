#!/bin/bash
# Brittain3 49M pilot: corpus -> decontaminate -> merge -> pack -> train.
#
# Each step is gated on the previous one succeeding. A failure stops the chain
# rather than training on a half-built dataset, which would burn ten hours and
# produce a number nobody can trust.
#
#   bash scripts/run_pilot_pipeline.sh
#
# Resumable: every step skips itself when its output already exists. Delete the
# output to force a rebuild.
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1
PY=/usr/local/bin/python3
RAW=data/raw/brittain3-pilot
PROC=data/processed/brittain3
mkdir -p runs "$PROC"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[$(date '+%H:%M:%S')] FAILED: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. collect
if [ ! -f "$RAW/corpus2.jsonl" ]; then
    log "waiting for the expanded corpus build to finish ..."
    while pgrep -f "build_tokenizer_corpus_v3.py" > /dev/null; do sleep 20; done
fi
[ -f "$RAW/corpus2.jsonl" ] || die "$RAW/corpus2.jsonl was never written; see runs/pilot-corpus-build2.log"
log "corpus2.jsonl present ($(du -h "$RAW/corpus2.jsonl" | cut -f1))"

# ---------------------------------------------------------- 2. decontaminate
if [ ! -f "$RAW/corpus2.clean.jsonl" ]; then
    log "decontaminating against the novice suite ..."
    $PY -u scripts/prepare/decontaminate_v3.py \
        --input "$RAW/corpus2.jsonl" \
        --output "$RAW/corpus2.clean.jsonl" \
        --report "$RAW/decontamination2.report.json" \
        --overwrite > runs/pilot-decontaminate2.log 2>&1 \
        || die "decontamination (see runs/pilot-decontaminate2.log)"
fi
# Nested quoting inside $( ... ) with an f-string mangles the message under bash.
# Keep the Python simple and let it do its own formatting.
log "decontaminated: $($PY - "$RAW/decontamination2.report.json" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
print("{:,} kept, {} removed".format(report["documents_kept"], report["documents_removed"]))
EOF
)"

# ------------------------------------------------------------------ 3. merge
if [ ! -f "$RAW/pilot2.jsonl" ]; then
    [ -f data/generated/brittain3-pilot/exercises.jsonl ] || die "exercises.jsonl missing"
    log "merging corpus with the verified exercises ..."
    cat "$RAW/corpus2.clean.jsonl" data/generated/brittain3-pilot/exercises.jsonl > "$RAW/pilot2.jsonl" \
        || die "merge"
fi
log "merged: $(wc -l < "$RAW/pilot2.jsonl" | tr -d ' ') documents, $(du -h "$RAW/pilot2.jsonl" | cut -f1)"

# ------------------------------------------------------------------- 4. pack
# Both context stages need their own packed file. Packing is the memory-heavy
# step; pack_segments_streaming keeps it near 12-25GB rather than 49GB+.
for BLOCK in 1024 2048; do
    if [ ! -f "$PROC/train_$BLOCK.npz" ]; then
        log "packing at block size $BLOCK (this takes a while) ..."
        $PY -u scripts/prepare/prepare_brittain3.py \
            --input "$RAW/pilot2.jsonl" \
            --output-dir "$PROC" \
            --block-size "$BLOCK" \
            > "runs/pilot-pack-$BLOCK.log" 2>&1 \
            || die "packing at $BLOCK (see runs/pilot-pack-$BLOCK.log)"
    fi
    log "packed $BLOCK: $(tail -2 "runs/pilot-pack-$BLOCK.log" | head -1)"
done

# ------------------------------------------------------------------ 5. train
log "starting the 49M pilot training run"
log "  config configs/training/brittain3_49m_pilot.json"
log "  707,788,800 tokens across a 1K and a 2K stage"
exec caffeinate -is $PY -u scripts/train/pretrain_v3.py \
    --config configs/training/brittain3_49m_pilot.json \
    >> runs/pilot-train.log 2>&1
