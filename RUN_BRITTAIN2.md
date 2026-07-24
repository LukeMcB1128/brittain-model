# Running BRITTAIN-2-coder end to end (GCP L4)

~235M params, ~14.7B tokens of Python/JS/TS + 15% English. **~9–10 days, ~$180.**
Budget ~$90 in reserve for the SFT stage and mistakes.

Each phase has a **GATE** — a cheap check that must pass before you spend on the next
one. Do not skip them; a tokenizer bug found on day 6 costs $100 and a week.

---

## Phase 0 — start the box

From **Cloud Shell** (`@cloudshell` prompt):

```bash
gcloud compute instances start brittain-train --zone=us-central1-c
```

**If it fails with `ZONE_RESOURCE_POOL_EXHAUSTED` (stockout):** just retry a few
times over 15–30 min — capacity comes back. If it stays dry, create a fresh box in
another zone (your L4 quota covers all of us-central1):

```bash
gcloud compute instances create brittain-train2 \
  --zone=us-central1-a --machine-type=g2-standard-8 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE --boot-disk-size=250GB \
  --metadata="install-nvidia-driver=True" --scopes=cloud-platform
```
(Then use `brittain-train2` / `us-central1-a` in every command below.)

## Phase 1 — connect and set up

```bash
gcloud compute ssh brittain-train --zone=us-central1-c
```
Now on the box (`@brittain-train` prompt):

```bash
cd brittain-model
git pull
pip install -r requirements.txt          # adds `tokenizers`
nvidia-smi                               # confirm the L4 is there
df -h /                                  # need ~60GB free (corpus is ~30GB)
```

**HuggingFace login** (The Stack is a *gated* dataset — one-time):
1. Accept the terms at https://huggingface.co/datasets/bigcode/the-stack-dedup
2. Make a read token at https://huggingface.co/settings/tokens
3. `huggingface-cli login` and paste it.

## Phase 2 — train the tokenizer (~10–20 min)

```bash
python3 train_tokenizer.py
```

> **GATE 1 (most important).** It prints a comparison like
> `gpt2: 68 tokens | code BPE: 50 tokens (26% fewer)`.
> You want **~20–30% fewer**. If it's near 0%, the pre-tokenizer isn't merging
> indentation — stop and fix before spending anything.

## Phase 3 — test data + test train (~1–2 h, a few $)

```bash
python3 prepare_code.py --tokens 2e8      # small 200M-token corpus
```
> **GATE 2.** Should report roughly `15% english` and write `data/train.bin`
> (~400MB) and `data/val.bin`.

```bash
BRITTAIN_PRESET=cloud_235m python3 train.py
```
> **GATE 3.** Watch for:
> - `Parameters: 235,176,960`
> - initial loss **~10.4** (= `ln(32000)`; if you see ~10.9 the vocab is wrong)
> - `~19k tok/s`
> - loss dropping over the first few hundred iters
>
> Then **Ctrl-C**. Delete the test checkpoint so the real run starts clean:
> ```bash
> rm -f brittain_235m.pt brittain_235m_best.pt
> ```

## Phase 4 — the real run

Build the full corpus (several hours, ~30GB — note this **overwrites** the old
BRITTAIN-1 FineWeb `.bin` files, which is fine, they're regenerable):

```bash
tmux new -s brittain2
python3 prepare_code.py --tokens 15e9
```

Then launch training **in the same tmux session**:

```bash
BRITTAIN_PRESET=cloud_235m python3 train.py
```
Detach with **Ctrl-b** then **d**. Training survives disconnects.

⚠️ **Do not stop the VM** during the run — a stopped GPU VM can hit a stockout and
refuse to restart. Leave it running; checkpoints land every ~1.5h and auto-resume
handles crashes.

## Phase 5 — checking in

```bash
gcloud compute ssh brittain-train --zone=us-central1-c    # get ON the box first
tmux attach -t brittain2
```
(`tmux attach` only works from the VM, not from Cloud Shell.)

Watch the eval lines. **val loss is the signal** — it should keep declining. If val
flattens while train keeps dropping, that's overfitting; tell Claude.

## Phase 6 — SFT, retrieve, shut down

Instruction-tune it (~1 h):
```bash
python3 prepare_sft.py && python3 train_sft.py
python3 chat.py brittain_235m_sft.pt
```

Shrink the checkpoint before downloading (drops optimizer state, 2.8GB → ~940MB):
```bash
python3 -c "
import torch
ck = torch.load('brittain_235m_best.pt', map_location='cpu')
torch.save({'model': ck['model'], 'cfg': ck['cfg'],
            'tokenizer': ck.get('tokenizer','code_bpe'), 'iter': ck.get('iter')},
           'brittain_235m_weights.pt')
print('done')"
```

Download (two hops — `cloudshell download` only exists in **Cloud Shell**):
```bash
exit                                      # leave the VM
gcloud compute scp brittain-train:~/brittain-model/brittain_235m_weights.pt ~/ --zone=us-central1-c
gcloud compute scp brittain-train:~/brittain-model/data/code_bpe.json ~/ --zone=us-central1-c
cloudshell download ~/brittain_235m_weights.pt
cloudshell download ~/code_bpe.json
```
**You need `code_bpe.json` too** — without it the model can't be tokenized locally.
Put it in `data/` on your Mac.

**Stop the box (ends billing):**
```bash
gcloud compute instances stop brittain-train --zone=us-central1-c
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ZONE_RESOURCE_POOL_EXHAUSTED` | Retry, or create in `us-central1-a`/`-b` |
| `CUDA out of memory` | Edit `cloud_235m`: `batch_size=8, grad_accum_steps=64` |
| `cloudshell: command not found` | You're on the VM — `exit` to Cloud Shell first |
| `does not have any valid credentials` | `gcloud config set account luke.brittain@gmail.com`; if the metadata server is wedged, **⋮ → Restart** Cloud Shell |
| `tmux: can't find session` | SSH onto the VM *first*, then `tmux attach` |
| Gated dataset / 401 from HF | `huggingface-cli login`, and accept The Stack's terms on its page |

## Rough budget

| Phase | Time | Cost |
|---|---|---|
| Tokenizer + tests | ~2 h | ~$2 |
| Corpus build | ~3–8 h | ~$5 |
| **Pretraining** | **~9 days** | **~$165** |
| SFT | ~1 h | ~$1 |
| **Total** | | **~$175** (leaves ~$95) |
