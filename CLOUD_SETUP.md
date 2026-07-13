# Training BRITTAIN-124M on Google Cloud ($300 credits)

Goal: pretrain the 124M BRITTAIN from scratch on FineWeb-Edu. Budget ~$40–70,
well inside your $300. This is a base model (coherent text/code completion), not
an instruction-following assistant — see the SFT capstone at the end.

## 0. The gotcha that wastes everyone's first day

A fresh Google Cloud account has a **GPU quota of 0**, and the free-trial credit
can't launch GPUs until you do two things:

1. **Upgrade to a full/paid Cloud Billing account.** This does NOT charge you —
   it unlocks GPUs and keeps spending your $300 credit, not real money.
   (Billing → "Activate full account" / "Upgrade".)
2. **Request a GPU quota increase.** IAM & Admin → Quotas → filter for
   `NVIDIA_L4_GPUS` (or `GPUS_ALL_REGIONS`) → select a region → Edit → request
   **1**. Approval is usually minutes–hours; occasionally a day. If denied for
   one region/GPU, try another region.

Do this first. Budget a possible day of waiting.

## 1. Pick the machine

Use **Vertex AI Workbench** (managed GPU notebook — far less setup than a raw VM,
uses your credits):

- Vertex AI → Workbench → Instances → Create New
- GPU: **NVIDIA L4** (24 GB, ~$0.70/hr — the sweet spot). A100 if you got quota
  and want it ~4x faster (~$3.67/hr).
- Boot disk: **100 GB+** (the 10B-token corpus is ~20 GB on disk).
- Environment: any recent PyTorch/CUDA image.

> Cost control: **stop the instance when you're not using it** — you're billed
> per hour it runs. Stopped instances cost almost nothing.

## 2. Set up the repo on the box

Open the Workbench JupyterLab terminal:

```bash
git clone <your-brittain-repo>        # or upload the files
cd BRITTAIN\ MODEL
pip install torch numpy tiktoken datasets tqdm
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"   # -> True
```

## 3. Build the data (do a small test first!)

```bash
# 1B-token test corpus — confirms the pipeline before the full run
python3 prepare_fineweb.py --tokens 1e9

# quick sanity train (few hundred iters), watch val loss drop, then Ctrl-C
BRITTAIN_PRESET=cloud_124m python3 train.py
```

Once that clearly learns, build the real corpus and train for real:

```bash
python3 prepare_fineweb.py --tokens 10e9         # ~10B tokens, a few hours
BRITTAIN_PRESET=cloud_124m python3 train.py      # the real run
```

`train.py` checkpoints every 500 iters to `brittain_124m.pt` (+ `_best.pt`) and
auto-resumes, so a crash or a stop/start costs at most 500 iters.

**Tip:** run training under `nohup ... &` or `tmux` so it survives you closing
the browser tab.

## 4. Time & cost estimate

124M params × 10B tokens ≈ ~6e19 FLOPs.
- **A100** (~130 TFLOPS realized): ~13 GPU-hrs ≈ **~$50**.
- **L4** (~40 TFLOPS realized): ~40 GPU-hrs ≈ **~$30** (cheaper/hr, slower).

Either way: one run is a fraction of the $300, so you can afford to iterate.

## 5. Get your model back

```bash
python3 sample.py brittain_124m_best.pt          # try it on the box
```

Download `brittain_124m_best.pt` (~0.5 GB) from JupyterLab to run locally on your
Mac with `python3 sample.py brittain_124m_best.pt` (it auto-uses MPS).

## 6. Capstone: make it instructable (the SFT stage)

After pretraining, the "type an instruction, get an attempt" behavior comes from
one more short, cheap fine-tune on instruction data (Alpaca format + loss
masking). That's a separate script — ask and it gets built once you have a base
model to fine-tune.
```
