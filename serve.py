"""
Ollama-compatible HTTP server for any BRITTAIN checkpoint.

    pip install fastapi uvicorn

    # autocomplete (base models — raw prefix continuation, no template)
    python3 serve.py brittain_235m_weights.pt --raw
    python3 serve.py brittain_50m_bs.pt --raw

    # instruction mode (SFT'd models — wraps input in the Alpaca template)
    python3 serve.py brittain_124m_sft.pt

Serves on http://localhost:11435 (11434 is real Ollama, so both can coexist).
Point Continue.dev / Brittain Code at it as an Ollama provider.

RAW vs TEMPLATED
  --raw is for autocomplete: the prompt goes to the model untouched, generation
  is short, and it stops at a blank line. This is the right mode for base models
  like the coders — completion is what they natively do.
  Without --raw, /api/generate and /api/chat wrap input in the instruction
  template, which only makes sense for an SFT'd checkpoint.

  Per-request `"raw": true/false` overrides the CLI default, matching Ollama's
  own API.

LIMITATION: these models were not trained with fill-in-the-middle, so they can
only continue a prefix — good at end-of-line/end-of-function, poor at editing
mid-file. FIM would be a data-format change in prepare_code.py for a future run.
"""
import sys
import json
import time
import codecs
import argparse

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from model import Brittain, GPTConfig
from tok_util import load_tokenizer
from sft_prompt import format_prompt
import model_bs

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?", default="brittain_124m_sft.pt")
ap.add_argument("--raw", action="store_true",
                help="default to raw completion (autocomplete) instead of the template")
ap.add_argument("--name", default=None, help="model name advertised to clients")
ap.add_argument("--port", type=int, default=11435)
args = ap.parse_args()

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

# --- load: BS checkpoints are a bare ModuleList state_dict with no 'cfg' ---
_ck = torch.load(args.checkpoint, map_location=device)
if not isinstance(_ck, dict) or "cfg" not in _ck:
    model, enc = model_bs.load(args.checkpoint, device)
    BLOCK = model.block
else:
    cfg = GPTConfig(**_ck["cfg"])
    model = Brittain(cfg).to(device)
    model.load_state_dict(_ck["model"])
    model.eval()
    enc = load_tokenizer(_ck)
    BLOCK = cfg.block_size
del _ck

NAME = args.name or args.checkpoint.replace(".pt", "")
NPARAM = f"{model.num_params()/1e6:.0f}M"
IS_BRITTAIN = isinstance(model, Brittain)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _step(ids, temperature, top_p, rep):
    if IS_BRITTAIN:
        return model.generate(ids, max_new_tokens=1, temperature=temperature,
                              top_p=top_p, repetition_penalty=rep)
    return model.generate(ids, 1, temperature=temperature, top_p=top_p,
                          repetition_penalty=rep)


def stream_pieces(prompt, raw, opts):
    """Yield decoded text pieces, honouring Ollama-style options and stops."""
    if raw:
        # autocomplete: near-greedy, short, stop at a blank line
        temperature = opts.get("temperature", 0.2)
        max_new = opts.get("num_predict", 48)
        stops = opts.get("stop") or ["\n\n"]
        rep = opts.get("repeat_penalty", 1.05)
    else:
        temperature = opts.get("temperature", 0.5)
        max_new = opts.get("num_predict", 400)
        stops = opts.get("stop") or []
        rep = opts.get("repeat_penalty", 1.12)
    top_p = opts.get("top_p", 0.95)

    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    ids = ids[:, -BLOCK:]                       # never exceed the context window
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    acc = ""
    with torch.no_grad():
        for _ in range(max_new):
            ids = _step(ids[:, -BLOCK:], temperature, top_p, rep)
            nxt = ids[0, -1].item()
            if nxt == enc.eot:
                break
            piece = utf8.decode(enc.token_bytes(nxt))
            if not piece:
                continue
            acc += piece
            hit = next((s for s in stops if s in acc), None)
            if hit:                              # emit up to the stop, then halt
                cut = acc[:acc.index(hit)]
                tail = cut[len(acc) - len(piece):] if len(cut) > len(acc) - len(piece) else ""
                if tail:
                    yield tail
                return
            yield piece


@app.get("/api/tags")
def tags():
    return {"models": [{"name": NAME, "model": NAME, "modified_at": now(),
                        "size": 0, "digest": NAME,
                        "details": {"family": "brittain", "parameter_size": NPARAM},
                        "context": str(BLOCK)}]}


@app.get("/api/version")
def version():
    return {"version": "brittain-0.2"}


@app.post("/api/show")
async def show(req: Request):
    return {"details": {"family": "brittain", "parameter_size": NPARAM},
            "capabilities": ["completion"], "context_length": BLOCK}


@app.post("/api/generate")
async def generate(req: Request):
    body = await req.json()
    opts = body.get("options") or {}
    raw = body.get("raw", args.raw)
    prompt = body.get("prompt", "")
    if not raw:
        prompt = format_prompt(prompt)

    def gen():
        for p in stream_pieces(prompt, raw, opts):
            yield json.dumps({"model": NAME, "created_at": now(),
                              "response": p, "done": False}) + "\n"
        yield json.dumps({"model": NAME, "created_at": now(), "response": "",
                          "done": True, "done_reason": "stop"}) + "\n"

    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return JSONResponse({"model": NAME, "created_at": now(),
                         "response": "".join(stream_pieces(prompt, raw, opts)),
                         "done": True})


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    opts = body.get("options") or {}
    msgs = body.get("messages", [])
    user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
    # single-turn: these models never saw multi-turn conversations in training
    prompt = user if args.raw else format_prompt(user)

    def gen():
        for p in stream_pieces(prompt, args.raw, opts):
            yield json.dumps({"model": NAME, "created_at": now(),
                              "message": {"role": "assistant", "content": p},
                              "done": False}) + "\n"
        yield json.dumps({"model": NAME, "created_at": now(),
                          "message": {"role": "assistant", "content": ""},
                          "done": True, "done_reason": "stop"}) + "\n"

    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return JSONResponse({"model": NAME, "created_at": now(),
                         "message": {"role": "assistant",
                                     "content": "".join(stream_pieces(prompt, args.raw, opts))},
                         "done": True})


if __name__ == "__main__":
    mode = "RAW completion (autocomplete)" if args.raw else "instruction template"
    print(f"BRITTAIN '{NAME}' ({NPARAM}, ctx {BLOCK}) on http://localhost:{args.port}")
    print(f"mode: {mode} | tokenizer: {enc.name} | device: {device}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
