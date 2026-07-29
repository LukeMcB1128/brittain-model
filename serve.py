"""
Ollama-compatible HTTP server for BRITTAIN checkpoints. Serves several at once
so clients can switch between them.

    pip install fastapi uvicorn

    # serve everything you have; mode is inferred per model
    python3 serve.py brittain_50m_bs.pt brittain_235m_weights.pt brittain_124m_sft.pt

    # rename for nicer client-side labels
    python3 serve.py brittain_50m_bs.pt=brittain2-xs-coder:50m

Serves on http://localhost:11435 (11434 is real Ollama, so both can coexist).
Point Continue.dev / Brittain Code at it as an Ollama provider; the models show
up in /api/tags and requests route on the "model" field.

MODES
  raw       — prompt goes to the model untouched, short generation, stops at a
              blank line. Correct for BASE models (the coders): completion is
              what they natively do.
  instruct  — wraps input in the Alpaca template. Only correct for SFT'd
              checkpoints; templating a base model produces garbage.

  Inferred per model from the filename ("sft"/"instruct" -> instruct, else raw).
  Override globally with --raw / --instruct, or per request with "raw": true.

MEMORY: every checkpoint listed is loaded at startup. Roughly 1.5GB for the
52M, 2GB for the 124M, 3GB for the 235M in fp32.

LIMITATION: no fill-in-the-middle in training, so these continue a prefix only —
good at end-of-line/end-of-function, poor at editing mid-file.
"""
import os
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
ap.add_argument("checkpoints", nargs="+",
                help="checkpoint paths, optionally PATH=display-name")
ap.add_argument("--raw", action="store_true", help="force raw mode for all models")
ap.add_argument("--instruct", action="store_true", help="force template mode for all models")
ap.add_argument("--port", type=int, default=11435)
args = ap.parse_args()

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))


class Loaded:
    def __init__(self, path, name):
        ck = torch.load(path, map_location=device)
        # BS checkpoints are a bare ModuleList state_dict with no 'cfg' key
        if not isinstance(ck, dict) or "cfg" not in ck:
            self.model, self.enc = model_bs.load(path, device)
            self.block = self.model.block
        else:
            cfg = GPTConfig(**ck["cfg"])
            self.model = Brittain(cfg).to(device)
            self.model.load_state_dict(ck["model"])
            self.model.eval()
            self.enc = load_tokenizer(ck)
            self.block = cfg.block_size
        self.name = name
        self.params = self.model.num_params()
        self.is_brittain = isinstance(self.model, Brittain)
        low = os.path.basename(path).lower()
        if args.raw:
            self.raw_default = True
        elif args.instruct:
            self.raw_default = False
        else:
            self.raw_default = not ("sft" in low or "instruct" in low)


MODELS = {}
for spec in args.checkpoints:
    path, _, disp = spec.partition("=")
    name = disp or os.path.basename(path)[:-3] if path.endswith(".pt") else disp or path
    MODELS[name] = Loaded(path, name)
DEFAULT = next(iter(MODELS))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pick(body):
    """Route on the request's model field, falling back to the first loaded."""
    want = (body or {}).get("model") or DEFAULT
    if want in MODELS:
        return MODELS[want]
    # tolerate ollama-style "name:tag" and bare-stem mismatches
    base = want.split(":")[0]
    for k, v in MODELS.items():
        if k == base or k.split(":")[0] == base:
            return v
    return MODELS[DEFAULT]


def stream_pieces(M, prompt, raw, opts):
    if raw:
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

    ids = torch.tensor([M.enc.encode(prompt)], dtype=torch.long, device=device)
    ids = ids[:, -M.block:]
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    acc = ""
    with torch.no_grad():
        for _ in range(max_new):
            if M.is_brittain:
                ids = M.model.generate(ids[:, -M.block:], max_new_tokens=1,
                                       temperature=temperature, top_p=top_p,
                                       repetition_penalty=rep)
            else:
                ids = M.model.generate(ids[:, -M.block:], 1, temperature=temperature,
                                       top_p=top_p, repetition_penalty=rep)
            nxt = ids[0, -1].item()
            if nxt == M.enc.eot:
                break
            piece = utf8.decode(M.enc.token_bytes(nxt))
            if not piece:
                continue
            prev_len = len(acc)
            acc += piece
            hit = next((s for s in stops if s in acc), None)
            if hit:
                cut = acc[:acc.index(hit)]
                if len(cut) > prev_len:
                    yield cut[prev_len:]
                return
            yield piece


@app.get("/api/tags")
def tags():
    return {"models": [
        {"name": m.name, "model": m.name, "modified_at": now(), "size": 0,
         "digest": m.name, "context": str(m.block),
         "details": {"family": "brittain",
                     "parameter_size": f"{m.params/1e6:.0f}M"}}
        for m in MODELS.values()]}


@app.get("/api/version")
def version():
    return {"version": "brittain-0.3"}


@app.post("/api/show")
async def show(req: Request):
    M = pick(await req.json())
    return {"details": {"family": "brittain",
                        "parameter_size": f"{M.params/1e6:.0f}M"},
            "capabilities": ["completion"], "context_length": M.block}


@app.post("/api/generate")
async def generate(req: Request):
    body = await req.json()
    M = pick(body)
    opts = body.get("options") or {}
    raw = body.get("raw", M.raw_default)
    prompt = body.get("prompt", "")
    if not raw:
        prompt = format_prompt(prompt)

    def gen():
        for p in stream_pieces(M, prompt, raw, opts):
            yield json.dumps({"model": M.name, "created_at": now(),
                              "response": p, "done": False}) + "\n"
        yield json.dumps({"model": M.name, "created_at": now(), "response": "",
                          "done": True, "done_reason": "stop"}) + "\n"

    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return JSONResponse({"model": M.name, "created_at": now(),
                         "response": "".join(stream_pieces(M, prompt, raw, opts)),
                         "done": True})


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    M = pick(body)
    opts = body.get("options") or {}
    msgs = body.get("messages", [])
    user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
    # single-turn: these models never saw multi-turn conversations in training
    raw = body.get("raw", M.raw_default)
    prompt = user if raw else format_prompt(user)

    def gen():
        for p in stream_pieces(M, prompt, raw, opts):
            yield json.dumps({"model": M.name, "created_at": now(),
                              "message": {"role": "assistant", "content": p},
                              "done": False}) + "\n"
        yield json.dumps({"model": M.name, "created_at": now(),
                          "message": {"role": "assistant", "content": ""},
                          "done": True, "done_reason": "stop"}) + "\n"

    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return JSONResponse({"model": M.name, "created_at": now(),
                         "message": {"role": "assistant",
                                     "content": "".join(stream_pieces(M, prompt, raw, opts))},
                         "done": True})


if __name__ == "__main__":
    print(f"BRITTAIN serving {len(MODELS)} model(s) on http://localhost:{args.port}"
          f"  [device {device}]")
    for m in MODELS.values():
        print(f"  {m.name:<30} {m.params/1e6:6.0f}M  ctx {m.block:<5} "
              f"{m.enc.name:<9} {'raw' if m.raw_default else 'instruct'}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
