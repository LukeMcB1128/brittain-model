"""
Ollama-compatible HTTP server for BRITTAIN checkpoints. Serves several at once
so clients can switch between them.

    pip install fastapi uvicorn

    # serve everything you have; mode is inferred per model
    python3 scripts/inference/serve.py checkpoints/brittain2_50m_bs.pt \
        checkpoints/brittain2_235m_weights.pt checkpoints/brittain2_235m_fim.pt

    # rename for nicer client-side labels
    python3 scripts/inference/serve.py \
        checkpoints/brittain2_50m_bs.pt=brittain2-xs-coder:50m-bs

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

FILL-IN-THE-MIDDLE: for a real BRITTAIN FIM checkpoint, client marker dialects
are translated to BRITTAIN's three sentinel tokens and the suffix is preserved.
For older causal checkpoints, the markers are stripped and only the prefix is
used, preserving the legacy autocomplete fallback.
"""
import os
import json
import time
import codecs
import asyncio
import threading
import argparse
import sys
import itertools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from brittain.model import Brittain, GPTConfig
from brittain.model_v3 import Brittain3, Brittain3Config
from brittain import model_bs
from brittain.loading import document_prefix
from brittain.paths import CHECKPOINT_DIR
from brittain.prompts import format_prompt
from brittain.tokenizer import load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("checkpoints", nargs="*",
                help="checkpoint paths, optionally PATH=display-name. "
                     "Omit to auto-discover all .pt files under checkpoints/.")
ap.add_argument("--raw", action="store_true", help="force raw mode for all models")
ap.add_argument("--instruct", action="store_true", help="force template mode for all models")
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--host", default="127.0.0.1",
                help="127.0.0.1 keeps it local; ngrok tunnels to it either way")
ap.add_argument("--cors-origin", action="append", default=None,
                help="allowed browser origin, repeatable. Defaults to '*'. A "
                     "GitHub Pages site should name its own origin — that does "
                     "not stop curl, but it stops other sites embedding this "
                     "endpoint from a visitor's browser.")
args = ap.parse_args()
args.cors_origin = args.cors_origin or ["*"]

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))


def load_card(path, checkpoint):
    """Descriptive metadata for a checkpoint: languages, mixture, provenance.

    Two sources, most authoritative first.

    1. A `corpus` block inside the checkpoint itself. This travels with the
       weights and cannot drift from them, which is why the tokenizer identity
       and the full training plan already live there.
    2. `card.json` beside the checkpoint, or `<name>.card.json` next to a
       loose .pt. Convenient for checkpoints trained before the payload carried
       a corpus block — like the 49M pilot — but it is a separate file, so it can
       be lost or go stale. Prefer (1) for anything trained from here on.

    Returns {} when neither exists. A model with no card still serves; it just
    reports nothing beyond what is derivable from the weights.
    """
    embedded = {}
    if isinstance(checkpoint, dict):
        embedded = checkpoint.get("corpus") or {}
        meta = checkpoint.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("corpus"), dict):
            embedded = {**meta["corpus"], **embedded}
    candidates = [Path(path).parent / "card.json",
                  Path(str(path)[:-3] + ".card.json") if str(path).endswith(".pt") else None]
    for candidate in candidates:
        if candidate and candidate.is_file():
            try:
                on_disk = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  [warn] unreadable card {candidate}: {exc}")
                continue
            # The embedded block wins: it cannot have drifted from the weights.
            return {**on_disk, **embedded}
    return embedded


class Loaded:
    def __init__(self, path, name):
        ck = torch.load(path, map_location=device, weights_only=False)
        # BS checkpoints are a bare ModuleList state_dict with no 'cfg' key
        if not isinstance(ck, dict) or "cfg" not in ck:
            self.model, self.enc = model_bs.load(path, device)
            self.block = self.model.block
        else:
            if ck.get("architecture") == "brittain3":
                cfg = Brittain3Config(**ck["cfg"])
                self.model = Brittain3(cfg).to(device)
                self.block = cfg.max_seq_len
            else:
                cfg = GPTConfig(**ck["cfg"])
                self.model = Brittain(cfg).to(device)
                self.block = cfg.block_size
            self.model.load_state_dict(ck["model"])
            self.model.eval()
            self.enc = load_tokenizer(ck)
        self.name = name
        self.params = self.model.num_params()
        # Brittain3 saw every pretraining document wrapped as
        # <|repo_start|>repo<|file_start|>path, so an unframed prompt is out of
        # distribution: the model emits <|file_end|><|repo_end|> and stops. The
        # server must supply the framing or every completion comes back empty.
        # Brittain1/2 have no such tokens and correctly get "".
        self.frame = document_prefix(self.enc, "workspace/project", "main.py")
        self.frame_ids = self.enc.encode(self.frame) if self.frame else []
        self.card = load_card(path, ck if isinstance(ck, dict) else {})
        self.is_brittain = isinstance(self.model, (Brittain, Brittain3))
        self.supports_fim = bool(getattr(self.enc, "has_fim", False))
        low = os.path.basename(path).lower()
        if args.raw:
            self.raw_default = True
        elif args.instruct:
            self.raw_default = False
        else:
            self.raw_default = not ("sft" in low or "instruct" in low)


# Brittain3 training writes weights.pt/best.pt/latest.pt into a per-run
# DIRECTORY, so the old one-level glob missed them entirely, and naming by file
# stem would have served them as "weights" and "best" — useless in a model picker
# and ambiguous the moment there are two runs.
# latest.pt only: mid-run it duplicates best.pt, after a run it duplicates the
# final weights. Everything else in a run directory is offered.
RUN_DIR_SKIP = {"latest.pt"}


def run_dir_label(run_name, stem):
    """Name a checkpoint found inside a per-run directory.

    Training writes weights.pt/best.pt, but a finished model is often renamed to
    something meaningful like `brittain3-xs-coder:49m-pilot.pt` — which is
    already a good served name and should be used as-is. Only the generic
    training filenames need the directory to disambiguate them.
    """
    if stem == "weights":
        return run_name
    if stem == "best":
        return f"{run_name}-best"
    return stem


def discover():
    """Find filed checkpoints, repo-root checkpoints, and Brittain3 run dirs."""
    named = []
    found = list(CHECKPOINT_DIR.glob("*.pt")) + list(PROJECT_ROOT.glob("*.pt"))
    for path in sorted({p.resolve() for p in found if "model_backup" not in p.name},
                       key=lambda p: p.stat().st_mtime, reverse=True):
        named.append(str(path))
    for run in sorted(CHECKPOINT_DIR.glob("*/"), key=lambda p: p.name):
        for candidate in sorted(run.glob("*.pt")):
            if candidate.name in RUN_DIR_SKIP:
                continue
            named.append(f"{candidate.resolve()}={run_dir_label(run.name, candidate.stem)}")
    return named


specs = args.checkpoints or discover()
if not specs:
    ap.error(f"no checkpoints given and no .pt files found under {CHECKPOINT_DIR}")

MODELS = {}
for spec in specs:
    path, _, disp = spec.partition("=")
    stem = os.path.basename(path)[:-3] if path.endswith(".pt") else os.path.basename(path)
    name = disp or stem
    try:
        MODELS[name] = Loaded(path, name)
    except Exception as exc:                 # incompatible/corrupt checkpoint
        print(f"  [skip] {path}: {type(exc).__name__}: {exc}")
if not MODELS:
    ap.error("no checkpoints could be loaded")
DEFAULT = next(iter(MODELS))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=args.cors_origin,
                   allow_methods=["*"], allow_headers=["*"])
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- public limits
# Tunnelled through ngrok, this endpoint is reachable by anyone who has the URL,
# and every request runs on Luke's laptop. Three bounds, none of which a normal
# user or Continue.dev will ever touch:
#
#   * generation length — an unbounded num_predict holds the GPU for as long as
#     the caller likes.
#   * prompt length — checked in CHARACTERS, before tokenizing, because
#     tokenizing a 50MB prompt IS the denial of service. Truncating to block_size
#     afterwards is too late.
#   * request rate per IP — a token bucket, so bursts are fine and sustained
#     hammering is not.
#
# Deliberately NOT an in-flight request counter. Generation is serialised per
# token, so concurrent callers interleave and all make progress. A counter would
# need decrementing in a generator's finally, and an abandoned generator stays
# suspended without running it — exactly the bug that wedged this server before.
MAX_NEW_TOKENS = 512
MAX_PROMPT_CHARS = 20_000
RATE_BURST = 4          # requests available instantly
RATE_PER_MIN = 40       # sustained refill

_buckets = {}
_bucket_lock = threading.Lock()


def rate_limited(request):
    """Token bucket per client IP. True when the caller should back off."""
    ip = (request.client.host if request.client else "?")
    nowt = time.time()
    with _bucket_lock:
        tokens, last = _buckets.get(ip, (float(RATE_BURST), nowt))
        tokens = min(RATE_BURST, tokens + (nowt - last) * RATE_PER_MIN / 60.0)
        if tokens < 1.0:
            _buckets[ip] = (tokens, nowt)
            return True
        _buckets[ip] = (tokens - 1.0, nowt)
        if len(_buckets) > 4096:            # bound the map itself
            for k in [k for k, (_, t) in _buckets.items() if nowt - t > 3600]:
                _buckets.pop(k, None)
    return False


def too_many(request):
    if rate_limited(request):
        return JSONResponse({"error": "rate limited — slow down"}, status_code=429)
    return None


# Continue.dev and most autocomplete clients assume a fill-in-the-middle model
# and wrap the prompt in sentinel tokens. BRITTAIN never saw those tokens, so
# they'd arrive as noise. We strip them and keep only the prefix.
FIM_MARKERS = [
    ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),        # StarCoder / SantaCoder
    ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),  # Qwen / CodeQwen
    ("<PRE>", "<SUF>", "<MID>"),                             # CodeLlama
    ("<｜fim▁begin｜>", "<｜fim▁hole｜>",
     "<｜fim▁end｜>"),                          # DeepSeek
]


def strip_fim(prompt):
    """-> (prefix, suffix). suffix is None when the prompt wasn't FIM-wrapped.

    We can't *use* the suffix as context (the model has no FIM training), but
    it tells us what already follows the cursor, which makes a useful stop
    sequence so the completion doesn't duplicate it."""
    for pre, suf, mid in FIM_MARKERS:
        if pre in prompt and suf in prompt:
            body = prompt.split(pre, 1)[1]
            prefix, rest = body.split(suf, 1)
            suffix = rest.split(mid, 1)[0] if mid in rest else rest
            return prefix, suffix
    # Codestral-style: [SUFFIX]...[PREFIX]...
    if "[SUFFIX]" in prompt and "[PREFIX]" in prompt:
        after = prompt.split("[SUFFIX]", 1)[1]
        suffix, prefix = after.split("[PREFIX]", 1)
        return prefix, suffix
    return prompt, None


def normalize_fim(prompt, canonical=("<fim_prefix>", "<fim_suffix>", "<fim_middle>")):
    """Translate common client FIM dialects into BRITTAIN's PSM format."""
    for pre, suf, mid in FIM_MARKERS:
        if pre in prompt and suf in prompt:
            body = prompt.split(pre, 1)[1]
            prefix, rest = body.split(suf, 1)
            suffix = rest.split(mid, 1)[0] if mid in rest else rest
            prepared = f"{canonical[0]}{prefix}{canonical[1]}{suffix}{canonical[2]}"
            return prepared, suffix
    if "[SUFFIX]" in prompt and "[PREFIX]" in prompt:
        after = prompt.split("[SUFFIX]", 1)[1]
        suffix, prefix = after.split("[PREFIX]", 1)
        prepared = f"{canonical[0]}{prefix}{canonical[1]}{suffix}{canonical[2]}"
        return prepared, suffix
    return prompt, None


def prepare_raw_completion(prompt, request_suffix, supports_fim, canonical=None):
    """Normalize wrapped FIM or Ollama's separate prompt/suffix form."""
    if supports_fim:
        prepared, embedded_suffix = normalize_fim(
            prompt, canonical or ("<fim_prefix>", "<fim_suffix>", "<fim_middle>")
        )
    else:
        prepared, embedded_suffix = strip_fim(prompt)
    if embedded_suffix is not None:
        return prepared, embedded_suffix
    if request_suffix is None:
        return prepared, None
    if supports_fim:
        markers = canonical or ("<fim_prefix>", "<fim_suffix>", "<fim_middle>")
        return f"{markers[0]}{prepared}{markers[1]}{request_suffix}{markers[2]}", request_suffix
    return prepared, request_suffix


def suffix_stop(suffix):
    """First substantial line after the cursor, used as a stop sequence."""
    if not suffix:
        return []
    first = suffix.lstrip("\n").split("\n", 1)[0].strip()
    return [first] if len(first) >= 4 else []


_warned_names = set()


def pick(body):
    """Route on the request's model field, falling back to the first loaded."""
    # Ollama uses `model` for generation but `name` for /api/show.
    want = ((body or {}).get("model") or (body or {}).get("name") or DEFAULT)
    if want in MODELS:
        return MODELS[want]
    # tolerate ollama-style "name:tag" and bare-stem mismatches
    base = want.split(":")[0]
    for k, v in MODELS.items():
        if k == base or k.split(":")[0] == base:
            return v
    # SAY SO. A silent fallback means a client asking for the 235M FIM model gets
    # a 52M BrittainScript model and no indication why the completions are wrong.
    if want not in _warned_names:
        _warned_names.add(want)
        print(f"  [warn] no model named {want!r}; serving {DEFAULT!r} instead. "
              f"Loaded: {', '.join(MODELS)}", flush=True)
    return MODELS[DEFAULT]


# Serialises all generation — see the comment at its use below.
GPU_LOCK = threading.Lock()
REQUEST_IDS = itertools.count(1)


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
    # Clamp, don't trust. Reachable from the public internet through ngrok, an
    # unbounded num_predict holds the one GPU for as long as the caller asks.
    max_new = max(1, min(int(max_new), MAX_NEW_TOKENS))

    token_ids = M.enc.encode(prompt)
    # An EMPTY prompt is normal, not a client error: the cursor sits at the top of
    # a file, so the FIM prefix is "". Feeding a zero-length tensor to the model
    # crashes on [:, -1, :] ("index -1 is out of bounds for dimension 1 with size
    # 0"). Seed with end-of-text instead, which is exactly the start-of-document
    # state these models saw between every training document.
    if not token_ids and not M.frame_ids:
        token_ids = [M.enc.eot]
    # Framing goes in FRONT and survives truncation. Trimming the combined
    # sequence with [:, -block:] would drop the <|repo_start|> prefix on any
    # prompt near the context limit — silently reintroducing the empty-completion
    # bug for exactly the long files where autocomplete matters most. For a framed
    # model an empty prompt needs no EOT seed: the frame IS the start-of-document
    # state, and it is what the model was trained to continue from.
    room = M.block - len(M.frame_ids)
    token_ids = M.frame_ids + token_ids[-room:]
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    acc = ""
    def token_stream():
        """One token at a time, with the KV cache held open for the whole
        completion. Both architectures implement stream() now — the BS models
        used to lack it and fell back to a full forward per token, which made the
        52M slower than the 235M."""
        yield from M.model.stream(ids, max_new, temperature=temperature,
                                  top_p=top_p, repetition_penalty=rep)

    # ONE REQUEST AT A TIME. StreamingResponse runs a sync generator in Starlette's
    # threadpool, so overlapping requests put two threads inside the same model.
    # The MPS backend is not thread-safe and the process segfaults — which it did,
    # after the second /api/generate, because Continue.dev cancels and re-fires
    # autocomplete on nearly every keystroke. There is one GPU; serialising costs
    # nothing real and turns a crash into a short wait.
    #
    # BOUNDED acquisition, not a plain `with`. The lock is held across yields, so a
    # generator abandoned mid-stream — Continue.dev cancels constantly — can leave
    # it held. An unbounded wait then blocks every later request forever, which is
    # indistinguishable from a dead server. Waiting a bounded time and refusing is
    # recoverable; hanging is not.
    def locked_tokens():
        """Serialise GPU work per TOKEN, not per request.

        Holding one lock for a whole completion deadlocks: a generator abandoned
        mid-stream — Continue.dev cancels on nearly every keystroke — is left
        SUSPENDED at a yield, so nothing runs its finally and the lock is never
        released. Every later request then waits forever, which is what wedged
        this server.

        Locking each token instead means an abandoned generator holds nothing: it
        is suspended between tokens, with the lock already released. Concurrent
        requests interleave rather than queue, which is fine — each has its own KV
        cache and the weights are read-only. The only thing that must not happen
        concurrently is two threads inside the model at once, and that is exactly
        what this prevents.
        """
        it = token_stream()
        while True:
            GPU_LOCK.acquire()
            try:
                tok = next(it)
            except StopIteration:
                return
            finally:
                GPU_LOCK.release()
            yield tok

    with torch.no_grad():
        for tok in locked_tokens():
            nxt = tok[0, -1].item()
            stop_ids = {M.enc.eot, *getattr(M.enc, "special_ids", {}).values()}
            if nxt in stop_ids:
                break
            piece = utf8.decode(M.enc.token_bytes(nxt))
            if not piece:
                continue
            prev_len = len(acc)
            acc += piece
            # Search for stops only AFTER the leading whitespace. A middle usually
            # opens on a fresh line, so acc starts "\n\n..."; searching the whole
            # string matches the default "\n\n" stop at index 0 the moment real
            # content arrives, cuts to "" and yields NOTHING. Deferring the check
            # until acc.strip() was non-empty only delayed that — the stop still
            # fired retroactively at position 0, and Continue.dev rendered an
            # empty completion for every suggestion that began with a blank line.
            lead = len(acc) - len(acc.lstrip())
            body = acc[lead:]
            hit = next((s for s in stops if s in body), None) if body.strip() else None
            if hit:
                cut = acc[:lead + body.index(hit)]
                if len(cut) > prev_len:
                    yield cut[prev_len:]
                return
            yield piece


def describe(m):
    """What a model is and how it wants to be talked to.

    A client picks its input shape from `mode`, so checkpoints can be swapped
    behind this without the client changing: 235m-fim -> 235m-fim-2k, 50m-bs ->
    50m-bs-4b, or a new *-instruct appearing, all show up correctly on their own.
    `fim` is derived from the tokenizer (vocab 32003) and `instruct` from the
    filename, so neither needs configuring.

      fim       prefix AND suffix; the model writes the middle
      raw       a prefix; the model continues it
      instruct  an instruction; the server applies the Alpaca template
    """
    # An instruct fine-tune can keep the FIM tokenizer from its base model.
    # It must still use the Alpaca prompt format. Report instruct first so the
    # browser sends an instruction instead of a prefix and suffix.
    mode = "instruct" if not m.raw_default else ("fim" if m.supports_fim else "raw")
    return {
        "name": m.name, "model": m.name, "modified_at": now(), "size": 0,
        "digest": m.name, "context": str(m.block),
        "mode": mode,
        # This is the public client capability, not merely the tokenizer
        # capability. An instruct model may retain FIM tokens but must not be
        # called as a FIM model.
        "supports_fim": mode == "fim",
        "max_tokens": MAX_NEW_TOKENS,
        "defaults": ({"temperature": 0.2, "num_predict": 64}
                     if m.raw_default else
                     {"temperature": 0.5, "num_predict": 256}),
        # A client cannot use a framed model correctly without knowing this.
        # It is applied server-side too, so a client that ignores it still works.
        "prompt_framing": m.frame or None,
        "languages": (m.card.get("primary_languages")
                      or sorted(m.card.get("languages", {}), key=lambda k: -m.card["languages"][k])[:3]
                      or None),
        "details": {"family": "brittain",
                    "parameter_size": f"{m.params/1e6:.0f}M",
                    "tokenizer": m.enc.name,
                    "mode": mode,
                    **({"languages": ", ".join(m.card["primary_languages"])}
                       if m.card.get("primary_languages") else {})},
    }


@app.get("/api/tags")
def tags():
    return {"models": [describe(m) for m in MODELS.values()]}


@app.get("/api/version")
def version():
    return {"version": "brittain-0.3"}


@app.post("/api/show")
async def show(req: Request):
    M = pick(await req.json())
    card = M.card
    info = {
        "general.architecture": "brittain",
        "general.parameter_count": M.params,
        "brittain.context_length": M.block,
        "brittain.tokenizer": M.enc.name,
        "brittain.vocab_size": M.enc.vocab_size,
    }
    # Dotted namespaced keys are Ollama's own convention for architecture facts,
    # so unknown ones are ignored by clients rather than rejected.
    if M.frame:
        info["brittain.prompt_framing"] = M.frame
    for key in ("languages", "primary_languages", "mixture", "corpus_tokens",
                "corpus_config_sha256", "training_tokens", "epochs", "notes"):
        if key in card:
            info[f"brittain.{key}"] = card[key]
    capabilities = ["completion"]
    if M.supports_fim and M.raw_default:
        capabilities.append("infill")
    if M.frame:
        capabilities.append("document-framing")
    result = {"details": {"family": "brittain",
                          "parameter_size": f"{M.params/1e6:.0f}M"},
              "capabilities": capabilities,
              "parameters": f"num_ctx {M.block}",
              "model_info": info,
              "context_length": M.block}
    if M.supports_fim and M.raw_default:
        # Continue's Ollama provider detects native FIM support by looking for
        # `.Suffix` in /api/show's template. Without this it silently chooses
        # streamComplete(prompt) instead of streamFim(prefix, suffix), even when
        # the selected model has an autocomplete role.
        result["template"] = "{{ .Prompt }}{{ .Suffix }}"
    return result


@app.post("/api/generate")
async def generate(req: Request):
    limited = too_many(req)
    if limited:
        return limited
    body = await req.json()
    M = pick(body)
    opts = body.get("options") or {}
    raw = body.get("raw", M.raw_default)
    prompt = body.get("prompt") or ""
    request_suffix = body.get("suffix")
    # Checked in CHARACTERS, before tokenizing — tokenizing a huge prompt IS the
    # denial of service, so truncating to block_size afterwards is too late.
    if len(prompt) + len(request_suffix or "") > MAX_PROMPT_CHARS:
        return JSONResponse(
            {"error": f"prompt too long (limit {MAX_PROMPT_CHARS} characters)"},
            status_code=413)
    stream = body.get("stream", True)
    request_id = next(REQUEST_IDS)
    print(f"[generate {request_id}] model={M.name!r} raw={raw} stream={stream} "
          f"prompt_chars={len(prompt)} "
          f"suffix_chars={len(request_suffix) if isinstance(request_suffix, str) else 0}",
          flush=True)

    # Continue sends empty probe/replacement requests during autocomplete. Reply
    # successfully without spending a generation on an empty document.
    empty_request = prompt == "" and request_suffix is None
    if raw:
        canonical = (("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
                     if M.enc.name == "brittain3_bpe" else None)
        prompt, suffix = prepare_raw_completion(
            prompt, request_suffix, M.supports_fim, canonical)
        if suffix is not None and not opts.get("stop"):
            extra = suffix_stop(suffix)
            if extra:
                opts = {**opts, "stop": ["\n\n"] + extra}
    else:
        prompt = format_prompt(prompt)

    def gen():
        chars = 0
        chunks = 0
        preview = ""
        completed = False
        try:
            if not empty_request:
                for p in stream_pieces(M, prompt, raw, opts):
                    chars += len(p)
                    chunks += 1
                    if len(preview) < 80:
                        preview += p[:80 - len(preview)]
                    yield json.dumps({"model": M.name, "created_at": now(),
                                      "response": p, "done": False}) + "\n"
            yield json.dumps({"model": M.name, "created_at": now(), "response": "",
                              "done": True, "done_reason": "stop"}) + "\n"
            completed = True
        finally:
            state = "complete" if completed else "cancelled"
            print(f"[generate {request_id}] {state} chunks={chunks} chars={chars} "
                  f"preview={preview!r}", flush=True)

    if stream:
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    # to_thread, NOT a direct call. This endpoint is async, so joining the
    # generator inline blocks uvicorn's event loop for the whole generation and
    # the server stops answering EVERYTHING — /api/version included. That looks
    # exactly like a dead server to a client that is still logging connections.
    text = "" if empty_request else await asyncio.to_thread(
        lambda: "".join(stream_pieces(M, prompt, raw, opts)))
    print(f"[generate {request_id}] complete chars={len(text)} "
          f"preview={text[:80]!r}", flush=True)
    return JSONResponse({"model": M.name, "created_at": now(),
                         "response": text, "done": True})


@app.post("/api/chat")
async def chat(req: Request):
    limited = too_many(req)
    if limited:
        return limited
    body = await req.json()
    M = pick(body)
    opts = body.get("options") or {}
    msgs = body.get("messages", [])
    if sum(len(m.get("content") or "") for m in msgs) > MAX_PROMPT_CHARS:
        return JSONResponse(
            {"error": f"prompt too long (limit {MAX_PROMPT_CHARS} characters)"},
            status_code=413)
    user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
    # single-turn: these models never saw multi-turn conversations in training
    raw = body.get("raw", M.raw_default)
    if raw:
        canonical = (("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
                     if M.enc.name == "brittain3_bpe" else None)
        prompt, _ = (normalize_fim(
            user, canonical or ("<fim_prefix>", "<fim_suffix>", "<fim_middle>")
        ) if M.supports_fim
                     else strip_fim(user))
    else:
        prompt = format_prompt(user)

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
        mode = ("instruct" if not m.raw_default
                else ("fim" if m.supports_fim else "raw"))
        print(f"  {m.name:<30} {m.params/1e6:6.0f}M  ctx {m.block:<5} "
              f"{m.enc.name:<9} {mode}")
    uvicorn.run(app, host=args.host, port=args.port)
