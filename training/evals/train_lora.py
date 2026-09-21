# -*- coding: utf-8 -*-
"""QLoRA post-train for Brittain 4, sized for one 12 GB RTX 3060 under WSL.

Every memory decision here comes from something measured on this machine, not
from a rule of thumb. The comments say which, because each was a surprise once.

  model.train()        Peak was 20.5 GB at a 512-token window until this call
                       was added. HF's block skips checkpointing entirely when
                       `self.training` is false -- gradient_checkpointing_enable
                       alone does nothing. Adding it took peak to 9.02 GB.

  chunked loss         The vocabulary is 248,320. Materialising logits for a
                       4,096-token window costs 4096 * 248320 * 2 bytes = 2.0 GB
                       in bf16, and cross-entropy upcasts to float32, so the
                       real cost is several times that. The lm_head is applied
                       in slices under checkpointing instead, which trades a
                       little recompute for the difference between fitting and
                       not fitting.

  no expandable_segments
                       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True fails
                       under WSL with "Failed to create GPU mapping". It is the
                       usual advice and it does not work here.

  WSL's real ceiling   The CUDA context sees 10.98 of the 12.0 GiB, and the
                       driver spills to system RAM rather than raising OOM --
                       so an over-budget run does not crash, it just becomes
                       extremely slow. Peak memory is printed every few steps
                       because that is the only way to notice.

TRAIN/SERVE MISMATCH, STATED PLAINLY
The adapter is fitted to NF4-quantised fp16 weights and will be served against
an AutoRound W4A16 checkpoint. That normally transfers -- a LoRA is a low-rank
delta on the same base -- but "normally" is not evidence. Score every
checkpoint through vLLM on the served checkpoint, not through this stack.

Prompts are rendered with enable_thinking=False and thinking traces are dropped
from targets, which trains the path the web chat and compaction actually use.
Brittain Code's thinking-on path is therefore a regression risk and has to be
evaluated separately rather than assumed.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"))
ap.add_argument("--mix", default="/home/lukeb/brittain4/data/train_mix.jsonl")
ap.add_argument("--out", default="/home/lukeb/brittain4/adapters/run1")
ap.add_argument("--prepare-only", action="store_true",
                help="render the mix, report it, and stop before loading the "
                     "model. Rendering faults are cheap to find here and "
                     "expensive to find eleven hours in.")
ap.add_argument("--window", type=int, default=2048,
                help="measured: 2048 peaks at 9.59 GiB and fits; "
                     "4096 peaks at 11.81 GiB and spills")
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--alpha", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--save-every", type=int, default=100, help="optimizer steps")
ap.add_argument("--loss-chunk", type=int, default=512, help="tokens per lm_head slice")
ap.add_argument("--max-steps", type=int, default=0, help="0 = full run; >0 for a smoke test")
ap.add_argument("--seed", type=int, default=4)
ap.add_argument("--chat-template",
                default="/home/lukeb/brittain4/chat_template_brittain4.jinja",
                help="the template the SERVER uses; training with any other "
                     "one is a train/serve mismatch in the prompt itself")
args = ap.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)


def resolve_base(path):
    """Accept either a snapshot directory or the hub cache root."""
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snaps = sorted(__import__("glob").glob(os.path.join(path, "snapshots", "*")))
    if not snaps:
        raise SystemExit("no snapshot under %s" % path)
    return snaps[-1]


BASE = resolve_base(args.base)
print("base   : %s" % BASE)
print("mix    : %s" % args.mix)
print("output : %s" % args.out)

tok = AutoTokenizer.from_pretrained(BASE)

# The served template, not the base model's. vLLM is started with
# --chat-template chat_template_brittain4.jinja, which adds an identity system
# block when the caller sends no system message. Training without it taught the
# adapter a prompt shape that never reaches production.
if args.chat_template:
    if not os.path.exists(args.chat_template):
        raise SystemExit("chat template not found: %s" % args.chat_template)
    tok.chat_template = open(args.chat_template, encoding="utf-8").read()
    print("template: %s (%d bytes)"
          % (args.chat_template, len(tok.chat_template)))
else:
    print("template: base tokenizer default -- this will not match the server")

# ---------------------------------------------------------------- rendering
def normalize_calls(message):
    """Tool-call arguments must be a mapping before the template sees them.

    The corpus carries both shapes -- 21,589 calls store arguments as a dict and
    11,536 as a JSON string, because the models that produced them serialised
    differently. The chat template does `arguments | items`, which raises
    "Can only get item pairs from a mapping" on the string form. Normalising
    here keeps that difference out of both the template and the trainer.
    """
    calls = message.get("tool_calls")
    if not calls:
        return message
    fixed = []
    for call in calls:
        fn = dict(call.get("function") or {})
        name = fn.get("name") or call.get("name")
        arguments = fn.get("arguments", call.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                # Malformed JSON from the source model. Training on it would
                # teach the model to emit the same broken shape.
                return None
        if not isinstance(arguments, dict):
            return None
        fixed.append({"id": call.get("id", ""), "type": "function",
                      "function": {"name": name, "arguments": arguments}})
    out = dict(message)
    out["tool_calls"] = fixed
    return out


def render_fit(messages, assistant, kwargs, prompt_ids, full_ids):
    """Clip, then drop old turns, until the sequence fits the window.

    Split out so the tools-declared attempt and the tools-free retry share one
    implementation. Two copies of this would drift, and the difference would
    show up as an unexplained gap between prompt shapes.
    """
    original = messages
    for divisor in (5, 10, 20, 40):
        if len(full_ids) <= args.window:
            break
        budget = args.window // divisor
        if budget < 48:
            break
        clipped = []
        for message in original:
            content = message.get("content") or ""
            if not isinstance(content, str) or not content.strip():
                clipped.append(message)
                continue
            ids = tok(content, add_special_tokens=False).input_ids
            if len(ids) <= budget:
                clipped.append(message)
                continue
            head = tok.decode(ids[: budget * 2 // 3])
            tail = tok.decode(ids[-(budget // 3):])
            trimmed = dict(message)
            trimmed["content"] = "%s\n\n... [%d tokens omitted] ...\n\n%s" % (
                head, len(ids) - budget, tail)
            clipped.append(trimmed)
        if clipped == messages:
            continue
        messages = clipped
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs)
            full_text = tok.apply_chat_template(
                messages + [assistant], add_generation_prompt=False, **kwargs)
        except Exception:
            return None
        if not full_text.startswith(prompt_text):
            return None
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            return None

    attempts = 0
    while len(full_ids) > args.window and attempts < 6:
        attempts += 1
        head = 1 if messages and messages[0].get("role") == "system" else 0
        if len(messages) - head <= 1:
            break
        excess = len(full_ids) - args.window
        shed = 0
        cut = head
        while cut < len(messages) - 1 and shed < excess:
            body = messages[cut].get("content") or ""
            shed += (len(tok(body, add_special_tokens=False).input_ids) + 8
                     if isinstance(body, str) else 8)
            cut += 1
        messages = messages[:head] + messages[cut:]
        while (len(messages) - head > 1
               and messages[head].get("role") == "tool"):
            messages = messages[:head] + messages[head + 1:]
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs)
            full_text = tok.apply_chat_template(
                messages + [assistant], add_generation_prompt=False, **kwargs)
        except Exception:
            return None
        if not full_text.startswith(prompt_text):
            return None
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            return None

    if len(full_ids) > args.window:
        return None
    prompt_len = len(prompt_ids)
    return full_ids, [-100] * prompt_len + full_ids[prompt_len:]


def render(row):
    """Return (input_ids, labels) with the loss masked to the target turn.

    The prompt and the full sequence are both produced by the chat template, and
    the target is their difference. Hand-assembling the target text would let
    training drift from what the server renders -- the exact class of mismatch
    that has broken this project's results before.
    """
    # An assistant turn with nothing before it is not a step the model can be
    # asked to reproduce, and apply_chat_template refuses an empty conversation.
    if not row.get("messages"):
        return None

    messages = []
    for message in row["messages"]:
        fixed = normalize_calls(dict(message))
        if fixed is None:
            return None
        messages.append(fixed)

    target = row["target"]
    assistant = {"role": "assistant", "content": target.get("content") or ""}
    if target.get("tool_calls"):
        # The template serialises these into the XML shape the qwen3_xml parser
        # reads back at serve time.
        fixed = normalize_calls({"tool_calls": target["tool_calls"]})
        if fixed is None:
            return None
        assistant["tool_calls"] = fixed["tool_calls"]

    kwargs = dict(tokenize=False, chat_template_kwargs={"enable_thinking": False})
    # Declared on trajectory rows only; None elsewhere, which renders no tool
    # block at all. Both renders below must receive it.
    if row.get("tools"):
        kwargs["tools"] = row["tools"]
    try:
        prompt_text = tok.apply_chat_template(
            messages, add_generation_prompt=True, **kwargs)
        full_text = tok.apply_chat_template(
            messages + [assistant], add_generation_prompt=False, **kwargs)
    except TypeError:
        # Older signatures take enable_thinking directly.
        tools = row.get("tools") or None
        prompt_text = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False, tools=tools)
        full_text = tok.apply_chat_template(
            messages + [assistant], add_generation_prompt=False, tokenize=False,
            enable_thinking=False, tools=tools)

    if not full_text.startswith(prompt_text):
        return None                      # template did not extend the prompt
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    if len(full_ids) <= len(prompt_ids):
        return None

    # Clip oversized bodies before dropping anything. A single tool result can
    # exceed the whole remaining window, in which case no amount of dropping
    # old turns helps and the example is lost entirely.
    original = messages
    for divisor in (5, 10, 20, 40):
        if len(full_ids) <= args.window:
            break
        budget = args.window // divisor
        if budget < 48:
            break            # below this a clipped result outlines nothing
        clipped = []
        for message in original:
            content = message.get("content") or ""
            if not isinstance(content, str) or not content.strip():
                clipped.append(message)
                continue
            ids = tok(content, add_special_tokens=False).input_ids
            if len(ids) <= budget:
                clipped.append(message)
                continue
            head = tok.decode(ids[: budget * 2 // 3])
            tail = tok.decode(ids[-(budget // 3):])
            trimmed = dict(message)
            trimmed["content"] = "%s\n\n... [%d tokens omitted] ...\n\n%s" % (
                head, len(ids) - budget, tail)
            clipped.append(trimmed)
        if clipped == messages:
            continue
        messages = clipped
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs)
            full_text = tok.apply_chat_template(
                messages + [assistant], add_generation_prompt=False, **kwargs)
        except Exception:
            return None
        if not full_text.startswith(prompt_text):
            return None
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            return None

    # An over-long example loses old TURNS, not the head of the sequence.
    # Token-level front truncation would eat the tool block, which the template
    # emits before any message: at a 2,048 window that mangled 295 of 300
    # trajectory examples, leaving a fragment of JSON in place of the tool
    # declarations. Dropping whole messages keeps both the tool block and the
    # target, and matches what the app does when a session outgrows its window.
    attempts = 0
    while len(full_ids) > args.window and attempts < 6:
        attempts += 1
        head = 1 if messages and messages[0].get("role") == "system" else 0
        if len(messages) - head <= 1:
            break                        # only the latest turn left to keep

        # Shed the excess in one pass. Message cost is estimated from its own
        # text plus a small per-turn overhead for the role header, which is
        # close enough -- the render below is what decides.
        excess = len(full_ids) - args.window
        shed = 0
        cut = head
        while cut < len(messages) - 1 and shed < excess:
            body = messages[cut].get("content") or ""
            shed += len(tok(body, add_special_tokens=False).input_ids) + 8 if isinstance(body, str) else 8
            for call in messages[cut].get("tool_calls") or []:
                shed += len(tok(json.dumps(call.get("function") or {}),
                                add_special_tokens=False).input_ids)
            cut += 1
        messages = messages[:head] + messages[cut:]
        # A tool result whose call was just dropped is not a valid opening.
        while (len(messages) - head > 1
               and messages[head].get("role") == "tool"):
            messages = messages[:head] + messages[head + 1:]
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs)
            full_text = tok.apply_chat_template(
                messages + [assistant], add_generation_prompt=False, **kwargs)
        except Exception:
            return None
        if not full_text.startswith(prompt_text):
            return None
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            return None

    if len(full_ids) > args.window and kwargs.get("tools"):
        # The declaration is what does not fit. Re-render without it rather
        # than lose the step: a tools-free example still teaches which tool to
        # reach for, which is what run 1 trained on and what it improved.
        kwargs.pop("tools")
        messages = original
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs)
            full_text = tok.apply_chat_template(
                messages + [assistant], add_generation_prompt=False, **kwargs)
        except Exception:
            return None
        if not full_text.startswith(prompt_text):
            return None
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            return None
        return render_fit(messages, assistant, kwargs, prompt_ids, full_ids)

    if len(full_ids) > args.window:
        # The target alone outgrows the window. Training a clipped target
        # would teach a truncated answer, so this example is dropped.
        return None

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return full_ids, labels


print("\nrendering...")
examples = []
skipped = 0
for line in open(args.mix, encoding="utf-8"):
    built = render(json.loads(line))
    if built is None:
        skipped += 1
        continue
    examples.append(built)
random.shuffle(examples)
trained_tokens = sum(sum(1 for l in lab if l != -100) for _, lab in examples)
print("  %d examples (%d unusable), %s target tokens"
      % (len(examples), skipped, "{:,}".format(trained_tokens)))

# The template emits the tool block before any message. If truncation cut into
# it, the opening <tools> survives without its closing tag -- a fragment of
# JSON that teaches nothing and is invisible in a loss curve.
intact = partial = 0
for ids, _labels in examples:
    text = tok.decode(ids)
    opened, closed = "<tools>" in text, "</tools>" in text
    if opened and closed:
        intact += 1
    elif opened or closed:
        partial += 1
print("  tool blocks: %d intact, %d mangled" % (intact, partial))
if partial:
    raise SystemExit("%d examples carry a truncated tool block; training on "
                     "those teaches a fragment of JSON" % partial)
if intact == 0 and any('"tools"' in line for line in open(args.mix, encoding="utf-8")):
    raise SystemExit("the mix declares tools but no rendered example kept a "
                     "tool block -- the agent half of the mix was dropped, "
                     "which reads as a clean render and trains nothing")

if args.prepare_only:
    lengths = sorted(len(ids) for ids, _ in examples)
    print("\n  sequence length: median %d, p90 %d, max %d (window %d)"
          % (lengths[len(lengths) // 2], lengths[int(len(lengths) * 0.9)],
             lengths[-1], args.window))
    raise SystemExit(0)

# ------------------------------------------------------------------- model
print("\nloading base in NF4...")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map={"": 0},
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# The module namespace is verified rather than assumed. A regex written for the
# multimodal class (model.language_model.layers.N.*) matched nothing against the
# text-only class (model.layers.N.*), and training ran to completion producing
# an adapter that changed no outputs at all.
linear_names = {name.split(".")[-1] for name, mod in model.named_modules()
                if isinstance(mod, torch.nn.Linear) or "Linear4bit" in type(mod).__name__}
targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj") if n in linear_names]
if len(targets) < 4:
    raise SystemExit("expected attention and MLP projections, found %s in %s"
                     % (targets, sorted(linear_names)[:20]))
print("  LoRA targets: %s" % ", ".join(targets))

# prepare_model_for_kbit_training upcasts what it touches to fp32 for numerical
# stability. Cheap and right for the norms; for the two vocabulary-sized tensors
# it is 8.14 GB, measured. Neither is trained -- the LoRA targets attention and
# MLP projections -- so both go back to bf16. This alone took the resident floor
# from 11.95 GB, already over the 10.98 GiB the WSL context can see, to 7.89 GB.
for module in (model.get_input_embeddings(), model.get_output_embeddings()):
    if module is not None and module.weight.dtype == torch.float32:
        module.to(torch.bfloat16)

model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
    bias="none", task_type="CAUSAL_LM", target_modules=targets,
))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print("  trainable: %s of %s (%.3f%%)"
      % ("{:,}".format(trainable), "{:,}".format(total), 100.0 * trainable / total))
if trainable == 0:
    raise SystemExit("no trainable parameters -- the adapter would be a no-op")

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.train()          # without this, checkpointing silently does nothing

# --------------------------------------------------------------------- loss
def chunked_loss(hidden, labels, head, chunk):
    """Cross-entropy without ever holding full-vocabulary logits.

    hidden: [1, T, H]  labels: [1, T]. Shifted so position t predicts t+1.
    """
    hidden = hidden[0, :-1, :]
    target = labels[0, 1:]
    keep = target != -100
    if keep.sum() == 0:
        return None
    hidden, target = hidden[keep], target[keep]
    total = hidden.new_zeros(())

    # The loss must be computed INSIDE the checkpoint. With only the head
    # inside, each slice still retained its log-softmax -- the same
    # 248,320-wide tensor the chunking was meant to avoid, 508 MB per slice,
    # held until backward. That was 7.9 GB of the original overshoot, and it
    # looked like it was working: the loss descended and the adapter was valid.
    # Inside, only a scalar survives and the slice is recomputed in backward.
    def slice_loss(x, y):
        return torch.nn.functional.cross_entropy(head(x).float(), y, reduction="sum")

    for start in range(0, hidden.size(0), chunk):
        total = total + checkpoint(slice_loss, hidden[start:start + chunk],
                                   target[start:start + chunk], use_reentrant=False)
    return total / target.numel()


head = model.get_output_embeddings()


def transformer_body(peft_model):
    """The decoder stack, not the causal LM that wraps it.

    Under PEFT, `model.model` is the *base* causal LM, so calling it runs the
    lm_head too and returns [1, T, 248320] logits -- which is precisely the
    allocation the chunked loss exists to avoid, and then fails with a shape
    error when those logits are fed back into the head. The body is one level
    further in, and the assertion below makes a wrong guess fail immediately
    rather than after a long run.
    """
    base = peft_model.get_base_model()
    body = getattr(base, "model", None)
    if body is None:
        raise SystemExit("could not reach the decoder stack under %s" % type(base).__name__)
    return body


body = transformer_body(model)
with torch.no_grad():
    probe = body(input_ids=torch.tensor([[1, 2, 3]], device="cuda"), use_cache=False)
    probe_hidden = probe.last_hidden_state if hasattr(probe, "last_hidden_state") else probe[0]
expected = model.config.hidden_size
if probe_hidden.shape[-1] != expected:
    raise SystemExit("body returns width %d, expected hidden_size %d -- wrong module"
                     % (probe_hidden.shape[-1], expected))
print("  decoder body: %s -> hidden %d" % (type(body).__name__, probe_hidden.shape[-1]))
del probe, probe_hidden
torch.cuda.empty_cache()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

micro_total = int(len(examples) * args.epochs)
opt_steps = max(1, micro_total // args.accum)
if args.max_steps:
    opt_steps = min(opt_steps, args.max_steps)
    micro_total = opt_steps * args.accum
schedule = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr, total_steps=opt_steps, pct_start=0.03, anneal_strategy="cos")

print("\ntraining: %d micro-batches -> %d optimizer steps (accum %d)"
      % (micro_total, opt_steps, args.accum))
print("=" * 78)

torch.cuda.reset_peak_memory_stats()
started = time.time()
step = 0
seen_tokens = 0
seq_tokens = 0
running = 0.0
counted = 0

for micro in range(micro_total):
    ids, labels = examples[micro % len(examples)]
    input_ids = torch.tensor([ids], device="cuda")
    label_ids = torch.tensor([labels], device="cuda")

    out = body(input_ids=input_ids, use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    loss = chunked_loss(hidden, label_ids, head, args.loss_chunk)
    if loss is None:
        continue
    (loss / args.accum).backward()
    running += loss.item()
    counted += 1
    seen_tokens += int((label_ids != -100).sum())
    seq_tokens += int(input_ids.numel())

    if (micro + 1) % args.accum == 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 5 == 0 or step == 1:
            peak = torch.cuda.max_memory_allocated() / 1e9
            elapsed = time.time() - started
            done = (micro + 1) / max(1, micro_total)
            print("step %4d/%d  loss %.4f  lr %.2e  peak %.2f GB  "
                  "%.0f seq tok/s (%.0f target)  eta %.1f h"
                  % (step, opt_steps, running / max(1, counted),
                     schedule.get_last_lr()[0], peak,
                     seq_tokens / max(1e-9, elapsed),
                     seen_tokens / max(1e-9, elapsed),
                     (elapsed / max(1e-9, done) - elapsed) / 3600))
            running, counted = 0.0, 0

        # Every checkpoint is kept and none is called "best". Picking by name
        # has been the wrong choice four times out of four on this project;
        # the eval decides, after the fact, on the served stack.
        if step % args.save_every == 0 or step == opt_steps:
            path = os.path.join(args.out, "step-%04d" % step)
            model.save_pretrained(path)
            tok.save_pretrained(path)
            print("  saved %s" % path)

print("=" * 78)
elapsed = time.time() - started
print("done in %.1f min, peak %.2f GB (%.2f GiB), %s target / %s sequence tokens"
      % (elapsed / 60, torch.cuda.max_memory_allocated() / 1e9,
         torch.cuda.max_memory_allocated() / 2**30,
         "{:,}".format(seen_tokens), "{:,}".format(seq_tokens)))
print("\nEvaluate every checkpoint through vLLM against the w4a16 checkpoint.")
print("A score from this process would be measuring the wrong stack.")
