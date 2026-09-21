"""Score a model on the driver eval, through the real serving stack.

    python3 run_driver_eval.py --model <path> --eval eval_driver.jsonl \
        --defs tool_defs.json --out results.json

MEASURED ON vLLM, NOT IN A NOTEBOOK
A train/serve mismatch has broken this project's results twice. The eval loads
the same quantized weights vLLM serves, through vLLM, with the same chat
template -- so a number here is a number about the thing that ships.

THE FORMAT IS QWEN'S OWN, WHICH IS ALSO BRITTAIN'S FALLBACK
Qwen3.5's chat template instructs the model to answer with

    <tool_call>
    <function=read_file>
    <parameter=path>
    src/main.js
    </parameter>
    </function>
    </tool_call>

which is exactly the markup parseRawToolCalls() rescues at main.js:1345. So the
native format and the app's recovery path agree, and this parser mirrors the
app's: a truncated trailing <parameter> is still recovered, and values matching
true|false|null|number are coerced.

WHAT IS SCORED
  emitted_call     a parseable call came out at all
  name_valid       the name is one of the 55
  required_present every required parameter is there
  schema_valid     arguments validate against that tool's JSON Schema
  name_match       same tool the reference model chose

The first four need no ground truth and are the ones that decide whether the
model can drive the app. name_match is a weaker signal: the reference is a
stronger model's choice, not a correct answer.
"""
import argparse
import collections
import json
import os
import re
import sys

CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.S)
FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|\Z)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)(?:\n?</parameter>|\Z)", re.S)


def coerce(value):
    """Turn one <parameter> body into a JSON value.

    Scalars follow the app's own coercion (true|false|null|number). Objects and
    arrays have to be parsed too: the chat template renders complex arguments
    with |tojson on the way in, so the model emits them as JSON text on the way
    out, and a structured tool-call parser is what turns them back.

    Leaving them as strings scored six of seven schema failures in the first
    baseline run -- "[{...}]" is not of type 'array' -- which was the scorer
    being wrong, not the model.
    """
    v = value.strip()
    if v in ("true", "false"):
        return v == "true"
    if v == "null":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v[:1] in ("[", "{"):
        try:
            return json.loads(v)
        except ValueError:
            return value          # malformed JSON is a real failure; keep it
    return value


def parse_calls(text):
    out = []
    for block in CALL_RE.findall(text or ""):
        for name, body in FUNC_RE.findall(block):
            args = {k: coerce(v) for k, v in PARAM_RE.findall(body)}
            out.append({"name": name, "arguments": args})
    return out


def as_object(args):
    """Arguments reach us as an object or a JSON string, depending on transport.

    Ollama stores them as an object; the OpenAI path stores the raw string, and
    197 of 963 calls in this corpus are strings. The chat template runs |items
    over them and dies on anything that is not a mapping. The app resolves this
    the same way at main.js: parse it, and let an unparseable string degrade to
    {} rather than failing the request.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def to_hf_messages(context):
    """Ollama storage shape -> what the chat template expects."""
    msgs = []
    for m in context:
        role = m.get("role")
        if role == "tool":
            msgs.append({"role": "tool", "content": str(m.get("content") or "")})
        elif role == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [{
                    "type": "function",
                    "function": {
                        "name": (c.get("function") or {}).get("name") or "",
                        "arguments": as_object((c.get("function") or {}).get("arguments")),
                    },
                } for c in calls]
            msgs.append(msg)
        elif role in ("user", "system"):
            msgs.append({"role": role, "content": str(m.get("content") or "")})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--defs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--think", action="store_true", help="leave thinking enabled")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import jsonschema
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    defs = json.load(open(args.defs, encoding="utf-8"))
    schemas = {d["function"]["name"]: (d["function"].get("parameters") or {}) for d in defs}
    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    prompts, sent_names = [], []
    for r in rows:
        kwargs = {} if args.think else {"enable_thinking": False}
        # Send the MCP tools this conversation had connected alongside the 55,
        # or the model is penalised for calling something it can see was there.
        sent = defs + (r.get("extra_tools") or [])
        prompts.append(tok.apply_chat_template(
            to_hf_messages(r["context"]), tools=sent,
            add_generation_prompt=True, tokenize=False, **kwargs))
        sent_names.append({d["function"]["name"] for d in sent})

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, enforce_eager=True, max_num_seqs=4,
              max_num_batched_tokens=2048, attention_backend="TRITON_ATTN")
    outs = llm.generate(prompts, SamplingParams(max_tokens=768, temperature=0.0))

    tally = collections.Counter()
    per_tool = collections.defaultdict(collections.Counter)
    records = []
    for r, o, offered in zip(rows, outs, sent_names):
        text = o.outputs[0].text
        calls = parse_calls(text)
        ref = r["reference"]["tool_names"][0]
        # Keep the raw text on every record, not just failures. A scoring bug
        # cost a whole rerun once (complex arguments left as strings); with the
        # generation saved, a fix is a rescore instead of another GPU hour.
        rec = {"chat_id": r["chat_id"], "reference": ref,
               "emitted": [c["name"] for c in calls], "raw": text}

        tally["total"] += 1
        per_tool[ref]["total"] += 1
        if calls:
            tally["emitted_call"] += 1
            per_tool[ref]["emitted_call"] += 1
            first = calls[0]
            if first["name"] in offered:
                tally["name_valid"] += 1
                per_tool[ref]["name_valid"] += 1
                schema = schemas.get(first["name"])
                if schema is None:
                    tally["schema_uncheckable"] += 1
                    schema = {}
                required = schema.get("required") or []
                if all(k in first["arguments"] for k in required):
                    tally["required_present"] += 1
                try:
                    jsonschema.validate(first["arguments"], schema)
                    tally["schema_valid"] += 1
                    per_tool[ref]["schema_valid"] += 1
                except jsonschema.ValidationError as e:
                    rec["schema_error"] = str(e).splitlines()[0][:120]
                except Exception:
                    tally["schema_uncheckable"] += 1
            else:
                rec["invalid_name"] = first["name"]
            if first["name"] == ref:
                tally["name_match"] += 1
                per_tool[ref]["name_match"] += 1
        records.append(rec)

    total = max(1, tally["total"])
    summary = {
        "label": args.label or args.model,
        "model": args.model,
        "thinking": bool(args.think),
        "n": tally["total"],
        "rates": {k: round(100 * tally[k] / total, 1) for k in
                  ("emitted_call", "name_valid", "required_present",
                   "schema_valid", "name_match")},
        "counts": dict(tally),
    }
    print(json.dumps(summary["rates"], indent=2))
    print("\nn = %d" % tally["total"])
    print("\nper reference tool (emitted / valid name / schema ok / matched):")
    for name in sorted(per_tool):
        c = per_tool[name]
        print("   %-28s %2d  %2d  %2d  %2d   of %2d"
              % (name, c["emitted_call"], c["name_valid"], c["schema_valid"],
                 c["name_match"], c["total"]))

    json.dump({"summary": summary, "records": records},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
    # vLLM 0.28's engine teardown does not return on this setup: generate()
    # completes, results are written, and the process then sits holding the
    # GPU session forever. Twice this stalled a run whose numbers were
    # already on disk. Everything is flushed by now, so leave without it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
