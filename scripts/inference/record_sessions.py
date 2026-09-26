# -*- coding: utf-8 -*-
"""Fold recorded exchanges into one record per conversation.

The recorder writes one line per request. An agent loop -- and the web chat's
tool rounds -- resend the whole conversation on every step, so a 30-step
Brittain Code session is 30 lines, each containing the one before. A request
continues a session when the session's messages are a prefix of its own, under
the same system prompt and surface. Each session keeps its longest exchange:
the full trajectory as the client replayed it, plus the final reply.

Sessions are split once, by a hash of their first exchange, into train and
held-out. The split is fixed at creation so a held-out conversation never
drifts into training data when the file is regenerated -- the probe
contamination of run 4 is the reason.

    python record_sessions.py                       # all days -> sessions.jsonl
    python record_sessions.py --records DIR --out FILE
"""
import argparse
import glob
import hashlib
import json
import os

HELDOUT_PERCENT = 10


def turn_key(message):
    return json.dumps({key: message.get(key) for key in ("role", "content", "tool_calls", "tool_call_id")},
                      sort_keys=True)


def fold(exchanges):
    """Exchanges in time order -> sessions, each ending at its longest exchange."""
    open_sessions = []
    for exchange in sorted(exchanges, key=lambda item: item["at"]):
        turns = [turn_key(message) for message in exchange.get("messages") or []]
        best = None
        for session in open_sessions:
            if (session["prompt"] == exchange.get("prompt") and session["surface"] == exchange.get("surface")
                    and len(session["turns"]) <= len(turns) and turns[:len(session["turns"])] == session["turns"]
                    and (best is None or len(session["turns"]) > len(best["turns"]))):
                best = session
        if best is None:
            best = {"prompt": exchange.get("prompt"), "surface": exchange.get("surface"),
                    "first": exchange, "exchanges": []}
            open_sessions.append(best)
        best["turns"] = turns
        best["last"] = exchange
        best["exchanges"].append(exchange["id"])
    sessions = []
    for session in open_sessions:
        first, last = session["first"], session["last"]
        bucket = int(hashlib.sha256(first["id"].encode("utf-8")).hexdigest(), 16) % 100
        sessions.append({
            "id": first["id"],
            "split": "heldout" if bucket < HELDOUT_PERCENT else "train",
            "surface": session["surface"],
            "user_agent": last.get("user_agent"),
            "model": last.get("model"),
            "prompt": session["prompt"],
            "started": first["at"],
            "ended": last["at"],
            "steps": len(session["exchanges"]),
            "exchanges": session["exchanges"],
            "outcome": last.get("outcome"),
            "messages": last.get("messages") or [],
            "reply": last.get("reply") or {},
        })
    return sessions


def load(records):
    exchanges = []
    for path in sorted(glob.glob(os.path.join(records, "exchanges-*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        exchanges.append(json.loads(line))
                    except ValueError:
                        continue  # a torn last line from a crash
    return exchanges


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", default=os.path.expanduser("~/brittain4/records"))
    parser.add_argument("--out", default=None, help="default: <records>/sessions.jsonl")
    args = parser.parse_args()
    exchanges = load(args.records)
    sessions = fold(exchanges)
    out = args.out or os.path.join(args.records, "sessions.jsonl")
    handle = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        for session in sessions:
            stream.write(json.dumps(session, ensure_ascii=False) + "\n")
    by_surface = {}
    for session in sessions:
        by_surface[session["surface"]] = by_surface.get(session["surface"], 0) + 1
    print("%d exchanges -> %d sessions %s -> %s" % (len(exchanges), len(sessions), by_surface, out))


if __name__ == "__main__":
    main()
