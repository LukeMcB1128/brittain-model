# -*- coding: utf-8 -*-
"""Pull chat_exchanges out of a D1 backup into JSONL for grading.

`npm run db:backup` writes the whole database as SQL: users, sessions,
accounts and verification rows alongside the exchanges. Only the exchanges
are wanted here, and the rest should not be copied around, so this restores
the dump into an in-memory database -- the same trick backup-db.mjs uses to
verify it -- and writes out only what is needed to grade the model.

Nothing it writes belongs in the repository. The output carries whatever
people typed into the chat.

    python3 extract_exchanges.py ../../site/backups/brittain-<ts>.sql \\
        --model run3-step-0116 --out exchanges.jsonl
"""
import argparse
import io
import json
import os
import sqlite3
import sys


def load(path):
    """Restore the dump in memory. Never against a real database."""
    db = sqlite3.connect(":memory:")
    db.executescript(io.open(path, encoding="utf-8").read())
    names = {row[0] for row in
             db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "chat_exchanges" not in names:
        raise SystemExit("no chat_exchanges table in %s (found: %s)"
                         % (path, ", ".join(sorted(names)) or "nothing"))
    return db


def rows(db, model=None, limit=None):
    sql = "SELECT id, created_at, model, payload FROM chat_exchanges"
    args = []
    if model:
        sql += " WHERE model = ?"
        args.append(model)
    sql += " ORDER BY created_at DESC"
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    for row_id, created, model_name, payload in db.execute(sql, args):
        try:
            body = json.loads(payload)
        except ValueError:
            # A truncated row is worth knowing about, not worth dying over.
            sys.stderr.write("unparseable payload on %s\n" % row_id)
            continue
        yield {
            "id": row_id,
            "at": created,
            "model": model_name,
            "messages": body.get("messages") or [],
            "reply": body.get("reply") or "",
            "tools": body.get("tools") or [],
            "finish_reason": body.get("finishReason"),
            "usage": body.get("usage"),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--model", default=None,
                        help="only this served model, e.g. run3-step-0116")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="exchanges.jsonl")
    args = parser.parse_args()

    db = load(args.dump)
    written = with_tools = 0
    models = {}
    # 0600: the rest of the dump is private and so is this.
    handle = io.open(args.out, "w", encoding="utf-8")
    try:
        os.chmod(args.out, 0o600)
    except OSError:
        pass
    for record in rows(db, args.model, args.limit):
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        with_tools += bool(record["tools"])
        models[record["model"]] = models.get(record["model"], 0) + 1
    handle.close()

    print("wrote %d exchanges to %s" % (written, args.out))
    print("  %d called at least one tool" % with_tools)
    for name, count in sorted(models.items(), key=lambda kv: -kv[1]):
        print("  %-24s %d" % (name, count))


if __name__ == "__main__":
    main()
