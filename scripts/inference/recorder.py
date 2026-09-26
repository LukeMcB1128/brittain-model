# -*- coding: utf-8 -*-
"""Record what BRITTAIN-4 is asked and what it answers, from the traffic itself.

The web chat logs its exchanges to D1 from the gateway. Brittain Code talks to
vLLM through the Cloudflare tunnel directly, so its sessions -- the agentic
ones, where the worst failures happen -- were kept nowhere. This sits between
the tunnel and vLLM and keeps a copy of both surfaces in one format:

    tunnel -> recorder :11435 -> vLLM 127.0.0.1:11436

Every chunk is written back to the client the moment it arrives; the copy is
assembled after the stream ends. Nothing about recording is allowed to delay or
break a reply: a failed write is dropped, not retried.

WHAT IS RECORDED
Only POSTs to .../completions that came through Cloudflare (they carry a
cf-ray header). Eval scripts on this machine hit the same port and must not end
up as "real usage" training data. A client can opt out of a request with
X-Brittain-Record: off. The gateway labels its requests with
X-Brittain-Surface; anything unlabelled is recorded as "api" with its user
agent, which is how Brittain Code shows up.

WHERE
~/brittain4/records/ (owner-only), never the repository:
    exchanges-YYYY-MM-DD.jsonl   one line per request, redacted
    prompts/<hash>.json          each distinct system prompt + tool list, once
An agent loop resends the whole conversation every step, so exchanges repeat
their prefixes; record_sessions.py folds them into one record per session.

    python recorder.py --port 11435 --upstream http://127.0.0.1:11436
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from aiohttp import ClientSession, ClientTimeout, client_exceptions, web

UPSTREAM = web.AppKey("upstream", str)
STORE = web.AppKey("store", object)
RECORD_ALL = web.AppKey("record_all", bool)
PENDING = web.AppKey("pending", set)
CLIENT = web.AppKey("client", ClientSession)

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "content-length"}
# The response is copied as raw bytes, so its content-encoding stays true.
MAX_CAPTURE = 8_000_000
MAX_STRING = 50_000
MAX_REQUEST = 64 * 1024 * 1024

# Same shapes as site/server/transcript-log.js. CLI sessions read .env files and
# print command output; a record that keeps a key outlives the session it was in.
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED_GOOGLE_API_KEY]"),
    (re.compile(r"\b(?:sk|ghp|gho|ghu|ghs|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{16,}"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\b(password|passwd|pwd|api[_-]?key|secret|access[_-]?token)(\s*[:=]\s*)(?!\[REDACTED)([^\s\"',;}\\]{4,})",
                re.IGNORECASE), r"\1\2[REDACTED]"),
]


def redact(text):
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > MAX_STRING:
        text = "%s...[%d more characters]" % (text[:MAX_STRING], len(text) - MAX_STRING)
    return text


def scrub(value):
    """Redact every string; drop attachment bytes, keep that there was one."""
    if isinstance(value, str):
        if value.startswith("data:") and len(value) > 200:
            return "[%s omitted]" % value[5:value.find(";")] if ";" in value[:80] else "[data omitted]"
        return redact(value)
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    return value


def assemble(raw, streamed):
    """The reply as the client saw it: content, reasoning and tool calls."""
    text = raw.decode("utf-8", "replace")
    if not streamed:
        try:
            body = json.loads(text)
        except ValueError:
            return {"unparsed": text[:2000]}
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {"content": message.get("content") or choice.get("text") or "",
                "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
                "tool_calls": message.get("tool_calls") or [],
                "finish_reason": choice.get("finish_reason"), "usage": body.get("usage")}
    content, reasoning, calls = [], [], {}
    finish = usage = error = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        usage = chunk.get("usage") or usage
        error = chunk.get("error") or error
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or choice.get("text") or "")
            reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0),
                                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    reply = {"content": "".join(content), "reasoning": "".join(reasoning),
             "tool_calls": [calls[index] for index in sorted(calls)], "finish_reason": finish, "usage": usage}
    if error:
        reply["error"] = error
    return reply


def surface(headers):
    label = headers.get("X-Brittain-Surface", "")
    return label if re.fullmatch(r"[a-z][a-z0-9-]{0,31}", label) else "api"


def should_record(request, record_all):
    if request.method != "POST" or not request.path.endswith("/completions"):
        return False
    if request.headers.get("X-Brittain-Record", "").lower() == "off":
        return False
    return record_all or "Cf-Ray" in request.headers


def build(request, body, raw, status, outcome, started, truncated):
    """(exchange, prompt) -- the prompt is stored once per distinct hash."""
    try:
        payload = json.loads(body)
    except ValueError:
        payload = {}
    payload = scrub(payload if isinstance(payload, dict) else {})
    messages = payload.get("messages") or []
    lead = 0
    while lead < len(messages) and isinstance(messages[lead], dict) and messages[lead].get("role") == "system":
        lead += 1
    prompt = {"system": messages[:lead], "tools": payload.get("tools")}
    prompt_hash = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if status == 200:
        reply = scrub(assemble(raw, bool(payload.get("stream"))))
    else:
        reply = {"error": redact(raw[:2000].decode("utf-8", "replace"))}
    exchange = {
        "id": uuid.uuid4().hex,
        "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "surface": surface(request.headers),
        "user_agent": request.headers.get("User-Agent", "")[:200],
        "model": payload.get("model"),
        "prompt": prompt_hash,
        "sampling": {key: value for key, value in payload.items()
                     if key not in ("messages", "tools", "stream", "stream_options", "prompt")},
        "status": status,
        "outcome": outcome,
        "truncated": truncated,
        "duration_ms": int((time.time() - started) * 1000),
        "messages": messages[lead:] if messages else [{"role": "user", "content": payload.get("prompt", "")}],
        "reply": reply,
    }
    return exchange, prompt_hash, prompt


class Store:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        os.makedirs(os.path.join(root, "prompts"), mode=0o700, exist_ok=True)

    def _append(self, path, line):
        handle = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(handle, line)
        finally:
            os.close(handle)

    def write(self, exchange, prompt_hash, prompt):
        with self.lock:
            prompt_path = os.path.join(self.root, "prompts", prompt_hash + ".json")
            if not os.path.exists(prompt_path):
                self._append(prompt_path, json.dumps(prompt, ensure_ascii=False).encode("utf-8"))
            day = exchange["at"][:10]
            line = json.dumps(exchange, ensure_ascii=False) + "\n"
            self._append(os.path.join(self.root, "exchanges-%s.jsonl" % day), line.encode("utf-8"))


async def proxy(request):
    app = request.app
    started = time.time()
    body = await request.read()
    record = should_record(request, app[RECORD_ALL])
    headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_BY_HOP}
    headers.pop("X-Brittain-Record", None)
    try:
        upstream = await app[CLIENT].request(request.method, app[UPSTREAM] + request.path_qs,
                                               headers=headers, data=body or None, allow_redirects=False)
    except (client_exceptions.ClientError, asyncio.TimeoutError, OSError):
        return web.json_response({"error": {"message": "BRITTAIN-4 is not reachable right now.",
                                             "type": "upstream_unavailable"}}, status=502)
    captured = bytearray()
    truncated = False
    outcome = "complete"
    response = None
    try:
        response = web.StreamResponse(status=upstream.status, reason=upstream.reason,
                                      headers={key: value for key, value in upstream.headers.items()
                                               if key.lower() not in HOP_BY_HOP})
        await response.prepare(request)
        async for chunk in upstream.content.iter_any():
            if record:
                if len(captured) + len(chunk) <= MAX_CAPTURE:
                    captured.extend(chunk)
                else:
                    truncated = True
            await response.write(chunk)
        await response.write_eof()
        return response
    except ConnectionResetError:
        # The client went away. Closing the upstream connection (finally,
        # below) is what tells vLLM to abort instead of finishing for nobody.
        outcome = "client_disconnected"
        return response
    except asyncio.CancelledError:
        outcome = "client_disconnected"
        raise
    except (client_exceptions.ClientError, asyncio.TimeoutError):
        outcome = "upstream_failed"
        raise
    finally:
        upstream.close()
        if record:
            remember(app, request, body, bytes(captured), upstream.status, outcome, started, truncated)


def remember(app, request, body, raw, status, outcome, started, truncated):
    def work():
        try:
            app[STORE].write(*build(request, body, raw, status, outcome, started, truncated))
        except Exception as error:  # a lost record is worth less than a reply
            print("recorder: dropped a record: %r" % error, file=sys.stderr)
    task = asyncio.get_running_loop().run_in_executor(None, work)
    app[PENDING].add(task)
    task.add_done_callback(app[PENDING].discard)


def make_app(upstream, records, record_all=False):
    app = web.Application(client_max_size=MAX_REQUEST)
    app[UPSTREAM] = upstream.rstrip("/")
    app[STORE] = Store(records)
    app[RECORD_ALL] = record_all
    app[PENDING] = set()

    async def open_client(app):
        # No read timeout: a long thinking reply streams for minutes. No
        # decompression: the bytes go back exactly as vLLM sent them.
        app[CLIENT] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10),
                                      auto_decompress=False)

    async def close_client(app):
        await app[CLIENT].close()
        if app[PENDING]:
            await asyncio.gather(*app[PENDING], return_exceptions=True)

    app.on_startup.append(open_client)
    app.on_cleanup.append(close_client)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--upstream", default="http://127.0.0.1:11436")
    parser.add_argument("--records", default=os.path.expanduser("~/brittain4/records"))
    parser.add_argument("--record-all", action="store_true",
                        help="record local requests too, not only tunnel traffic")
    args = parser.parse_args()
    os.umask(0o077)
    web.run_app(make_app(args.upstream, args.records, args.record_all),
                host=args.host, port=args.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
