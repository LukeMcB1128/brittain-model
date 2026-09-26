# -*- coding: utf-8 -*-
"""The recorder against a fake vLLM.  python -m unittest recorder_test"""
import asyncio
import glob
import json
import os
import tempfile
import unittest

from aiohttp import ClientSession, web

import record_sessions
import recorder


def sse(*chunks):
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"


STREAM = [
    {"choices": [{"index": 0, "delta": {"reasoning": "Check the file first."}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": "}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"a.py\"}"}}]}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
]


class Fake:
    """vLLM stand-in. /v1/chat/completions streams STREAM; /slow waits to be released."""

    def __init__(self):
        self.release = asyncio.Event()
        self.aborted = asyncio.Event()
        self.seen = []

    async def chat(self, request):
        self.seen.append((dict(request.headers), await request.json()))
        body = await request.json()
        if not body.get("stream"):
            return web.json_response({"choices": [{"index": 0, "message": {"content": "Plain.", "reasoning": "r"},
                                                   "finish_reason": "stop"}]})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        payload = sse(*STREAM)
        await response.write(payload[:40])
        if body.get("hold"):
            try:
                await self.release.wait()
                for _ in range(200):
                    await response.write(b"data: {}\n\n")
                    await asyncio.sleep(0.01)
            except (ConnectionResetError, asyncio.CancelledError):
                self.aborted.set()
                raise
        await response.write(payload[40:])
        await response.write_eof()
        return response

    async def health(self, request):
        return web.Response(text="ok")

    async def denied(self, request):
        return web.json_response({"error": "bad key"}, status=401)


class RecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = Fake()
        upstream = web.Application()
        upstream.router.add_post("/v1/chat/completions", self.fake.chat)
        upstream.router.add_get("/health", self.fake.health)
        upstream.router.add_post("/v1/denied/completions", self.fake.denied)
        self.upstream_runner = web.AppRunner(upstream, handler_cancellation=True)
        await self.upstream_runner.setup()
        site = web.TCPSite(self.upstream_runner, "127.0.0.1", 0)
        await site.start()
        upstream_port = site._server.sockets[0].getsockname()[1]
        self.app = recorder.make_app("http://127.0.0.1:%d" % upstream_port, self.tmp.name)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base = "http://127.0.0.1:%d" % site._server.sockets[0].getsockname()[1]
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.runner.cleanup()
        await self.upstream_runner.cleanup()
        self.tmp.cleanup()

    async def settle(self):
        await asyncio.sleep(0.05)
        if self.app[recorder.PENDING]:
            await asyncio.gather(*list(self.app[recorder.PENDING]))

    def exchanges(self):
        rows = []
        for path in glob.glob(os.path.join(self.tmp.name, "exchanges-*.jsonl")):
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle)
        return rows

    async def post(self, body, headers=None, path="/v1/chat/completions"):
        headers = dict({"Cf-Ray": "abc-LHR", "Authorization": "Bearer key"}, **(headers or {}))
        async with self.client.post(self.base + path, json=body, headers=headers) as response:
            return response.status, await response.read()

    async def test_stream_passes_through_unchanged_and_is_assembled(self):
        body = {"model": "run5b-step-0124", "stream": True, "temperature": 0.6,
                "messages": [{"role": "system", "content": "You are BRITTAIN-4."},
                             {"role": "user", "content": "Read a.py"}],
                "tools": [{"type": "function", "function": {"name": "read_file"}}]}
        status, raw = await self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(raw, sse(*STREAM))
        self.assertEqual(self.fake.seen[0][0]["Authorization"], "Bearer key")
        await self.settle()
        [row] = self.exchanges()
        self.assertEqual(row["surface"], "api")
        self.assertEqual(row["model"], "run5b-step-0124")
        self.assertEqual(row["sampling"], {"model": "run5b-step-0124", "temperature": 0.6})
        self.assertEqual(row["messages"], [{"role": "user", "content": "Read a.py"}])
        reply = row["reply"]
        self.assertEqual(reply["reasoning"], "Check the file first.")
        self.assertEqual(reply["tool_calls"], [{"id": "call_1", "type": "function",
                                                "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"}}])
        self.assertEqual(reply["finish_reason"], "tool_calls")
        self.assertEqual(reply["usage"]["completion_tokens"], 5)
        prompt_file = os.path.join(self.tmp.name, "prompts", row["prompt"] + ".json")
        with open(prompt_file, encoding="utf-8") as handle:
            prompt = json.load(handle)
        self.assertEqual(prompt["system"][0]["content"], "You are BRITTAIN-4.")
        self.assertEqual(prompt["tools"][0]["function"]["name"], "read_file")

    async def test_first_chunk_arrives_before_the_stream_ends(self):
        body = {"stream": True, "hold": True, "messages": [{"role": "user", "content": "hi"}]}
        async with self.client.post(self.base + "/v1/chat/completions", json=body,
                                    headers={"Cf-Ray": "x"}) as response:
            first = await asyncio.wait_for(response.content.read(10), timeout=2)
            self.assertTrue(first.startswith(b"data: "))
            self.fake.release.set()
            await response.read()

    async def test_only_tunnel_traffic_is_recorded(self):
        body = {"messages": [{"role": "user", "content": "local eval"}]}
        async with self.client.post(self.base + "/v1/chat/completions", json=body) as response:
            self.assertEqual(await response.json(), {"choices": [{"index": 0, "message": {"content": "Plain.", "reasoning": "r"},
                                                                  "finish_reason": "stop"}]})
        await self.post(body, {"X-Brittain-Record": "off"})
        self.assertNotIn("X-Brittain-Record", self.fake.seen[-1][0])
        await self.post(dict(body, messages=[{"role": "user", "content": "from the tunnel"}]),
                        {"X-Brittain-Surface": "web-chat"})
        await self.settle()
        [row] = self.exchanges()
        self.assertEqual(row["surface"], "web-chat")
        self.assertEqual(row["reply"]["content"], "Plain.")

    async def test_secrets_and_attachments_are_not_kept(self):
        key = "sk-" + "a" * 30
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "my .env has OPENAI_API_KEY=%s and password: hunter22" % key},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}}]}]}
        await self.post(body)
        await self.settle()
        text = json.dumps(self.exchanges())
        self.assertNotIn(key, text)
        self.assertNotIn("hunter22", text)
        self.assertNotIn("AAAAAAAA", text)
        self.assertIn("[image/png omitted]", text)
        # vLLM still got the real request.
        self.assertIn(key, json.dumps(self.fake.seen[-1][1]))

    async def test_other_routes_and_errors_pass_through(self):
        async with self.client.get(self.base + "/health") as response:
            self.assertEqual((response.status, await response.text()), (200, "ok"))
        status, raw = await self.post({"messages": []}, path="/v1/denied/completions")
        self.assertEqual((status, json.loads(raw)), (401, {"error": "bad key"}))
        await self.settle()
        [row] = self.exchanges()
        self.assertEqual(row["status"], 401)

    async def test_vllm_down_is_a_clear_502(self):
        app = recorder.make_app("http://127.0.0.1:9", self.tmp.name)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with self.client.post("http://127.0.0.1:%d/v1/chat/completions" % port, json={}) as response:
                self.assertEqual(response.status, 502)
                self.assertEqual((await response.json())["error"]["type"], "upstream_unavailable")
        finally:
            await runner.cleanup()

    async def test_a_client_that_leaves_stops_the_generation(self):
        body = {"stream": True, "hold": True, "messages": [{"role": "user", "content": "long one"}]}
        response = await self.client.post(self.base + "/v1/chat/completions", json=body, headers={"Cf-Ray": "x"})
        await response.content.read(10)
        response.close()
        self.fake.release.set()
        await asyncio.wait_for(self.fake.aborted.wait(), timeout=5)
        await self.settle()
        [row] = self.exchanges()
        self.assertEqual(row["outcome"], "client_disconnected")


class SessionTest(unittest.TestCase):
    def exchange(self, id, at, messages, prompt="p", surface="api"):
        return {"id": id, "at": at, "prompt": prompt, "surface": surface, "messages": messages,
                "reply": {"content": id}}

    def test_agent_steps_fold_into_one_session(self):
        user = {"role": "user", "content": "fix the test"}
        call = {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}
        result = {"role": "tool", "tool_call_id": "1", "content": "FAILED"}
        other = {"role": "user", "content": "unrelated question"}
        sessions = record_sessions.fold([
            self.exchange("a1", "2026-09-26T10:00:00", [user]),
            self.exchange("b1", "2026-09-26T10:00:01", [other]),
            self.exchange("a2", "2026-09-26T10:00:02", [user, call, result]),
            self.exchange("c1", "2026-09-26T10:00:03", [user], prompt="another prompt"),
        ])
        self.assertEqual([(s["id"], s["steps"], s["reply"]["content"]) for s in sessions],
                         [("a1", 2, "a2"), ("b1", 1, "b1"), ("c1", 1, "c1")])
        self.assertEqual(sessions[0]["messages"], [user, call, result])
        self.assertTrue(all(s["split"] in ("train", "heldout") for s in sessions))
        again = record_sessions.fold([self.exchange("a1", "t", [user])])
        self.assertEqual(again[0]["split"], sessions[0]["split"])


if __name__ == "__main__":
    unittest.main()
