import test from 'node:test';
import assert from 'node:assert/strict';
import { readCompletion } from './brittain4.js';
const sse = text => new Response(text, { headers: { 'Content-Type': 'text/event-stream' } });
test('SSE handles split UTF-8, empty choice frames, usage and DONE', async () => {
  const text = 'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\r\n\r\ndata: {"choices":[{"index":0,"delta":{"content":"Hi 🌍"}}]}\n\ndata: {"choices":[],"usage":{"total_tokens":12}}\n\ndata: [DONE]\n\n';
  const bytes = new TextEncoder().encode(text);
  const stream = new ReadableStream({ start(c) { for (const byte of bytes) c.enqueue(new Uint8Array([byte])); c.close(); } });
  const chunks = [];
  await readCompletion(new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } }), c => chunks.push(c));
  assert.equal(chunks.map(c => c.text).join(''), 'Hi 🌍');
  assert.equal(chunks.at(-1).usage.total_tokens, 12);
});
test('premature EOF keeps failure visible', async () => {
  await assert.rejects(readCompletion(sse('data: {"choices":[]}\n\n'), () => {}), /before the reply finished/);
});
test('both API error shapes and tunnel HTML fail clearly', async () => {
  for (const error of ['Unauthorized', { message: 'Context too long' }]) {
    await assert.rejects(readCompletion(Response.json({ error }, { status: 400 }), () => {}), new RegExp(typeof error === 'string' ? error : error.message));
  }
  await assert.rejects(readCompletion(new Response('<html/>'), () => {}), /unexpected response/);
});
test('finish length and inline stream errors are handled', async () => {
  const chunks = [];
  await readCompletion(sse('data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n'), c => chunks.push(c));
  assert.equal(chunks[0].finishReason, 'length');
  await assert.rejects(readCompletion(sse('data: {"error":{"message":"Failed"}}\n'), () => {}), /Failed/);
});
