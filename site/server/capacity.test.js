import test from 'node:test';
import assert from 'node:assert/strict';
import { ChatCapacity, internalChatRequest, relayResponse } from './capacity.js';
import { handleApi } from './gateway.js';

test('capacity gate rejects calls without a verified internal user', async () => {
  const gate = new ChatCapacity({ waitUntil() {} }, { CHAT_MAX_CONCURRENT: '1' });
  const response = await gate.fetch(new Request('https://site.example/api/chat'));
  assert.equal(response.status, 401);
});

test('capacity gate limits one response per user and the configured global total', async () => {
  const gate = new ChatCapacity({ waitUntil() {} }, { CHAT_MAX_CONCURRENT: '1' });
  gate.activeUsers.set('user-1', { expiresAt: Date.now() + 60_000 });
  const duplicate = await gate.fetch(internalChatRequest(new Request('https://site.example/api/chat'), 'user-1'));
  assert.equal(duplicate.status, 409);
  const full = await gate.fetch(internalChatRequest(new Request('https://site.example/api/chat'), 'user-2'));
  assert.equal(full.status, 503);
  assert.equal(full.headers.get('Retry-After'), '10');
});

test('canceling the relayed response cancels its source and releases capacity', async () => {
  let sourceCancelled = false;
  let released = false;
  const source = new ReadableStream({
    cancel() { sourceCancelled = true; },
  });
  const response = relayResponse(new Response(source), () => { released = true; });
  await response.body.cancel('stopped');
  assert.equal(sourceCancelled, true);
  assert.equal(released, true);
});

test('an expired response lock cannot block an account', () => {
  const gate = new ChatCapacity({ waitUntil() {} }, {});
  let released = false;
  const stop = new AbortController();
  gate.activeUsers.set('user-1', { expiresAt: Date.now() - 1, stop, release() { released = true; gate.activeUsers.delete('user-1'); } });
  gate.purgeExpired();
  assert.equal(gate.activeUsers.has('user-1'), false);
  assert.equal(stop.signal.aborted, true);
  assert.equal(released, true);
});

test('a stop request aborts the active model call and frees the account slot', async () => {
  let upstreamSignal;
  const gate = new ChatCapacity({}, {}, async request => {
    upstreamSignal = request.signal;
    return new Response(new ReadableStream({
      start(controller) {
        request.signal.addEventListener('abort', () => controller.error(new DOMException('Stopped', 'AbortError')), { once: true });
      },
    }), { headers: { 'Content-Type': 'text/event-stream' } });
  });
  gate.reserveHourly = async () => ({ allowed: true });
  const id = crypto.randomUUID();
  const chat = internalChatRequest(new Request('https://site.example/api/chat', { method: 'POST', headers: { 'x-brittain-request-id': id } }), 'user-1');
  const response = await gate.fetch(chat);
  assert.equal(response.status, 200);
  const wrongStop = internalChatRequest(new Request('https://site.example/api/chat/cancel', { method: 'POST', headers: { 'x-brittain-request-id': crypto.randomUUID() } }), 'user-1');
  assert.equal((await gate.fetch(wrongStop)).status, 204);
  assert.equal(upstreamSignal.aborted, false);
  assert.equal(gate.activeUsers.size, 1);
  const stop = internalChatRequest(new Request('https://site.example/api/chat/cancel', { method: 'POST', headers: { 'x-brittain-request-id': id } }), 'user-1');
  assert.equal((await gate.fetch(stop)).status, 204);
  assert.equal(upstreamSignal.aborted, true);
  assert.equal(gate.activeUsers.size, 0);
  const next = await gate.fetch(internalChatRequest(new Request('https://site.example/api/chat', { method: 'POST' }), 'user-1'));
  assert.equal(next.status, 200);
  await next.body.cancel();
});

test('stop releases a slot before the model stream starts', async () => {
  let started;
  const ready = new Promise(resolve => { started = resolve; });
  const gate = new ChatCapacity({}, {}, async request => {
    started();
    await new Promise(resolve => request.signal.addEventListener('abort', resolve, { once: true }));
    return new Response('late reply');
  });
  gate.reserveHourly = async () => ({ allowed: true });
  const id = crypto.randomUUID();
  const pending = gate.fetch(internalChatRequest(new Request('https://site.example/api/chat', { method: 'POST', headers: { 'x-brittain-request-id': id } }), 'user-1'));
  await ready;
  const stop = await gate.fetch(internalChatRequest(new Request('https://site.example/api/chat/cancel', { method: 'POST', headers: { 'x-brittain-request-id': id } }), 'user-1'));
  assert.equal(stop.status, 204);
  assert.equal(gate.activeUsers.size, 0);
  assert.equal((await pending).status, 499);
});

test('a stop that arrives first prevents the matching chat from starting', async () => {
  let started = false;
  const gate = new ChatCapacity({}, {}, async () => { started = true; return new Response('reply'); });
  gate.reserveHourly = async () => ({ allowed: true });
  const id = crypto.randomUUID();
  const stop = internalChatRequest(new Request('https://site.example/api/chat/cancel', { method: 'POST', headers: { 'x-brittain-request-id': id } }), 'user-1');
  assert.equal((await gate.fetch(stop)).status, 204);
  const chat = internalChatRequest(new Request('https://site.example/api/chat', { method: 'POST', headers: { 'x-brittain-request-id': id } }), 'user-1');
  assert.equal((await gate.fetch(chat)).status, 499);
  assert.equal(started, false);
  assert.equal(gate.activeUsers.size, 0);
});

test('the stop route reaches the model request through the gateway', async () => {
  let upstreamSignal;
  let started;
  const ready = new Promise(resolve => { started = resolve; });
  const upstream = async (_url, options) => {
    upstreamSignal = options.signal;
    started();
    return new Response(new ReadableStream({
      start(controller) {
        options.signal.addEventListener('abort', () => controller.error(new DOMException('Stopped', 'AbortError')), { once: true });
      },
    }), { headers: { 'Content-Type': 'text/event-stream' } });
  };
  const gate = new ChatCapacity({}, { BRITTAIN4_API_KEY: 'test-only-secret' }, (request, env, _fetch, user) => handleApi(request, env, upstream, user));
  gate.reserveHourly = async () => ({ allowed: true });
  const id = crypto.randomUUID();
  const chat = internalChatRequest(new Request('https://site.example/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'x-brittain-request-id': id },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'Explain this task.' }] }),
  }), 'user-1');
  const response = await gate.fetch(chat);
  assert.equal(response.status, 200);
  await ready;
  const stop = internalChatRequest(new Request('https://site.example/api/chat/cancel', { method: 'POST', headers: { 'x-brittain-request-id': id } }), 'user-1');
  assert.equal((await gate.fetch(stop)).status, 204);
  assert.equal(upstreamSignal.aborted, true);
  assert.equal(gate.activeUsers.size, 0);
});

test('hourly capacity is stored and survives a new gate instance', async () => {
  const values = new Map();
  const state = {
    waitUntil() {},
    blockConcurrencyWhile: callback => callback(),
    storage: {
      get: async key => values.get(key),
      put: async (key, value) => values.set(key, value),
    },
  };
  const first = new ChatCapacity(state, {});
  assert.equal((await first.reserveHourly(1)).allowed, true);
  const restarted = new ChatCapacity(state, {});
  const full = await restarted.reserveHourly(1);
  assert.equal(full.allowed, false);
  assert.ok(full.retryAfter > 0);
});

test('parallel requests cannot bypass the user or global limit during storage writes', async () => {
  const gate = new ChatCapacity({}, { CHAT_MAX_CONCURRENT: '1' });
  let resolve;
  gate.reserveHourly = () => new Promise(done => { resolve = done; });
  const first = gate.fetch(internalChatRequest(new Request('https://site.example/api/chat', { method: 'POST' }), 'user-1'));
  assert.equal((await gate.fetch(internalChatRequest(new Request('https://site.example/api/chat'), 'user-1'))).status, 409);
  assert.equal((await gate.fetch(internalChatRequest(new Request('https://site.example/api/chat'), 'user-2'))).status, 503);
  resolve({ allowed: false, retryAfter: 60 });
  assert.equal((await first).status, 503);
  assert.equal(gate.activeUsers.size, 0);
});

test('a failed capacity write and an already canceled request release the slot', async () => {
  const gate = new ChatCapacity({}, {});
  gate.reserveHourly = async () => { throw new Error('storage unavailable'); };
  await assert.rejects(gate.fetch(internalChatRequest(new Request('https://site.example/api/chat'), 'user-1')), /storage unavailable/);
  assert.equal(gate.activeUsers.size, 0);
  const controller = new AbortController();
  controller.abort();
  const canceled = await gate.fetch(internalChatRequest(new Request('https://site.example/api/chat', { signal: controller.signal }), 'user-1'));
  assert.equal(canceled.status, 499);
  assert.equal(gate.activeUsers.size, 0);
});

test('finishing an expired request does not remove a replacement slot', async () => {
  const gate = new ChatCapacity({}, {});
  let resolve;
  gate.reserveHourly = () => new Promise(done => { resolve = done; });
  const first = gate.fetch(internalChatRequest(new Request('https://site.example/api/chat'), 'user-1'));
  const replacement = { expiresAt: Date.now() + 900000 };
  gate.activeUsers.set('user-1', replacement);
  resolve({ allowed: false, retryAfter: 60 });
  await first;
  assert.equal(gate.activeUsers.get('user-1'), replacement);
});
