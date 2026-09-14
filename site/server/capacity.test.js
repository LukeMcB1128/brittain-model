import test from 'node:test';
import assert from 'node:assert/strict';
import { ChatCapacity, internalChatRequest, relayResponse } from './capacity.js';

test('capacity gate rejects calls without a verified internal user', async () => {
  const gate = new ChatCapacity({ waitUntil() {} }, { CHAT_MAX_CONCURRENT: '1' });
  const response = await gate.fetch(new Request('https://site.example/api/chat'));
  assert.equal(response.status, 401);
});

test('capacity gate limits one response per user and the configured global total', async () => {
  const gate = new ChatCapacity({ waitUntil() {} }, { CHAT_MAX_CONCURRENT: '1' });
  gate.activeUsers.set('user-1', Date.now() + 60_000);
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
  gate.activeUsers.set('user-1', Date.now() - 1);
  gate.purgeExpired();
  assert.equal(gate.activeUsers.has('user-1'), false);
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
