import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEntry, recordExchange, redact } from './transcript-log.js';

const exchange = {
  user: 'user-123',
  messages: [
    { role: 'system', content: 'You are BRITTAIN...' },
    { role: 'user', content: 'What is the weather in London?' },
    { role: 'assistant', content: 'Let me check.' },
  ],
  reply: 'It is 12°C and raining.',
  usage: { prompt_tokens: 120, completion_tokens: 40, total_tokens: 160 },
  finishReason: 'stop',
  startedAt: Date.now() - 1500,
};

test('an entry records the exchange without the system prompt', async () => {
  const entry = await buildEntry(exchange);
  assert.equal(entry.surface, 'web-chat');
  assert.equal(entry.model, 'brittain4');
  assert.equal(entry.finishReason, 'stop');
  assert.equal(entry.reply, 'It is 12°C and raining.');
  assert.deepEqual(entry.messages.map(m => m.role), ['user', 'assistant']);
  assert.ok(entry.durationMs >= 1000);
  assert.match(entry.at, /^\d{4}-\d{2}-\d{2}T/);
});

test('the user is grouped by digest, never stored by identity', async () => {
  const a = await buildEntry(exchange);
  const b = await buildEntry({ ...exchange, user: 'user-123' });
  const other = await buildEntry({ ...exchange, user: 'someone-else' });
  assert.equal(a.user, b.user, 'the same person must group together');
  assert.notEqual(a.user, other.user);
  assert.equal(JSON.stringify(a).includes('user-123'), false, 'the raw id must not be stored');
});

test('credentials are redacted before anything is written', () => {
  assert.match(redact('my key is AIzaBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'), /REDACTED_GOOGLE_API_KEY/);
  assert.match(redact('token ghp_abcdefghijklmnopqrstuvwxyz01'), /REDACTED_TOKEN/);
  assert.match(redact('password: hunter2'), /password: \[REDACTED\]/);
  assert.match(redact('Authorization: Bearer abcdefghijklmnopqrstuvwxyz'), /Bearer \[REDACTED_TOKEN\]/);
  assert.equal(redact('nothing sensitive here'), 'nothing sensitive here');
});

test('attachment bytes are counted, not stored', async () => {
  const entry = await buildEntry({
    ...exchange,
    messages: [{ role: 'user', content: [
      { type: 'text', text: 'What is in this picture?' },
      { type: 'image_url', image_url: { url: `data:image/png;base64,${'A'.repeat(200_000)}` } },
    ] }],
  });
  const serialized = JSON.stringify(entry);
  assert.equal(entry.messages[0].images, 1);
  assert.match(entry.messages[0].text, /What is in this picture/);
  assert.equal(serialized.includes('AAAAAAAAAA'), false, 'base64 payloads must never reach the store');
  assert.ok(serialized.length < 5_000);
});

test('a KV-shaped binding receives one chronologically sorted key', async () => {
  const written = [];
  const env = { CHAT_LOG: { put: async (key, value) => written.push([key, value]) } };
  assert.equal(await recordExchange(env, exchange), true);
  assert.equal(written.length, 1);
  assert.match(written[0][0], /^chat\/\d{4}-\d{2}-\d{2}T.*\/[0-9a-f-]{36}$/);
  assert.equal(JSON.parse(written[0][1]).reply, 'It is 12°C and raining.');
});

test('an HTTP sink is posted to, with the token when one is set', async () => {
  const seen = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => { seen.push({ url, options }); return new Response('', { status: 200 }); };
  try {
    const ok = await recordExchange({ CHAT_LOG_URL: 'https://sink.example/log', CHAT_LOG_TOKEN: 'secret' }, exchange);
    assert.equal(ok, true);
    assert.equal(seen[0].url, 'https://sink.example/log');
    assert.equal(seen[0].options.headers.Authorization, 'Bearer secret');
    assert.equal(JSON.parse(seen[0].options.body).finishReason, 'stop');
  } finally { globalThis.fetch = originalFetch; }
});

test('with nothing bound it does nothing at all', async () => {
  assert.equal(await recordExchange({}, exchange), false);
  assert.equal(await recordExchange(undefined, exchange), false);
});

test('a failing store is swallowed rather than thrown at the request', async () => {
  const env = { CHAT_LOG: { put: async () => { throw new Error('KV is down'); } } };
  assert.equal(await recordExchange(env, exchange), false);
  const bad = { CHAT_LOG_URL: 'https://sink.example/log' };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('network down'); };
  try { assert.equal(await recordExchange(bad, exchange), false); }
  finally { globalThis.fetch = originalFetch; }
});

test('an oversized exchange is dropped rather than stored', async () => {
  const written = [];
  const env = { CHAT_LOG: { put: async (key, value) => written.push([key, value]) } };
  const huge = { ...exchange, messages: Array.from({ length: 60 }, () => ({ role: 'user', content: 'x'.repeat(19_000) })) };
  assert.equal(await recordExchange(env, huge), false);
  assert.equal(written.length, 0);
});

test('a failed exchange is recorded, with its partial reply', async () => {
  const entry = await buildEntry({
    ...exchange, reply: 'I was part way through', finishReason: undefined,
    error: 'The model took too long to respond.',
  });
  assert.match(entry.error, /took too long/);
  assert.equal(entry.reply, 'I was part way through');
  assert.equal(entry.finishReason, null);
});

test('tool calls are recorded with arguments and outcome', async () => {
  const entry = await buildEntry({
    ...exchange,
    toolCalls: [
      { name: 'web_search', arguments: '{"query":"weather london"}', ok: true, result: '1. Met Office...' },
      { name: 'web_fetch', arguments: { url: 'https://example.com' }, ok: false, result: 'Error: blocked' },
    ],
  });
  assert.deepEqual(entry.tools.map(t => t.name), ['web_search', 'web_fetch']);
  assert.equal(entry.tools[0].ok, true);
  assert.equal(entry.tools[1].ok, false);
  assert.match(entry.tools[1].arguments, /example\.com/);
});
