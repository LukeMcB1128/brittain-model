import test from 'node:test';
import assert from 'node:assert/strict';
import worker from './worker.js';

test('maintenance mode rejects new generations before accessing the model or account database', async () => {
  const response = await worker.fetch(new Request('https://brittain.app/api/chat', { method: 'POST' }), { CHAT_ENABLED: 'false' });
  assert.equal(response.status, 503);
  assert.equal(response.headers.get('Retry-After'), '60');
  assert.match((await response.json()).error, /maintenance/);
});

test('health checks configuration and the database without querying the model', async () => {
  const env = { BETTER_AUTH_SECRET: 'test', BRITTAIN4_API_KEY: 'test', DB: { prepare(sql) {
    assert.equal(sql, 'SELECT 1 AS ok'); return { first: async () => ({ ok: 1 }) };
  } } };
  const response = await worker.fetch(new Request('https://brittain.app/api/health'), env);
  assert.deepEqual(await response.json(), { status: 'ok', chatPaused: false });
  assert.equal(response.headers.get('Cache-Control'), 'no-store');
  assert.equal(response.headers.get('X-Robots-Tag'), 'noindex, nofollow');
  assert.ok(response.headers.get('Strict-Transport-Security'));
  assert.ok(response.headers.get('Content-Security-Policy'));
  const unavailable = await worker.fetch(new Request('https://brittain.app/api/health'), {});
  assert.equal(unavailable.status, 503);
  assert.doesNotMatch(await unavailable.text(), /SECRET|API_KEY/);
});

test('account throttling runs before authentication and returns browser-readable errors', async () => {
  const response = await worker.fetch(new Request('https://brittain.app/api/auth/sign-in/email', {
    method: 'POST', headers: { 'CF-Connecting-IP': '203.0.113.4' },
  }), { AUTH_RATE_LIMITER: { limit: async () => ({ success: false }) } });
  assert.equal(response.status, 429);
  assert.match((await response.json()).message, /one minute/);
});

test('paused staging can check its database without a model credential', async () => {
  const env = { CHAT_ENABLED: 'false', BETTER_AUTH_SECRET: 'test', DB: { prepare: () => ({ first: async () => ({ ok: 1 }) }) } };
  const request = new Request('https://staging.example/api/health');
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { status: 'ok', chatPaused: true });
  assert.equal((await worker.fetch(request, { ...env, CHAT_ENABLED: 'true' })).status, 503);
  assert.equal((await worker.fetch(request, { ...env, DB: { prepare() { throw new Error('Database unavailable'); } } })).status, 503);
});
