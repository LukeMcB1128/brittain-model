import test from 'node:test';
import assert from 'node:assert/strict';
import { accountConfig, safeNextPath } from './auth-client.js';

test('account configuration failures do not silently disable account protection', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('unavailable', { status: 503 }));
  await assert.rejects(accountConfig(), /could not be loaded/);
});

test('post-login redirects cannot leave the site through slash or backslash paths', () => {
  for (const value of [null, 'https://other.example', '//other.example', '/\\other.example', '/\n/other.example']) {
    assert.equal(safeNextPath(value), '/chat');
  }
  assert.equal(safeNextPath('/chat/123?test=true'), '/chat/123?test=true');
});
