import test from 'node:test';
import assert from 'node:assert/strict';
import { handleApi } from './gateway.js';
const env = { BRITTAIN4_API_KEY: 'test-only-secret' };
function req(body = { messages: [{ role: 'user', content: 'Hello' }] }, extra = {}) {
  return new Request('https://site.example/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json', 'oai-authenticated-user-id': 'test-user', ...extra }, body: JSON.stringify(body) });
}
test('requires authenticated site identity, a secret, and same-origin POST', async () => {
  const request = req(); request.headers.delete('oai-authenticated-user-id');
  assert.equal((await handleApi(request, env)).status, 401);
  assert.equal((await handleApi(req(), {})).status, 503);
  assert.equal((await handleApi(req(undefined, { origin: 'https://evil.example' }), env)).status, 403);
});
test('validates messages and fixes server-owned model/options', async () => {
  assert.equal((await handleApi(req({ messages: [{ role: 'system', content: 'x' }] }), env)).status, 400);
  assert.equal((await handleApi(req({ messages: [{ role: 'user', content: 'x'.repeat(180001) }] }), env)).status, 413);
  let payload;
  const response = await handleApi(req({ model: 'other', max_tokens: 99999, messages: [{ role: 'user', content: 'Hello' }] }), env, async (url, options) => {
    assert.equal(url, 'https://fragility-devoutly-dazzling.ngrok-free.dev/v1/chat/completions');
    assert.equal(options.headers.Authorization, 'Bearer test-only-secret');
    payload = JSON.parse(options.body);
    return new Response('data: [DONE]\n', { headers: { 'Content-Type': 'text/event-stream' } });
  });
  assert.equal(response.status, 200);
  assert.equal(payload.model, 'brittain4');
  assert.equal(payload.max_tokens, 2048);
  assert.equal(payload.chat_template_kwargs.enable_thinking, false);
  assert.equal((await response.text()).includes('test-only-secret'), false);
});
test('upstream auth errors do not expose the key, HTML is rejected', async () => {
  const auth = await handleApi(req(), env, async () => Response.json({ error: 'test-only-secret' }, { status: 401 }));
  assert.equal(auth.status, 502);
  assert.equal((await auth.text()).includes('test-only-secret'), false);
  const html = await handleApi(req(), env, async () => new Response('<html/>'));
  assert.equal(html.status, 502);
});
test('session readiness verifies the model server rather than only the key', async () => {
  const session = () => new Request('https://site.example/api/session', { headers: { 'oai-authenticated-user-id': 'test-user' } });
  const down = await handleApi(session(), env, async () => new Response('', { status: 502 }));
  assert.equal((await down.json()).ready, false);
  const up = await handleApi(session(), env, async () => Response.json({ data: [{ id: 'brittain4', max_model_len: 32768 }] }));
  assert.equal((await up.json()).ready, true);
});
