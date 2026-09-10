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
  assert.equal((await handleApi(req({ messages: [{ role: 'user', content: 'x'.repeat(500001) }] }), env)).status, 400);
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
  assert.deepEqual(payload.tools.map(tool => tool.function.name), ['web_search', 'web_fetch', 'calculate']);
  assert.equal((await response.text()).includes('test-only-secret'), false);
});
test('accepts safe image parts and rejects remote or oversized attachment content', async () => {
  let payload;
  const content = [{ type: 'text', text: 'Describe this image.' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' } }];
  const response = await handleApi(req({ messages: [{ role: 'user', content }] }), env, async (_url, options) => {
    payload = JSON.parse(options.body);
    return sse([{ choices: [{ index: 0, delta: { content: 'An image.' }, finish_reason: 'stop' }] }, '[DONE]']);
  });
  await response.text();
  assert.deepEqual(payload.messages.at(-1).content, content);
  assert.equal((await handleApi(req({ messages: [{ role: 'user', content: [{ type: 'text', text: 'Read it.' }, { type: 'image_url', image_url: { url: 'https://example.com/image.png' } }] }] }), env)).status, 400);
  assert.equal((await handleApi(req({ messages: [{ role: 'assistant', content }] }), env)).status, 400);
});
test('upstream auth errors do not expose the key, HTML is rejected', async () => {
  const auth = await handleApi(req(), env, async () => Response.json({ error: 'test-only-secret' }, { status: 401 }));
  assert.equal(auth.status, 200);
  const authBody = await auth.text();
  assert.equal(authBody.includes('test-only-secret'), false);
  assert.match(authBody, /rejected its access key/);
  const html = await handleApi(req(), env, async () => new Response('<html/>'));
  assert.equal(html.status, 200);
  assert.match(await html.text(), /unexpected response/);
});
test('session readiness verifies the model server rather than only the key', async () => {
  const session = () => new Request('https://site.example/api/session', { headers: { 'oai-authenticated-user-id': 'test-user' } });
  const down = await handleApi(session(), env, async () => new Response('', { status: 502 }));
  assert.equal((await down.json()).ready, false);
  const up = await handleApi(session(), env, async () => Response.json({ data: [{ id: 'brittain4', max_model_len: 32768 }] }));
  assert.equal((await up.json()).ready, true);
});

function sse(events) {
  return new Response(events.map(event => event === '[DONE]' ? 'data: [DONE]\n\n' : `data: ${JSON.stringify(event)}\n\n`).join(''), { headers: { 'Content-Type': 'text/event-stream' } });
}

test('executes only declared tools and returns the final streamed reply', async () => {
  let round = 0;
  const response = await handleApi(req(), env, async (url, options) => {
    assert.equal(url.endsWith('/chat/completions'), true);
    const payload = JSON.parse(options.body);
    round += 1;
    if (round === 1) {
      return sse([
        { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call-1', type: 'function', function: { name: 'calculate', arguments: '{"expression":"2+2"}' } }] }, finish_reason: 'tool_calls' }] },
        '[DONE]',
      ]);
    }
    assert.equal(payload.messages[0].role, 'system');
    assert.match(payload.messages[0].content, /Use calculate for arithmetic/);
    assert.equal(payload.messages.at(-2).tool_calls[0].function.name, 'calculate');
    assert.equal(payload.messages.at(-1).role, 'tool');
    assert.match(payload.messages.at(-1).content, /"result":4/);
    return sse([
      { choices: [{ index: 0, delta: { content: 'The answer is 4.' } }] },
      { choices: [], usage: { prompt_tokens: 40, completion_tokens: 5, total_tokens: 45 } },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      '[DONE]',
    ]);
  });
  const body = await response.text();
  assert.equal(round, 2);
  assert.match(body, /"type":"tool"/);
  assert.match(body, /"status":"done"/);
  assert.match(body, /The answer is 4/);
  assert.match(body, /"type":"usage"/);
  assert.match(body, /"type":"done"/);
});

test('explicit calculator and web requests are routed before the answer', async () => {
  for (const [prompt, expected] of [
    ['Calculate 27 * 14.', 'calculate'],
    ['Search the web for current model releases.', 'web_search'],
    ['Read and summarize https://example.com/page', 'web_fetch'],
  ]) {
    let modelPayload;
    const response = await handleApi(req({ messages: [{ role: 'user', content: prompt }] }), env, async (url, options) => {
      if (!String(url).endsWith('/chat/completions')) {
        if (expected === 'web_search') return new Response('<a class="result__a" href="https://example.com/result">Result</a><a class="result__snippet">Summary</a>', { headers: { 'Content-Type': 'text/html' } });
        return new Response('<html><head><title>Example</title></head><body>Page text</body></html>', { headers: { 'Content-Type': 'text/html' } });
      }
      modelPayload = JSON.parse(options.body);
      return sse([{ choices: [{ index: 0, delta: { content: 'Done' }, finish_reason: 'stop' }] }, '[DONE]']);
    });
    const body = await response.text();
    assert.equal(modelPayload.tool_choice, 'auto');
    assert.equal(modelPayload.messages.at(-2).tool_calls[0].function.name, expected);
    assert.equal(modelPayload.messages.at(-1).role, 'tool');
    assert.match(body, new RegExp(`"name":"${expected}"`));
    assert.match(body, /"type":"done"/);
  }
});
