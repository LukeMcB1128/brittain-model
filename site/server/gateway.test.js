import test from 'node:test';
import assert from 'node:assert/strict';
import { PDFDocument } from 'pdf-lib';
import { handleApi, toolInstructions } from './gateway.js';
const env = { BRITTAIN4_API_KEY: 'test-only-secret' };
function req(body = { messages: [{ role: 'user', content: 'Hello' }] }, extra = {}) {
  return new Request('https://site.example/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json', 'oai-authenticated-user-id': 'test-user', ...extra }, body: JSON.stringify(body) });
}
test('builds the current date inside the request instead of at module initialization', () => {
  const instructions = toolInstructions(new Date('2026-09-18T12:00:00Z'));
  assert.match(instructions, /The current date is Friday, 2026-09-18\./);
  assert.doesNotMatch(instructions, /1970/);
});
test('requires authenticated site identity, a secret, and same-origin POST', async () => {
  const request = req(); request.headers.delete('oai-authenticated-user-id');
  assert.equal((await handleApi(request, env)).status, 401);
  assert.equal((await handleApi(req(), {})).status, 503);
  assert.equal((await handleApi(req(undefined, { origin: 'https://evil.example' }), env)).status, 403);
});
test('accepts the standalone Worker account identity without a Sites header', async () => {
  const request = req();
  request.headers.delete('oai-authenticated-user-id');
  const response = await handleApi(request, env, async () => sse([{ choices: [{ index: 0, delta: { content: 'Hello' }, finish_reason: 'stop' }] }, '[DONE]']), 'standalone-user');
  assert.equal(response.status, 200);
  assert.match(await response.text(), /Hello/);
});
test('canceling a reply aborts the upstream model request', async () => {
  let upstreamSignal;
  let markStarted;
  const started = new Promise(resolve => { markStarted = resolve; });
  const response = await handleApi(req(), env, async (_url, options) => {
    upstreamSignal = options.signal;
    markStarted();
    return new Response(new ReadableStream({
      start(controller) {
        options.signal.addEventListener('abort', () => {
          controller.error(new DOMException('Stopped', 'AbortError'));
        }, { once: true });
      },
    }), { headers: { 'Content-Type': 'text/event-stream' } });
  });
  await started;
  await response.body.cancel('stopped');
  assert.equal(upstreamSignal.aborted, true);
});
test('does not trust the legacy Sites header when the standalone Worker rejects the session', async () => {
  assert.equal((await handleApi(req(), env, fetch, null)).status, 401);
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
  assert.equal(payload.model, 'step-0100-mm');
  assert.equal(payload.max_tokens, 2048);
  assert.equal(payload.chat_template_kwargs.enable_thinking, false);
  assert.match(payload.messages[0].content, /The current date is \w+, \d{4}-\d{2}-\d{2}\./);
  assert.doesNotMatch(payload.messages[0].content, /1970-01-01/);
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
test('clips a large text attachment before it reaches the model context', async () => {
  let payload;
  const content = [
    { type: 'text', text: 'Review the attached content.' },
    { type: 'text', text: `Attached file: large.md\n\n${'x'.repeat(120_000)}` },
  ];
  const response = await handleApi(req({ messages: [{ role: 'user', content }] }), env, async (_url, options) => {
    payload = JSON.parse(options.body);
    return sse([{ choices: [{ index: 0, delta: { content: 'Reviewed.' }, finish_reason: 'stop' }] }, '[DONE]']);
  });
  assert.match(await response.text(), /Reviewed/);
  assert.ok(payload.messages.at(-1).content[1].text.length <= 48_000);
  assert.match(payload.messages.at(-1).content[1].text, /Middle content omitted/);
});
test('offers PDF tools only when a PDF attachment is present', async () => {
  let payload;
  const response = await handleApi(req({ messages: [{ role: 'user', content: 'Summarize the attached file.' }], attachments: [{ id: 'pdf-1', name: 'notes.pdf', type: 'application/pdf', dataUrl: `data:application/pdf;base64,${btoa('%PDF-test')}`, pageCount: 1, pageImages: [] }] }), env, async (_url, options) => {
    payload = JSON.parse(options.body);
    return sse([{ choices: [{ index: 0, delta: { content: 'Summary' }, finish_reason: 'stop' }] }, '[DONE]']);
  });
  await response.text();
  assert.deepEqual(payload.tools.slice(-6).map(tool => tool.function.name), ['pdf_info', 'pdf_render', 'pdf_fill_form', 'pdf_stamp', 'pdf_pages', 'pdf_merge']);
  assert.equal(payload.messages.at(-1).content, 'Summarize the attached file.');
});
test('inspects the only attached PDF without making the model guess its filename', async () => {
  const document = await PDFDocument.create();
  document.addPage([612, 792]);
  const pdf = Buffer.from(await document.save()).toString('base64');
  let payload;
  const response = await handleApi(req({ messages: [{ role: 'user', content: 'Inspect the attached PDF and tell me its page count.' }], attachments: [{ id: 'pdf-1', name: 'notes.pdf', type: 'application/pdf', dataUrl: `data:application/pdf;base64,${pdf}`, pageCount: 1, pageImages: [] }] }), env, async (_url, options) => {
    payload = JSON.parse(options.body);
    return sse([{ choices: [{ index: 0, delta: { content: 'It has one page.' }, finish_reason: 'stop' }] }, '[DONE]']);
  });
  const body = await response.text();
  assert.equal(payload.messages.at(-2).tool_calls[0].function.name, 'pdf_info');
  assert.deepEqual(JSON.parse(payload.messages.at(-2).tool_calls[0].function.arguments), {});
  assert.match(payload.messages.at(-1).content, /"page_count": 1/);
  assert.match(body, /"name":"pdf_info"/);
});
test('executes a PDF edit and streams a downloadable result', async () => {
  const document = await PDFDocument.create();
  document.addPage([612, 792]);
  document.addPage([612, 792]);
  const pdf = Buffer.from(await document.save()).toString('base64');
  let round = 0;
  const response = await handleApi(req({ messages: [{ role: 'user', content: 'Rotate page 1 in the attached PDF.' }], attachments: [{ id: 'pdf-1', name: 'pages.pdf', type: 'application/pdf', dataUrl: `data:application/pdf;base64,${pdf}`, pageCount: 2, pageImages: [] }] }), env, async (_url, options) => {
    const payload = JSON.parse(options.body);
    round += 1;
    if (round === 1) {
      assert.equal(payload.tool_choice, 'required');
      assert.deepEqual(payload.tools.map(tool => tool.function.name), ['pdf_pages']);
      return sse([{ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'pdf-call', type: 'function', function: { name: 'pdf_pages', arguments: '{"file":"pages.pdf","operation":"rotate","pages":"1","degrees":90}' } }] }, finish_reason: 'tool_calls' }] }, '[DONE]']);
    }
    assert.equal(payload.messages.at(-1).role, 'tool');
    assert.match(payload.messages.at(-1).content, /pages-rotated\.pdf/);
    return sse([{ choices: [{ index: 0, delta: { content: 'The rotated PDF is ready.' }, finish_reason: 'stop' }] }, '[DONE]']);
  });
  const body = await response.text();
  assert.equal(round, 2);
  assert.match(body, /"type":"artifact"/);
  assert.match(body, /pages-rotated\.pdf/);
  assert.match(body, /data:application\/pdf;base64/);
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
  // A LoRA entry reports max_model_len: null and points at the base it runs on.
  // Readiness must look for the adapter and take the window from its parent --
  // reading the adapter's own null and falling through to the default is how
  // the app once advertised a context window it did not have.
  const up = await handleApi(session(), env, async () => Response.json({
    data: [
      { id: 'brittain4', max_model_len: 32768 },
      { id: 'step-0100-mm', max_model_len: null, parent: 'brittain4' },
    ],
  }));
  const ready = await up.json();
  assert.equal(ready.ready, true);
  // The checkpoint id stays server-side; the client renders this as the
  // assistant's name.
  assert.equal(ready.model, 'brittain4');
  assert.equal(ready.context, 32768);

  // The base alone is not the served model, so it is not ready.
  const baseOnly = await handleApi(session(), env, async () => Response.json({ data: [{ id: 'brittain4', max_model_len: 32768 }] }));
  assert.equal((await baseOnly.json()).ready, false);
});

test('long chats compact older turns and send recent turns with reusable memory', async () => {
  const messages = [];
  for (let index = 0; index < 6; index += 1) {
    messages.push({ role: 'user', content: `Question ${index} ${'x'.repeat(7_000)}`, turnId: `turn-${index}` });
    messages.push({ role: 'assistant', content: `Answer ${index} ${'y'.repeat(7_000)}`, turnId: `turn-${index}` });
  }
  messages.push({ role: 'user', content: 'What did we decide?', turnId: 'turn-current' });
  let calls = 0;
  const response = await handleApi(req({ messages, memory: 'The user prefers concise replies.' }), env, async (_url, options) => {
    const payload = JSON.parse(options.body);
    calls += 1;
    if (!payload.stream) {
      assert.equal(payload.temperature, 0.1);
      assert.match(payload.messages[1].content, /user prefers concise replies/i);
      return Response.json({ choices: [{ message: { content: 'The user prefers concise replies. Several options were compared.' } }] });
    }
    assert.match(payload.messages[0].content, /Several options were compared/);
    assert.ok(payload.messages.length < messages.length + 1);
    assert.equal(payload.messages.at(-1).content, 'What did we decide?');
    return sse([{ choices: [{ index: 0, delta: { content: 'You chose the second option.' }, finish_reason: 'stop' }] }, '[DONE]']);
  });
  const body = await response.text();
  assert.equal(calls, 2);
  assert.match(body, /"type":"compaction","status":"running"/);
  assert.match(body, /"type":"compaction","status":"done"/);
  assert.match(body, /"throughTurnId":"turn-3"/);
  assert.match(body, /You chose the second option/);
});

function sse(events) {
  return new Response(events.map(event => event === '[DONE]' ? 'data: [DONE]\n\n' : `data: ${JSON.stringify(event)}\n\n`).join(''), { headers: { 'Content-Type': 'text/event-stream' } });
}

test('a tool reporting itself unavailable is withdrawn for the rest of the reply', async () => {
  // Real transcript: ten progressively reworded searches for one high school,
  // every one answered with a bot challenge, and the user got an error rather
  // than an answer. Withdrawing the tool is what makes the model stop.
  const challenge = '<html><body>Please complete the following challenge to confirm this search was made by a human. Select all squares containing a duck</body></html>';
  let searches = 0;
  let modelRounds = 0;
  const offeredPerRound = [];
  const response = await handleApi(req(), env, async (url, options) => {
    if (String(url).includes('duckduckgo')) {
      searches += 1;
      return new Response(challenge, { status: 202, headers: { 'Content-Type': 'text/html' } });
    }
    modelRounds += 1;
    offeredPerRound.push((JSON.parse(options.body).tools || []).map(tool => tool.function.name));
    if (modelRounds === 1) {
      return sse([
        { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 's1', type: 'function', function: { name: 'web_search', arguments: '{"query":"Austin High School Texas football"}' } }] }, finish_reason: 'tool_calls' }] },
        '[DONE]',
      ]);
    }
    return sse([
      { choices: [{ index: 0, delta: { content: 'Austin High School is in Austin, Texas.' } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      '[DONE]',
    ]);
  });
  const body = await response.text();
  assert.equal(searches, 1, 'a challenge must not be retried within one reply');
  assert.equal(offeredPerRound[0].includes('web_search'), true);
  assert.equal(offeredPerRound[1].includes('web_search'), false, 'web_search must be withdrawn after it reports unavailability');
  assert.equal(offeredPerRound[1].includes('calculate'), true, 'other tools stay available');
  assert.match(body, /Austin High School is in Austin/);
});

test('routed web search uses the configured API and sends results, never its key, to the model', async () => {
  let searches = 0;
  let modelCalls = 0;
  const response = await handleApi(req({ messages: [{ role: 'user', content: 'Search the web for current model releases.' }] }), { ...env, BRAVE_SEARCH_API_KEY: 'search-only-secret' }, async (url, options) => {
    if (new URL(url).hostname === 'api.search.brave.com') {
      searches++;
      assert.equal(options.headers['X-Subscription-Token'], 'search-only-secret');
      return Response.json({ type: 'search', web: { results: [{ title: 'Release', url: 'https://example.com/release', description: 'Release details' }] } });
    }
    modelCalls++;
    assert.doesNotMatch(options.body, /search-only-secret/);
    assert.match(JSON.parse(options.body).messages.at(-1).content, /Release details/);
    return sse([{ choices: [{ index: 0, delta: { content: 'Here is the release.' }, finish_reason: 'stop' }] }, '[DONE]']);
  });
  const body = await response.text();
  assert.equal(searches, 1);
  assert.equal(modelCalls, 1);
  assert.match(body, /Here is the release/);
  assert.doesNotMatch(body, /search-only-secret/);
});

test('running out of tool rounds ends with an answer instead of an error', async () => {
  // The cap used to throw, discarding everything gathered. One real session
  // lost five successful page reads that way.
  let sawToollessRound = false;
  let rounds = 0;
  const response = await handleApi(req(), env, async (url, options) => {
    const payload = JSON.parse(options.body);
    rounds += 1;
    if (payload.tools === undefined) {
      sawToollessRound = true;
      return sse([
        { choices: [{ index: 0, delta: { content: 'Here is what I found before running out of steps.' } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
        '[DONE]',
      ]);
    }
    return sse([
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: `c${rounds}`, type: 'function', function: { name: 'calculate', arguments: '{"expression":"1+1"}' } }] }, finish_reason: 'tool_calls' }] },
      '[DONE]',
    ]);
  });
  const body = await response.text();
  assert.equal(sawToollessRound, true, 'the final round must be sent with no tools');
  assert.match(body, /Here is what I found before running out of steps/);
  assert.doesNotMatch(body, /tool limit was reached/i);
});

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

// --- transcript records -----------------------------------------------------
// Nothing kept a record of what the API served: vLLM logs status lines with no
// message content, and the gateway streamed replies to the browser and forgot
// them. These pin the wiring, which the transcript-log unit tests cannot see.
const say = text => ({ choices: [{ index: 0, delta: { content: text }, finish_reason: null }] });
const stop = { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 11, completion_tokens: 5, total_tokens: 16 } };

function loggingEnv() {
  const written = [];
  return { written, env: { ...env, CHAT_LOG: { put: async (key, value) => written.push({ key, entry: JSON.parse(value) }) } } };
}

test('a served exchange is recorded once the reply has been delivered', async () => {
  const { written, env: logging } = loggingEnv();
  const response = await handleApi(
    req({ messages: [{ role: 'user', content: 'Who made you?' }] }), logging,
    async () => sse([say('I was made by '), say('Luke Brittain.'), stop, '[DONE]']));
  const body = await response.text();

  assert.match(body, /Luke Brittain/, 'the user still gets the reply');
  assert.equal(written.length, 1, 'exactly one record per exchange');
  const { entry } = written[0];
  assert.equal(entry.reply, 'I was made by Luke Brittain.');
  assert.equal(entry.finishReason, 'stop');
  assert.equal(entry.usage.total_tokens, 16);
  assert.deepEqual(entry.messages, [{ role: 'user', text: 'Who made you?' }]);
  assert.equal(entry.surface, 'web-chat');
  // The signed-in identity must not be stored, only a grouping key.
  assert.equal(JSON.stringify(entry).includes('test-user'), false);
  assert.match(written[0].key, /^chat\/\d{4}-\d{2}-\d{2}T/);
});

test('a store that is not configured changes nothing about the chat', async () => {
  const response = await handleApi(req(), env, async () => sse([say('hello'), stop, '[DONE]']));
  assert.equal(response.status, 200);
  assert.match(await response.text(), /"type":"done"/);
});

test('a store that throws does not break the reply', async () => {
  const broken = { ...env, CHAT_LOG: { put: async () => { throw new Error('KV is down'); } } };
  const response = await handleApi(req(), broken, async () => sse([say('still fine'), stop, '[DONE]']));
  const body = await response.text();
  assert.match(body, /still fine/);
  assert.match(body, /"type":"done"/);
  assert.doesNotMatch(body, /KV is down/, 'a storage failure must never reach the user');
});

test('a failed exchange is recorded with its partial reply', async () => {
  const { written, env: logging } = loggingEnv();
  const response = await handleApi(req(), logging, async () => sse([
    say('I started answering'),
    { choices: [], error: { message: 'upstream exploded' } },
    '[DONE]',
  ]));
  await response.text();
  assert.equal(written.length, 1, 'failures are worth recording too');
  assert.match(written[0].entry.error, /upstream exploded/);
  assert.equal(written[0].entry.reply, 'I started answering');
});

test('a served exchange reaches D1, the sink Sites actually uses', async () => {
  const rows = [];
  const DB = { prepare: sql => ({ bind: (...values) => ({ run: async () => rows.push({ sql, values }) }) }) };
  const response = await handleApi(
    req({ messages: [{ role: 'user', content: 'Hello there' }] }), { ...env, DB },
    async () => sse([say('Hi.'), stop, '[DONE]']));
  assert.match(await response.text(), /Hi\./);
  assert.equal(rows.length, 1);
  const payload = JSON.parse(rows[0].values.at(-1));
  assert.equal(payload.reply, 'Hi.');
  assert.deepEqual(payload.messages, [{ role: 'user', text: 'Hello there' }]);
});
