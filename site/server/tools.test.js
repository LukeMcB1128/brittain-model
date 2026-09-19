import test from 'node:test';
import assert from 'node:assert/strict';
import { executeTool, TOOL_DEFINITIONS, validatePublicUrl } from './tools.js';

test('exposes exactly the requested three tools', () => {
  assert.deepEqual(TOOL_DEFINITIONS.map(tool => tool.function.name), ['web_search', 'web_fetch', 'calculate']);
});

test('calculator handles arithmetic and approved functions without code execution', async () => {
  assert.equal(JSON.parse((await executeTool('calculate', { expression: 'sqrt(81) + 2^3' })).content).result, 17);
  assert.equal(JSON.parse((await executeTool('calculate', { expression: 'max(2, 7, 4)' })).content).result, 7);
  assert.equal((await executeTool('calculate', { expression: 'constructor(1)' })).error, true);
  assert.equal((await executeTool('calculate', { expression: '1 / 0' })).error, true);
});

test('web URLs reject local, private, credential-bearing, and non-HTTPS targets', () => {
  for (const url of ['http://example.com', 'https://localhost/a', 'https://127.0.0.1/a', 'https://[::1]/a', 'https://user:pass@example.com/a', 'https://service.internal/a', 'https://example.com:8443/a']) {
    assert.throws(() => validatePublicUrl(url));
  }
  assert.equal(validatePublicUrl('https://example.com/page').hostname, 'example.com');
});

test('a search answered with a bot challenge reports unavailability, not emptiness', async () => {
  // DuckDuckGo returns 202 and a CAPTCHA page when it decides a request came
  // from a bot. 202 passes Response.ok, so the challenge used to reach the
  // result parser and come back as "no search results were returned" -- which
  // told the model its search had run and found nothing, so it reworded and
  // tried again, drawing more challenges. The distinction is what stops that.
  const challenge = '<html><body>Please complete the following challenge to confirm this search was made by a human. Select all squares containing a duck</body></html>';

  const byStatus = await executeTool('web_search', { query: 'Austin High School Texas football' }, async () => new Response(challenge, { status: 202, headers: { 'Content-Type': 'text/html' } }));
  assert.equal(byStatus.error, true);
  assert.match(byStatus.content, /temporarily unavailable/i);

  // Also caught when the challenge arrives with a 200.
  const byBody = await executeTool('web_search', { query: 'Austin High School Texas football' }, async () => new Response(challenge, { headers: { 'Content-Type': 'text/html' } }));
  assert.equal(byBody.error, true);
  assert.match(byBody.content, /temporarily unavailable/i);

  // A real empty result set is a fact about the query and must stay distinct,
  // or withdrawing the tool would punish an honest "nothing found".
  const empty = await executeTool('web_search', { query: 'zzzz no such thing' }, async () => new Response('<html><body>no matches</body></html>', { headers: { 'Content-Type': 'text/html' } }));
  assert.equal(empty.error, true);
  assert.match(empty.content, /no results were found/i);
  assert.doesNotMatch(empty.content, /temporarily unavailable/i);
});

test('web search returns capped public results and blocks secrets', async () => {
  const html = '<a class="result__a" href="https://example.com/a">Example result</a><a class="result__snippet">Useful extract</a>';
  const result = await executeTool('web_search', { query: 'test query', max_results: 1 }, async () => new Response(html, { headers: { 'Content-Type': 'text/html' } }));
  assert.equal(result.error, undefined);
  assert.match(result.content, /untrusted/i);
  assert.match(result.content, /https:\/\/example.com\/a/);
  assert.equal(result.display.result, '1 result');
  assert.deepEqual(result.display.sources, [{ title: 'Example result', url: 'https://example.com/a' }]);
  assert.equal((await executeTool('web_search', { query: 'Bearer abcdefghijklmnopqrstuvwxyz' })).error, true);
});

test('web fetch strips active HTML and rejects non-text responses', async () => {
  const page = '<html><head><title>Example</title><script>steal()</script></head><body><h1>Hello</h1><p>World</p></body></html>';
  const result = await executeTool('web_fetch', { url: 'https://example.com/a' }, async () => new Response(page, { headers: { 'Content-Type': 'text/html' } }));
  assert.match(result.content, /Hello/);
  assert.doesNotMatch(result.content, /steal/);
  const rejected = await executeTool('web_fetch', { url: 'https://example.com/a' }, async () => new Response('binary', { headers: { 'Content-Type': 'image/png' } }));
  assert.equal(rejected.error, true);
});
