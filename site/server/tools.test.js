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

test('web search returns capped public results and blocks secrets', async () => {
  const html = '<a class="result__a" href="https://example.com/a">Example result</a><a class="result__snippet">Useful extract</a>';
  const result = await executeTool('web_search', { query: 'test query', max_results: 1 }, async () => new Response(html, { headers: { 'Content-Type': 'text/html' } }));
  assert.equal(result.error, undefined);
  assert.match(result.content, /untrusted/i);
  assert.match(result.content, /https:\/\/example.com\/a/);
  assert.equal(result.display.result, '1 result');
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
