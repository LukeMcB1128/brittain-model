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
  const empty = await executeTool('web_search', { query: 'zzzz no such thing' }, async () => new Response('<html><body><div class="no-results">No results found</div></body></html>', { headers: { 'Content-Type': 'text/html' } }));
  assert.equal(empty.error, undefined);
  assert.equal(empty.display.result, '0 results');
  assert.doesNotMatch(empty.content, /temporarily unavailable/i);
});

test('Brave search uses the server key, filters results, and preserves source text', async () => {
  let calls = 0;
  const result = await executeTool('web_search', { query: 'current facts', allowed_domains: ['example.com'], max_results: 2 }, async (url, options) => {
    calls++;
    assert.equal(url.origin, 'https://api.search.brave.com');
    assert.equal(url.searchParams.get('q'), 'current facts (site:example.com)');
    assert.equal(options.headers['X-Subscription-Token'], 'server-search-secret');
    assert.equal(options.redirect, 'error');
    return Response.json({ type: 'search', web: { results: [
      { title: 'Private', url: 'https://localhost/a' },
      { title: 'Wrong domain', url: 'https://other.com/a' },
      { title: '<b>One</b>', url: 'https://example.com/a', description: 'First &amp; current' },
      { title: 'Duplicate', url: 'https://example.com/a' },
      { title: 'Two', url: 'https://sub.example.com/b', description: 'Second' },
      { title: 'Three', url: 'https://example.com/c' },
    ] } });
  }, { braveSearchApiKey: 'server-search-secret' });
  assert.equal(calls, 1);
  assert.equal(result.error, undefined);
  assert.deepEqual(result.display.sources, [{ title: 'One', url: 'https://example.com/a' }, { title: 'Two', url: 'https://sub.example.com/b' }]);
  assert.match(result.content, /First & current/);
  assert.doesNotMatch(JSON.stringify(result), /server-search-secret/);
});

test('API outages, bad responses, and timeouts use one independent fallback', async () => {
  for (const failure of [() => new Response('limited', { status: 429 }), () => new Response('oops'), () => Response.json({ error: 'bad key' }), () => { throw new DOMException('timeout', 'TimeoutError'); }]) {
    const hosts = [];
    const result = await executeTool('web_search', { query: 'example' }, async url => {
      hosts.push(url.hostname);
      if (url.hostname === 'api.search.brave.com') return failure();
      return new Response('<a class="result__a" href="https://example.com/a">Example</a>');
    }, { braveSearchApiKey: 'secret' });
    assert.equal(result.error, undefined);
    assert.deepEqual(hosts, ['api.search.brave.com', 'html.duckduckgo.com']);
    assert.match(result.content, /DuckDuckGo HTML/);
  }
});

test('empty API searches do not trigger a fallback or disable search', async () => {
  let calls = 0;
  const result = await executeTool('web_search', { query: 'nothing' }, async () => {
    calls++;
    return Response.json({ type: 'search', web: { results: [] } });
  }, { braveSearchApiKey: 'secret' });
  assert.equal(calls, 1);
  assert.equal(result.error, undefined);
  assert.equal(result.display.result, '0 results');
});

test('provider failures and unrecognized HTML report unavailability without leaking errors', async () => {
  for (const response of [() => new Response('blocked', { status: 403 }), () => new Response('<html>Sign in</html>'), () => { throw new Error('internal-secret'); }]) {
    const result = await executeTool('web_search', { query: 'example' }, response);
    assert.equal(result.error, true);
    assert.match(result.content, /temporarily unavailable/);
    assert.doesNotMatch(result.content, /internal-secret/);
  }
});

test('stopping a search cancels its request and skips fallback', async () => {
  const controller = new AbortController();
  let calls = 0;
  const result = await executeTool('web_search', { query: 'example' }, async (_url, options) => {
    calls++;
    controller.abort();
    assert.equal(options.signal.aborted, true);
    options.signal.throwIfAborted();
  }, { braveSearchApiKey: 'secret', signal: controller.signal });
  assert.equal(result.error, true);
  assert.equal(calls, 1);
});

test('a search result never borrows the next result snippet', async () => {
  const result = await executeTool('web_search', { query: 'example' }, async () => new Response(
    '<h2><a class="result__a" href="https://example.com/a">First</a></h2><h2><a class="result__a" href="https://example.com/b">Second</a></h2><a class="result__snippet">Second only</a>',
  ));
  const data = JSON.parse(result.content.slice(result.content.indexOf('{')));
  assert.equal(data.results[0].snippet, '');
  assert.equal(data.results[1].snippet, 'Second only');
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
