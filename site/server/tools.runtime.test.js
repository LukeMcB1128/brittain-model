import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

test('Brave requests work in workerd and redirects never forward the API key', async () => {
  const bundle = await build({
    absWorkingDir: fileURLToPath(new URL('..', import.meta.url)),
    bundle: true, write: false, platform: 'browser', format: 'esm',
    stdin: { resolveDir: fileURLToPath(new URL('..', import.meta.url)), contents: `
      import { executeTool } from './server/tools.js';
      export default { async fetch(input) {
        const calls = [];
        const redirect = new URL(input.url).pathname === '/redirect';
        const result = await executeTool('web_search', { query: 'reference' }, async (url, options) => {
          // Use the real workerd Request constructor. A plain Node mock accepts
          // redirect: 'error', which hid the production failure.
          const request = new Request(url, options);
          const host = new URL(request.url).hostname;
          calls.push({ host, redirect: request.redirect, hasKey: request.headers.has('X-Subscription-Token') });
          if (host === 'api.search.brave.com') {
            if (redirect) return new Response(null, { status: 302, headers: { Location: 'https://untrusted.example/collect' } });
            return Response.json({ type: 'search', web: { results: [{ title: 'Reference', url: 'https://example.com/reference', description: 'Documentation' }] } });
          }
          return new Response('<a class="result__a" href="https://example.com/fallback">Reference</a>');
        }, { braveSearchApiKey: 'synthetic-runtime-test-key' });
        return Response.json({ result, calls });
      } }
    ` },
  });
  const runtime = new Miniflare(convertV4MiniflareOptions({
    modules: true, compatibilityDate: '2026-09-13', cf: false,
    script: bundle.outputFiles[0].text,
  }));
  try {
    const success = await (await runtime.dispatchFetch('https://test.local/success')).json();
    assert.equal(success.result.error, undefined);
    assert.match(success.result.content, /Brave Search/);
    assert.deepEqual(success.calls, [{ host: 'api.search.brave.com', redirect: 'manual', hasKey: true }]);
    const redirect = await (await runtime.dispatchFetch('https://test.local/redirect')).json();
    assert.match(redirect.result.content, /DuckDuckGo HTML/);
    assert.deepEqual(redirect.calls, [
      { host: 'api.search.brave.com', redirect: 'manual', hasKey: true },
      { host: 'html.duckduckgo.com', redirect: 'manual', hasKey: false },
    ]);
    assert.doesNotMatch(JSON.stringify(success) + JSON.stringify(redirect), /synthetic-runtime-test-key/);
  } finally { await runtime.dispose(); }
});
