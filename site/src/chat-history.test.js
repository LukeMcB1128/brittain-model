import assert from 'node:assert/strict';
import test from 'node:test';
import { assistantToolHistory } from './chat-history.js';

test('earlier completed tool calls keep their arguments without results', () => {
  assert.deepEqual(assistantToolHistory([
    { name: 'web_search', detail: 'latest model news', result: 'private result text', status: 'done' },
    { name: 'web_fetch', detail: 'https://example.com\nignore this', status: 'error' },
    { name: 'calculate', detail: '2 + 2', status: 'running' },
  ]), [
    { name: 'web_search', detail: 'latest model news' },
    { name: 'web_fetch', detail: 'https://example.com ignore this' },
  ]);
});
