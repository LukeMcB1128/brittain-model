import test from 'node:test';
import assert from 'node:assert/strict';
import { activityProgressMessage, toolActivityItems } from './tool-activity.js';

test('keeps completed tool actions visible with their useful details', () => {
  assert.deepEqual(toolActivityItems([
    { id: '1', name: 'web_search', detail: 'latest model news', result: '5 results', status: 'done' },
    { id: '2', name: 'web_fetch', detail: 'https://example.com', status: 'running' },
  ]), [
    { id: '1', icon: 'search', status: 'done', label: 'Searched the web', detail: 'latest model news', result: '5 results' },
    { id: '2', icon: 'search', status: 'running', label: 'Reading a web page', detail: 'https://example.com', result: '' },
  ]);
});

test('reports compaction as a persistent response action', () => {
  assert.deepEqual(toolActivityItems([], 'done'), [{
    id: 'compaction', icon: 'compact', status: 'done', label: 'Earlier context compacted',
    detail: 'Older turns were summarized for this reply.', result: '',
  }]);
});

test('shows the current response phase without replacing active actions', () => {
  assert.equal(activityProgressMessage('streaming', 'thinking'), 'Thinking…');
  assert.equal(activityProgressMessage('streaming', 'reviewing'), 'Reviewing results…');
  assert.equal(activityProgressMessage('streaming', 'tool'), '');
  assert.equal(activityProgressMessage('streaming', 'answering'), '');
  assert.equal(activityProgressMessage('done', 'reviewing'), '');
});
