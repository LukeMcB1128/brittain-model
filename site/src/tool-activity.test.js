import test from 'node:test';
import assert from 'node:assert/strict';
import { toolActivityMessage } from './tool-activity.js';

test('shows one short message for the latest active tool', () => {
  assert.equal(toolActivityMessage([
    { name: 'web_search', status: 'done' },
    { name: 'web_fetch', status: 'running' },
  ]), 'Searching the web…');
  assert.equal(toolActivityMessage([{ name: 'calculate', status: 'running' }]), 'Calculating…');
});

test('hides tool activity after the tool stops', () => {
  assert.equal(toolActivityMessage([{ name: 'web_search', status: 'done' }]), '');
  assert.equal(toolActivityMessage([{ name: 'web_search', status: 'error' }]), '');
  assert.equal(toolActivityMessage(), '');
});
