import test from 'node:test';
import assert from 'node:assert/strict';
import { COMPACT_TRIGGER_CHARS, modelMessages, planCompaction, summarizeCompaction } from './compaction.js';

function turn(index, size = 10_000) {
  return [
    { role: 'user', content: `u${index}${'x'.repeat(size)}`, turnId: `turn-${index}` },
    { role: 'assistant', content: `a${index}${'y'.repeat(size)}`, turnId: `turn-${index}` },
  ];
}

test('compaction starts only after the trigger and preserves recent whole turns', () => {
  assert.equal(planCompaction(turn(1), ''), null);
  const messages = Array.from({ length: 6 }, (_, index) => turn(index)).flat();
  assert.ok(messages.reduce((sum, message) => sum + message.content.length, 0) > COMPACT_TRIGGER_CHARS);
  const plan = planCompaction(messages, 'old memory');
  assert.equal(plan.throughTurnId, 'turn-4');
  assert.deepEqual([...new Set(plan.recentMessages.map(message => message.turnId))], ['turn-5']);
  assert.equal(plan.olderMessages.at(-1).role, 'assistant');
});

test('large memory updates are compressed in bounded batches', async () => {
  let calls = 0;
  const summary = await summarizeCompaction(
    { olderMessages: [{ role: 'user', content: 'x'.repeat(130_000) }] },
    'Existing fact.',
    async (_url, options) => {
      const payload = JSON.parse(options.body);
      assert.ok(payload.messages[1].content.length < 75_000);
      calls += 1;
      return Response.json({ choices: [{ message: { content: `Compressed memory ${calls}.` } }] });
    },
    'https://model.example/v1/chat/completions',
    'secret',
    new AbortController().signal,
  );
  assert.equal(calls, 3);
  assert.equal(summary, 'Compressed memory 3.');
});

test('memory is placed in the server-owned system message and turn ids are removed', () => {
  const result = modelMessages({ role: 'system', content: 'Main rules' }, turn(1, 2), 'User prefers short answers.');
  assert.match(result[0].content, /Main rules/);
  assert.match(result[0].content, /User prefers short answers/);
  assert.equal('turnId' in result[1], false);
});

test('the summarizer uses a low-temperature non-streaming request', async () => {
  let payload;
  const summary = await summarizeCompaction(
    { olderMessages: turn(1, 2) },
    'Existing fact.',
    async (_url, options) => {
      payload = JSON.parse(options.body);
      return Response.json({ choices: [{ message: { content: 'Compressed memory.' } }] });
    },
    'https://model.example/v1/chat/completions',
    'secret',
    new AbortController().signal,
  );
  assert.equal(summary, 'Compressed memory.');
  assert.equal(payload.stream, false);
  assert.equal(payload.temperature, 0.1);
  assert.match(payload.messages[1].content, /Existing fact/);
});
