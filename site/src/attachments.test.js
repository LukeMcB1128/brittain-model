import test from 'node:test';
import assert from 'node:assert/strict';
import { messageContent } from './attachments.js';

test('attachments become safe multimodal message parts', () => {
  const content = messageContent('Review these.', [
    { kind: 'text', name: 'notes.txt', content: 'Hello', truncated: false },
    { kind: 'image', name: 'photo.png', dataUrl: 'data:image/png;base64,abc=' },
  ]);
  assert.deepEqual(content, [
    { type: 'text', text: 'Review these.' },
    { type: 'text', text: 'Attached file: notes.txt\n\nHello' },
    { type: 'text', text: 'Attached image: photo.png' },
    { type: 'image_url', image_url: { url: 'data:image/png;base64,abc=' } },
  ]);
});
