import test from 'node:test';
import assert from 'node:assert/strict';
import { messageContent, requestAssets } from './attachments.js';

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

test('request assets include attached and generated PDFs once', () => {
  const pdf = { id: 'pdf-1', kind: 'pdf', name: 'form.pdf', type: 'application/pdf', dataUrl: 'data:application/pdf;base64,abc=', pageCount: 1, pageImages: [] };
  const webp = { id: 'image-1', kind: 'image', name: 'photo.webp', type: 'image/webp', dataUrl: 'data:image/webp;base64,abc=' };
  assert.deepEqual(requestAssets([{ attachments: [pdf, webp], artifacts: [pdf] }]), [{ id: 'pdf-1', name: 'form.pdf', type: 'application/pdf', dataUrl: pdf.dataUrl, pageCount: 1, pageImages: [] }]);
});
