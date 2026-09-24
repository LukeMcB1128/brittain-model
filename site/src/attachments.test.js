import test from 'node:test';
import assert from 'node:assert/strict';
import { chatForSave, messageContent, planRequestAttachments, requestAssets } from './attachments.js';

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

test('new attachments replace old file bytes without removing old PDF text', () => {
  const turns = Array.from({ length: 11 }, (_, index) => ({ attachments: [{
    id: `pdf-${index}`, kind: 'pdf', name: `file-${index}.pdf`, type: 'application/pdf',
    dataUrl: 'data:application/pdf;base64,YQ==', content: `Text from file ${index}`, pageCount: 1,
  }] }));
  const plan = planRequestAttachments(turns);
  assert.equal(plan.assets.length, 10);
  assert.equal(plan.retainedIds.has('pdf-0'), false);
  assert.equal(plan.retainedIds.has('pdf-10'), true);
  const oldContent = messageContent('Review it.', turns[0].attachments, plan.retainedIds);
  assert.match(oldContent[1].text, /Text from file 0/);
  assert.match(oldContent[1].text, /attach it again/);
});

test('old image bytes drop before a new image reaches the eight-image limit', () => {
  const turns = Array.from({ length: 9 }, (_, index) => ({ attachments: [{
    id: `image-${index}`, kind: 'image', name: `photo-${index}.png`, type: 'image/png',
    dataUrl: 'data:image/png;base64,YQ==',
  }] }));
  const plan = planRequestAttachments(turns);
  assert.equal(plan.assets.length, 8);
  assert.equal(messageContent('Old photo.', turns[0].attachments, plan.retainedIds).some(part => part.type === 'image_url'), false);
  assert.match(messageContent('Old photo.', turns[0].attachments, plan.retainedIds)[1].text, /attach it again/);
  assert.equal(messageContent('New photo.', turns[8].attachments, plan.retainedIds).some(part => part.type === 'image_url'), true);
});

test('a new PDF keeps prepared pages before old PDF bytes take request space', () => {
  const old = { id: 'old', kind: 'pdf', name: 'old.pdf', type: 'application/pdf', dataUrl: `data:application/pdf;base64,${'a'.repeat(13_500_000)}` };
  const pageData = `data:image/jpeg;base64,${'a'.repeat(6_900_000)}`;
  const current = { id: 'current', kind: 'pdf', name: 'current.pdf', type: 'application/pdf', dataUrl: 'data:application/pdf;base64,YQ==', pageImages: [{ page: 1, dataUrl: pageData }, { page: 2, dataUrl: pageData }] };
  const plan = planRequestAttachments([{ attachments: [old] }, { attachments: [current] }]);
  assert.deepEqual(plan.assets.map(asset => asset.id), ['current']);
  assert.equal(plan.assets[0].pageImages.length, 2);
});

test('saving a chat sends file text and names without repeated binary data', () => {
  const chat = { id: 'chat-1', contextStart: 1, memory: 'Earlier file summary.', messages: [{
    prompt: 'Review this.', answer: 'Done.',
    attachments: [{ id: 'pdf-1', kind: 'pdf', name: 'notes.pdf', content: 'Page 1 text', dataUrl: 'data:application/pdf;base64,abc=', pageImages: [{ page: 1, dataUrl: 'data:image/jpeg;base64,abc=' }] }],
    artifacts: [{ id: 'pdf-2', dataUrl: 'data:application/pdf;base64,abc=' }],
  }, { prompt: 'Keep this text.', attachments: [{ id: 'text-1', kind: 'text', name: 'latest.txt', content: 'Recent text' }] }] };
  const saved = chatForSave(chat);
  assert.equal(JSON.stringify(saved).includes('base64'), false);
  assert.equal(saved.messages[0].attachments[0].content, '');
  assert.equal(saved.messages[1].attachments[0].content, 'Recent text');
  assert.equal(saved.messages[0].attachments[0].name, 'notes.pdf');
  assert.equal(chat.messages[0].attachments[0].dataUrl, 'data:application/pdf;base64,abc=');
});
