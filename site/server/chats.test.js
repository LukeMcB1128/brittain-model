import test from 'node:test';
import assert from 'node:assert/strict';
import { cleanChat, handleChats } from './chats.js';

const chatId = '11111111-1111-4111-8111-111111111111';

test('saved chats keep useful text and remove attachment binary data', () => {
  const chat = cleanChat({
    id: chatId,
    title: 'Attachment chat',
    messages: [{
      id: 'turn-1', prompt: 'Read this', answer: 'Done', status: 'done',
      attachments: [
        { id: 'image-1', name: 'photo.png', kind: 'image', type: 'image/png', dataUrl: 'data:image/png;base64,secret' },
        { id: 'pdf-1', name: 'notes.pdf', kind: 'pdf', type: 'application/pdf', content: 'Page 1 text', dataUrl: 'data:application/pdf;base64,secret' },
      ],
    }],
  });
  assert.equal(JSON.stringify(chat).includes('base64'), false);
  assert.equal(chat.messages[0].attachments[1].content, 'Page 1 text');
});

test('chat routes require an account and same-origin writes', async () => {
  const DB = { prepare() { throw new Error('must not query'); } };
  const anonymous = await handleChats(new Request('https://site.example/api/chats'), { DB }, null);
  assert.equal(anonymous.status, 401);
  const crossOrigin = await handleChats(new Request(`https://site.example/api/chats/${chatId}`, {
    method: 'DELETE', headers: { Origin: 'https://other.example' },
  }), { DB }, { id: 'user-1' });
  assert.equal(crossOrigin.status, 403);
});

test('chat deletion is scoped to the signed-in user', async () => {
  let bound;
  const DB = {
    prepare(sql) {
      assert.match(sql, /DELETE FROM chats WHERE id = \? AND user_id = \?/);
      return { bind(...values) { bound = values; return { run: async () => ({ meta: { changes: 1 } }) }; } };
    },
  };
  const response = await handleChats(new Request(`https://site.example/api/chats/${chatId}`, {
    method: 'DELETE', headers: { Origin: 'https://site.example' },
  }), { DB }, { id: 'user-1' });
  assert.equal(response.status, 200);
  assert.deepEqual(bound, [chatId, 'user-1']);
  assert.equal((await response.json()).deleted, true);
});
