import test from 'node:test';
import assert from 'node:assert/strict';
import { cleanChat, handleChats } from './chats.js';

const chatId = '11111111-1111-4111-8111-111111111111';

test('saved chats keep useful text and remove attachment binary data', () => {
  const chat = cleanChat({
    id: chatId,
    title: 'Attachment chat',
    messages: [{
      id: 'turn-1', prompt: 'Read this', answer: 'Done', status: 'done', compactionStatus: 'done',
      attachments: [
        { id: 'image-1', name: 'photo.png', kind: 'image', type: 'image/png', dataUrl: 'data:image/png;base64,secret' },
        { id: 'pdf-1', name: 'notes.pdf', kind: 'pdf', type: 'application/pdf', content: 'Page 1 text', dataUrl: 'data:application/pdf;base64,secret' },
      ],
    }],
  });
  assert.equal(JSON.stringify(chat).includes('base64'), false);
  assert.equal(chat.messages[0].attachments[1].content, 'Page 1 text');
  assert.equal(chat.messages[0].compactionStatus, 'done');
});

test('chat list returns summaries without message payloads', async () => {
  const DB = {
    prepare(sql) {
      assert.match(sql, /SELECT id, title, created_at, updated_at/);
      return { bind: value => ({ all: async () => {
        assert.equal(value, 'user-1');
        return { results: [{ id: chatId, title: 'Saved chat', created_at: '2026-09-01', updated_at: '2026-09-02' }] };
      } }) };
    },
  };
  const response = await handleChats(new Request('https://site.example/api/chats'), { DB }, { id: 'user-1' });
  assert.deepEqual((await response.json()).chats, [{ id: chatId, title: 'Saved chat', createdAt: '2026-09-01', updatedAt: '2026-09-02' }]);
});

test('one chat is loaded only for its signed-in owner', async () => {
  let bound;
  const saved = cleanChat({ id: chatId, title: 'Loaded chat', messages: [] });
  const DB = {
    prepare(sql) {
      assert.match(sql, /WHERE id = \? AND user_id = \?/);
      return { bind(...values) { bound = values; return { first: async () => ({ payload: JSON.stringify(saved), created_at: '2026-09-01', updated_at: '2026-09-02' }) }; } };
    },
  };
  const response = await handleChats(new Request(`https://site.example/api/chats/${chatId}`), { DB }, { id: 'user-1' });
  assert.deepEqual(bound, [chatId, 'user-1']);
  assert.equal((await response.json()).chat.title, 'Loaded chat');
});

test('conversation rename updates the indexed title and saved payload', async () => {
  const statements = [];
  const saved = cleanChat({ id: chatId, title: 'Old title', messages: [] });
  const DB = {
    prepare(sql) {
      statements.push(sql);
      if (sql.includes('SELECT payload')) return { bind: () => ({ first: async () => ({ payload: JSON.stringify(saved) }) }) };
      return { bind: (...values) => ({ run: async () => { assert.equal(values[0], 'New title'); assert.equal(JSON.parse(values[1]).title, 'New title'); } }) };
    },
  };
  const response = await handleChats(new Request(`https://site.example/api/chats/${chatId}`, {
    method: 'PATCH', headers: { Origin: 'https://site.example', 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'New title' }),
  }), { DB }, { id: 'user-1' });
  assert.equal(response.status, 200);
  assert.match(statements.at(-1), /UPDATE chats SET title/);
  assert.equal((await response.json()).chat.title, 'New title');
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
