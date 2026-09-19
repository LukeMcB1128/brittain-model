const MAX_CHAT_BYTES = 900_000;
const ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f-]{27,}$/i;

function json(body, status = 200) {
  return Response.json(body, { status, headers: { 'Cache-Control': 'no-store' } });
}

function safeString(value, limit) {
  return typeof value === 'string' ? value.slice(0, limit) : '';
}

function cleanAttachment(item) {
  if (!item || typeof item !== 'object') return null;
  return {
    id: safeString(item.id, 100),
    name: safeString(item.name, 180),
    type: safeString(item.type, 100),
    kind: ['image', 'pdf', 'text'].includes(item.kind) ? item.kind : 'text',
    size: Number.isFinite(item.size) ? Math.max(0, Math.trunc(item.size)) : 0,
    truncated: Boolean(item.truncated),
    ...(['text', 'pdf'].includes(item.kind) ? { content: safeString(item.content, 120_000) } : {}),
  };
}

export function cleanChat(input) {
  if (!input || typeof input !== 'object' || !ID_PATTERN.test(input.id || '')) {
    throw new Error('The conversation is not valid.');
  }
  const messages = Array.isArray(input.messages) ? input.messages.slice(0, 200).map(message => ({
    id: safeString(message?.id, 100),
    prompt: safeString(message?.prompt, 500_000),
    answer: safeString(message?.answer, 500_000),
    status: ['done', 'error', 'stopped'].includes(message?.status) ? message.status : 'done',
    error: safeString(message?.error, 1_000),
    note: safeString(message?.note, 1_000),
    attachments: Array.isArray(message?.attachments) ? message.attachments.map(cleanAttachment).filter(Boolean).slice(0, 10) : [],
    tools: Array.isArray(message?.tools) ? message.tools.slice(0, 20).map(tool => ({
      id: safeString(tool?.id, 100),
      name: safeString(tool?.name, 50),
      label: safeString(tool?.label, 100),
      detail: safeString(tool?.detail, 1_000),
      result: safeString(tool?.result, 1_000),
      status: safeString(tool?.status, 20),
      sources: Array.isArray(tool?.sources) ? tool.sources.slice(0, 10).map(source => ({
        title: safeString(source?.title, 200),
        url: safeString(source?.url, 2_000),
      })).filter(source => /^https?:\/\//i.test(source.url)) : [],
    })) : [],
    usage: message?.usage && Number.isInteger(message.usage.total_tokens) ? {
      prompt_tokens: Number(message.usage.prompt_tokens) || 0,
      completion_tokens: Number(message.usage.completion_tokens) || 0,
      total_tokens: message.usage.total_tokens,
    } : undefined,
  })) : [];
  const chat = {
    id: input.id,
    title: safeString(input.title, 80) || 'New chat',
    messages,
    memory: safeString(input.memory, 30_000),
    contextStart: Number.isInteger(input.contextStart) ? Math.max(0, input.contextStart) : 0,
  };
  if (new TextEncoder().encode(JSON.stringify(chat)).length > MAX_CHAT_BYTES) {
    throw new Error('This conversation is too large to save. Start a new chat.');
  }
  return chat;
}

function sameOrigin(request) {
  const origin = request.headers.get('origin');
  return !origin || origin === new URL(request.url).origin;
}

export async function handleChats(request, env, user) {
  if (!user) return json({ error: 'Sign in to view conversations.' }, 401);
  if (!env.DB) return json({ error: 'Conversation storage is not configured.' }, 503);
  const url = new URL(request.url);
  const match = url.pathname.match(/^\/api\/chats(?:\/([^/]+))?$/);
  if (!match) return json({ error: 'Not found.' }, 404);
  const id = match[1] ? decodeURIComponent(match[1]) : '';
  if (request.method === 'GET' && !id) {
    const result = await env.DB.prepare(`
      SELECT id, title, created_at, updated_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100
    `).bind(user.id).all();
    return json({ chats: (result.results || []).map(row => ({
      id: row.id,
      title: row.title || 'New chat',
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    })) });
  }
  if (request.method === 'GET' && id) {
    if (!ID_PATTERN.test(id)) return json({ error: 'Conversation not found.' }, 404);
    const row = await env.DB.prepare(`
      SELECT payload, created_at, updated_at FROM chats WHERE id = ? AND user_id = ?
    `).bind(id, user.id).first();
    if (!row) return json({ error: 'Conversation not found.' }, 404);
    try {
      return json({ chat: { ...JSON.parse(row.payload), createdAt: row.created_at, updatedAt: row.updated_at } });
    } catch {
      return json({ error: 'The saved conversation is not valid.' }, 500);
    }
  }
  if (!sameOrigin(request)) return json({ error: 'Request origin is not allowed.' }, 403);
  if (request.method === 'PUT' && id) {
    if (!request.headers.get('content-type')?.includes('application/json')) return json({ error: 'Expected JSON.' }, 415);
    let input;
    try { input = await request.json(); } catch { return json({ error: 'Invalid request.' }, 400); }
    let chat;
    try { chat = cleanChat({ ...input, id }); } catch (error) { return json({ error: error.message }, 400); }
    const existing = await env.DB.prepare('SELECT user_id FROM chats WHERE id = ?').bind(id).first();
    if (existing && existing.user_id !== user.id) return json({ error: 'Conversation not found.' }, 404);
    const now = new Date().toISOString();
    await env.DB.prepare(`
      INSERT INTO chats (id, user_id, title, payload, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET title = excluded.title, payload = excluded.payload, updated_at = excluded.updated_at
      WHERE chats.user_id = excluded.user_id
    `).bind(id, user.id, chat.title, JSON.stringify(chat), now, now).run();
    return json({ chat: { ...chat, updatedAt: now } });
  }
  if (request.method === 'PATCH' && id) {
    if (!ID_PATTERN.test(id)) return json({ error: 'Conversation not found.' }, 404);
    if (!request.headers.get('content-type')?.includes('application/json')) return json({ error: 'Expected JSON.' }, 415);
    let input;
    try { input = await request.json(); } catch { return json({ error: 'Invalid request.' }, 400); }
    const title = safeString(input?.title, 80).trim();
    if (!title) return json({ error: 'Enter a conversation name.' }, 400);
    const row = await env.DB.prepare('SELECT payload FROM chats WHERE id = ? AND user_id = ?').bind(id, user.id).first();
    if (!row) return json({ error: 'Conversation not found.' }, 404);
    let chat;
    try { chat = cleanChat({ ...JSON.parse(row.payload), id, title }); }
    catch { return json({ error: 'The saved conversation is not valid.' }, 500); }
    const now = new Date().toISOString();
    await env.DB.prepare(`
      UPDATE chats SET title = ?, payload = ?, updated_at = ? WHERE id = ? AND user_id = ?
    `).bind(title, JSON.stringify(chat), now, id, user.id).run();
    return json({ chat: { id, title, updatedAt: now } });
  }
  if (request.method === 'DELETE' && id) {
    if (!ID_PATTERN.test(id)) return json({ error: 'Conversation not found.' }, 404);
    const result = await env.DB.prepare('DELETE FROM chats WHERE id = ? AND user_id = ?').bind(id, user.id).run();
    return json({ deleted: Number(result.meta?.changes || 0) > 0 });
  }
  return json({ error: 'Not found.' }, 404);
}
