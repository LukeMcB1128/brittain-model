// Records of what was actually served, because nothing else keeps them.
//
// vLLM logs one line per request -- timestamp, client IP, status -- and no
// message content. The gateway streamed replies to the browser and forgot
// them. So 608 real exchanges, including a full Playwright-driven session,
// were unrecoverable. Real usage is the most valuable source of training and
// eval data this project has, and it was being discarded at the moment it was
// most worth keeping.
//
// WHY THE SINK IS DUCK-TYPED
// The gateway is bundled for a Worker runtime (platform: 'browser'), so there
// is no filesystem and storage must arrive as an env binding. Which bindings
// the host offers is not knowable from this repo, so rather than commit to one
// and break the chat if it is absent, this detects what is there:
//
//   env.CHAT_LOG      a KV-like object with .put()  -> one key per exchange
//   env.CHAT_LOG_URL  an endpoint                   -> POST the JSON entry
//                     (env.CHAT_LOG_TOKEN is sent as a bearer token if set)
//   neither           no-op, and the gateway behaves exactly as before
//
// Logging must never break a chat. Every failure here is swallowed: a record
// is worth less than the reply the user is waiting for.

// Attachments are base64 images and PDFs running to megabytes. They are the
// bulk of a request and worthless as a record, so they are replaced by a
// marker rather than stored.
const MAX_MESSAGE_CHARS = 20_000;
const MAX_TOOL_RESULT_CHARS = 4_000;
const MAX_ENTRY_CHARS = 400_000;

// Same shapes tools.js refuses to send outward. The chat corpus already showed
// what turns up in real conversations -- API keys and passwords in plain text
// -- and a record that preserves them is a liability that outlives the chat.
const SECRET_PATTERNS = [
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, '[REDACTED_PRIVATE_KEY]'],
  [/\bAIza[0-9A-Za-z_-]{35}\b/g, '[REDACTED_GOOGLE_API_KEY]'],
  [/\b(?:sk|ghp|gho|ghu|ghs|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{16,}/g, '[REDACTED_TOKEN]'],
  [/\bAKIA[0-9A-Z]{16}\b/g, '[REDACTED_AWS_KEY]'],
  [/\bBearer\s+[A-Za-z0-9._-]{20,}/g, 'Bearer [REDACTED_TOKEN]'],
  [/\b(password|passwd|pwd|api[_-]?key|secret|access[_-]?token)(\s*[:=]\s*)(?!\[REDACTED)([^\s"',;}\\]{4,})/gi, '$1$2[REDACTED]'],
];

export function redact(value) {
  let text = String(value ?? '');
  for (const [pattern, replacement] of SECRET_PATTERNS) text = text.replace(pattern, replacement);
  return text;
}

function clip(text, limit) {
  const value = redact(text);
  return value.length > limit ? `${value.slice(0, limit)}…[${value.length - limit} more characters]` : value;
}

// Content is either a string or an array of parts, and the image parts carry
// data URLs. Keep the words, count the images, store neither the bytes nor a
// truncated fragment of them.
function summarizeContent(content) {
  if (typeof content === 'string') return { text: clip(content, MAX_MESSAGE_CHARS) };
  if (!Array.isArray(content)) return { text: '' };
  const text = content.filter(part => part?.type === 'text').map(part => part.text).join('\n');
  const images = content.filter(part => part?.type === 'image_url').length;
  return images ? { text: clip(text, MAX_MESSAGE_CHARS), images } : { text: clip(text, MAX_MESSAGE_CHARS) };
}

// The user id identifies a person. Grouping exchanges by who sent them is
// useful; storing who they are is not, for this purpose. A truncated digest
// keeps the grouping and drops the identity.
async function anonymousUser(user) {
  try {
    const bytes = new TextEncoder().encode(`brittain4:${user}`);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(digest)].slice(0, 8).map(b => b.toString(16).padStart(2, '0')).join('');
  } catch {
    return 'unknown';
  }
}

export async function buildEntry({ user, messages, reply, toolCalls = [], usage, finishReason, error, startedAt, surface = 'web-chat' }) {
  const conversation = (messages || [])
    // The system prompt is the same on every request and is in source control.
    .filter(message => message.role !== 'system')
    .map(message => ({ role: message.role, ...summarizeContent(message.content) }));

  return {
    id: crypto.randomUUID(),
    at: new Date().toISOString(),
    surface,
    model: 'brittain4',
    user: await anonymousUser(user),
    durationMs: startedAt ? Date.now() - startedAt : null,
    finishReason: finishReason ?? null,
    usage: usage ?? null,
    error: error ? clip(error, 500) : null,
    messages: conversation,
    reply: reply ? clip(reply, MAX_MESSAGE_CHARS) : '',
    tools: toolCalls.map(call => ({
      name: call.name,
      arguments: clip(typeof call.arguments === 'string' ? call.arguments : JSON.stringify(call.arguments ?? {}), 2_000),
      ok: call.ok !== false,
      result: clip(call.result ?? '', MAX_TOOL_RESULT_CHARS),
    })),
  };
}

async function write(env, entry) {
  const body = JSON.stringify(entry);
  // A single runaway exchange should not poison the store.
  if (body.length > MAX_ENTRY_CHARS) return false;

  if (env?.CHAT_LOG && typeof env.CHAT_LOG.put === 'function') {
    // Sorts chronologically as a string, so a listing is in order.
    await env.CHAT_LOG.put(`chat/${entry.at}/${entry.id}`, body);
    return true;
  }
  if (env?.CHAT_LOG_URL) {
    const response = await fetch(env.CHAT_LOG_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(env.CHAT_LOG_TOKEN ? { Authorization: `Bearer ${env.CHAT_LOG_TOKEN}` } : {}),
      },
      body,
      signal: AbortSignal.timeout(5_000),
    });
    return response.ok;
  }
  return false;
}

// Returns whether anything was stored. Never throws: the caller is on the
// request path and a lost record must not become a lost reply.
export async function recordExchange(env, details) {
  try {
    if (!env?.CHAT_LOG && !env?.CHAT_LOG_URL) return false;
    return await write(env, await buildEntry(details));
  } catch {
    return false;
  }
}
