import { handleApi } from './gateway.js';

const INTERNAL_USER_HEADER = 'x-brittain-internal-user';
const REQUEST_ID_HEADER = 'x-brittain-request-id';
const REQUEST_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function json(body, status, extra = {}) {
  return Response.json(body, {
    status,
    headers: { 'Cache-Control': 'no-store', ...extra },
  });
}

export function internalChatRequest(request, userId) {
  const headers = new Headers(request.headers);
  headers.set(INTERNAL_USER_HEADER, userId);
  return new Request(request, { headers });
}

export function relayResponse(response, onClose, onReady) {
  if (!response.body) {
    onClose();
    return response;
  }
  const reader = response.body.getReader();
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    onClose();
  };
  let canceling;
  const cancelSource = reason => {
    close();
    if (!canceling) canceling = reader.cancel(reason).finally(() => {
      try { reader.releaseLock(); } catch { /* A read can still own the reader. */ }
    });
    return canceling;
  };
  const stream = new ReadableStream({
    async pull(controller) {
      try {
        const next = await reader.read();
        if (next.done) {
          close();
          reader.releaseLock();
          controller.close();
          return;
        }
        controller.enqueue(next.value);
      } catch (error) {
        close();
        try { reader.releaseLock(); } catch { /* The source still owns the reader. */ }
        controller.error(error);
      }
    },
    async cancel(reason) {
      await cancelSource(reason);
    },
  });
  onReady?.(cancelSource);
  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

export class ChatCapacity {
  constructor(state, env, handleChat = handleApi) {
    this.state = state;
    this.env = env;
    this.handleChat = handleChat;
    this.activeUsers = new Map();
    this.pendingStops = new Map();
    this.hourly = null;
  }

  purgeExpired(now = Date.now()) {
    for (const [userId, stop] of this.pendingStops) {
      if (stop.expiresAt <= now) this.pendingStops.delete(userId);
    }
    for (const entry of this.activeUsers.values()) {
      if (entry.expiresAt <= now) {
        entry.stop.abort('Response lock expired.');
        void entry.cancelSource?.('Response lock expired.').catch(() => {});
        entry.release();
      }
    }
  }

  async reserveHourly(maximum) {
    const reserve = async () => {
      const now = Date.now();
      const stored = this.state.storage ? await this.state.storage.get('hourly-capacity') : this.hourly;
      const current = stored && now - stored.startedAt < 60 * 60 * 1000
        ? stored
        : { startedAt: now, count: 0 };
      if (current.count >= maximum) {
        return { allowed: false, retryAfter: Math.max(1, Math.ceil((current.startedAt + 60 * 60 * 1000 - now) / 1000)) };
      }
      const next = { ...current, count: current.count + 1 };
      this.hourly = next;
      if (this.state.storage) await this.state.storage.put('hourly-capacity', next);
      return { allowed: true, retryAfter: 0 };
    };
    return this.state.blockConcurrencyWhile ? this.state.blockConcurrencyWhile(reserve) : reserve();
  }

  async fetch(request) {
    const userId = request.headers.get(INTERNAL_USER_HEADER);
    if (!userId) return json({ error: 'Sign in to use chat.' }, 401);
    const requestId = request.headers.get(REQUEST_ID_HEADER);
    if (requestId && !REQUEST_ID_PATTERN.test(requestId)) return json({ error: 'Response ID is not valid.' }, 400);
    this.purgeExpired();
    if (new URL(request.url).pathname === '/api/chat/cancel') {
      if (request.method !== 'POST') return json({ error: 'Method not allowed.' }, 405);
      if (!requestId) return json({ error: 'Response ID is required.' }, 400);
      const entry = this.activeUsers.get(userId);
      if (entry?.requestId === requestId) {
        entry.stop.abort('The user stopped the response.');
        void entry.cancelSource?.('The user stopped the response.').catch(() => {});
        entry.release();
      } else {
        // Stop can reach the Durable Object before the chat request does.
        // Remember it briefly so that late request cannot start a response.
        this.pendingStops.set(userId, { requestId, expiresAt: Date.now() + 30_000 });
      }
      return new Response(null, { status: 204, headers: { 'Cache-Control': 'no-store' } });
    }
    if (requestId && this.pendingStops.get(userId)?.requestId === requestId) {
      this.pendingStops.delete(userId);
      return json({ error: 'Request canceled.' }, 499);
    }
    if (this.activeUsers.has(userId)) {
      return json({ error: 'Your account already has a response in progress.' }, 409);
    }
    const configured = Number.parseInt(this.env.CHAT_MAX_CONCURRENT || '4', 10);
    const maximum = Number.isInteger(configured) && configured > 0 ? Math.min(configured, 32) : 4;
    if (this.activeUsers.size >= maximum) {
      return json({ error: 'Brittain 4 is at capacity. Wait a moment and retry.' }, 503, { 'Retry-After': '10' });
    }
    const configuredHourly = Number.parseInt(this.env.CHAT_MAX_PER_HOUR || '621', 10);
    const hourlyMaximum = Number.isInteger(configuredHourly) && configuredHourly > 0 ? configuredHourly : 621;
    const configuredLock = Number.parseInt(this.env.CHAT_LOCK_TIMEOUT_SECONDS || '210', 10);
    const lockSeconds = Number.isInteger(configuredLock) && configuredLock > 0 ? Math.min(configuredLock, 900) : 210;
    // Reserve before the first await: simultaneous requests must see the slot.
    const entry = {
      expiresAt: Date.now() + lockSeconds * 1000,
      requestId: requestId || crypto.randomUUID(),
      stop: new AbortController(),
      cancelSource: null,
      release: null,
    };
    this.activeUsers.set(userId, entry);
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      // An expired request must not release a newer request for the same user.
      if (this.activeUsers.get(userId) === entry) this.activeUsers.delete(userId);
      request.signal.removeEventListener('abort', onClientAbort);
    };
    const onClientAbort = () => {
      entry.stop.abort('The client disconnected.');
      void entry.cancelSource?.('The client disconnected.').catch(() => {});
      release();
    };
    entry.release = release;
    request.signal.addEventListener('abort', onClientAbort, { once: true });
    try {
      if (request.signal.aborted) { onClientAbort(); return json({ error: 'Request canceled.' }, 499); }
      const hourly = await this.reserveHourly(hourlyMaximum);
      if (entry.stop.signal.aborted) { release(); return json({ error: 'Request canceled.' }, 499); }
      if (!hourly.allowed) {
        release();
        return json(
          { error: `Brittain 4 has reached its hourly demo capacity. Try again in ${Math.ceil(hourly.retryAfter / 60)} minutes.` },
          503,
          { 'Retry-After': String(hourly.retryAfter) },
        );
      }
      const chatRequest = new Request(request, { signal: AbortSignal.any([request.signal, entry.stop.signal]) });
      const response = await this.handleChat(chatRequest, this.env, fetch, userId);
      if (entry.stop.signal.aborted) {
        void response.body?.cancel().catch(() => {});
        release();
        return json({ error: 'Request canceled.' }, 499);
      }
      return relayResponse(response, release, cancel => { entry.cancelSource = cancel; });
    } catch (error) {
      release();
      throw error;
    }
  }
}
