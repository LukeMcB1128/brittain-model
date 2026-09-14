import { handleApi } from './gateway.js';

const INTERNAL_USER_HEADER = 'x-brittain-internal-user';

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

export class ChatCapacity {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.activeUsers = new Set();
    this.hourly = null;
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
    const hourly = await this.reserveHourly(hourlyMaximum);
    if (!hourly.allowed) {
      return json(
        { error: 'Brittain 4 has reached its hourly demo capacity. Please try again later.' },
        503,
        { 'Retry-After': String(hourly.retryAfter) },
      );
    }
    this.activeUsers.add(userId);
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      this.activeUsers.delete(userId);
    };
    try {
      const response = await handleApi(request, this.env, fetch, userId);
      if (!response.body) {
        release();
        return response;
      }
      const stream = new TransformStream();
      this.state.waitUntil(response.body.pipeTo(stream.writable).finally(release));
      return new Response(stream.readable, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } catch (error) {
      release();
      throw error;
    }
  }
}
