import { accountSession, createAuth } from './auth.js';
import { ChatCapacity, internalChatRequest } from './capacity.js';
import { handleChats } from './chats.js';
import { handleApi } from './gateway.js';

function securityHeaders(response) {
  const headers = new Headers(response.headers);
  headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  headers.set('X-Content-Type-Options', 'nosniff');
  headers.set('X-Frame-Options', 'DENY');
  headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  if (!headers.has('Content-Security-Policy')) {
    headers.set('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' https://challenges.cloudflare.com; frame-src https://challenges.cloudflare.com; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'");
  }
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function unavailable(message) {
  return Response.json({ error: message }, { status: 503, headers: { 'Cache-Control': 'no-store' } });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (url.pathname.startsWith('/api/auth/')) {
        return securityHeaders(await createAuth(env, request.url).handler(request));
      }
      if (url.pathname === '/api/account/config' && request.method === 'GET') {
        return securityHeaders(Response.json({
          emailVerification: Boolean(env.RESEND_API_KEY && env.AUTH_EMAIL_FROM),
          turnstileSiteKey: env.TURNSTILE_SITE_KEY || '',
        }, { headers: { 'Cache-Control': 'no-store' } }));
      }
      if (url.pathname.startsWith('/api/')) {
        const session = await accountSession(request, env);
        if (url.pathname.startsWith('/api/chats')) {
          return securityHeaders(await handleChats(request, env, session?.user));
        }
        if (url.pathname === '/api/chat' && session?.user) {
          if (env.CHAT_RATE_LIMITER) {
            const allowed = await env.CHAT_RATE_LIMITER.limit({ key: session.user.id });
            if (!allowed.success) {
              return securityHeaders(Response.json(
                { error: 'You are sending messages too quickly. Wait a moment and retry.' },
                { status: 429, headers: { 'Cache-Control': 'no-store' } },
              ));
            }
          }
          if (env.CHAT_CAPACITY) {
            const gate = env.CHAT_CAPACITY.get(env.CHAT_CAPACITY.idFromName('brittain-4'));
            return securityHeaders(await gate.fetch(internalChatRequest(request, session.user.id)));
          }
        }
        return securityHeaders(await handleApi(request, env, fetch, session?.user?.id || null));
      }
      // Static files and SPA navigation are served by the Workers asset
      // router. The Worker only handles API requests.
      return securityHeaders(new Response(null, { status: 404 }));
    } catch (error) {
      console.error('Brittain Worker request failed', error);
      return securityHeaders(unavailable('The site service is not configured yet.'));
    }
  },
};

export { ChatCapacity };
