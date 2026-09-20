const EMAIL_PATHS = new Set(['/api/auth/sign-up/email', '/api/auth/request-password-reset', '/api/auth/send-verification-email']);

export async function protectAccountRequest(request, env) {
  if (request.method !== 'POST') return null;
  // Cloudflare supplies this header. Do not trust client-supplied forwarding headers.
  const ip = request.headers.get('CF-Connecting-IP');
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(new URL(request.url).hostname);
  if (!ip && !local) return Response.json({ message: 'The request could not be verified.' }, { status: 403 });
  const key = ip || 'local-development';
  const path = new URL(request.url).pathname.replace(/\/+$/, '');
  const limits = [env.AUTH_RATE_LIMITER, ...(EMAIL_PATHS.has(path) ? [env.AUTH_EMAIL_RATE_LIMITER] : [])];
  for (const limiter of limits) {
    if (limiter && !(await limiter.limit({ key })).success) {
      return Response.json({ message: 'Too many account requests. Wait one minute and try again.' }, {
        status: 429,
        headers: { 'Cache-Control': 'no-store', 'Retry-After': '60' },
      });
    }
  }
  return null;
}

export function accountFeatures(env) {
  return {
    emailVerification: Boolean(env.RESEND_API_KEY && env.AUTH_EMAIL_FROM),
    turnstileSiteKey: env.TURNSTILE_SECRET_KEY && env.TURNSTILE_SITE_KEY ? env.TURNSTILE_SITE_KEY : '',
  };
}
