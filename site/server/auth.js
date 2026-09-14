import { betterAuth } from 'better-auth';
import { captcha } from 'better-auth/plugins';

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;',
  })[character]);
}

async function sendEmail(env, { to, subject, intro, action, url }) {
  if (!env.RESEND_API_KEY || !env.AUTH_EMAIL_FROM) {
    throw new Error('Transactional email is not configured.');
  }
  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from: env.AUTH_EMAIL_FROM,
      to: [to],
      subject,
      html: `<div style="font-family:system-ui,sans-serif;line-height:1.6;color:#18212a;max-width:560px;margin:auto"><h1 style="font-size:24px">${escapeHtml(subject)}</h1><p>${escapeHtml(intro)}</p><p><a href="${escapeHtml(url)}" style="display:inline-block;padding:12px 18px;background:#4a9eda;color:#101820;text-decoration:none;border-radius:8px;font-weight:600">${escapeHtml(action)}</a></p><p style="font-size:13px;color:#66707a">If you did not request this email, you can ignore it.</p></div>`,
    }),
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new Error('The account email could not be sent.');
}

async function userHash(userId) {
  const bytes = new TextEncoder().encode(`brittain4:${userId}`);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].slice(0, 8).map(byte => byte.toString(16).padStart(2, '0')).join('');
}

export function createAuth(env, requestUrl) {
  if (!env.DB) throw new Error('The account database is not configured.');
  if (!env.BETTER_AUTH_SECRET || env.BETTER_AUTH_SECRET.length < 32) {
    throw new Error('The account secret is not configured.');
  }
  const requestOrigin = new URL(requestUrl).origin;
  const hostname = new URL(requestUrl).hostname;
  const localRequest = ['localhost', '127.0.0.1', '::1'].includes(hostname);
  const configuredOrigin = env.BETTER_AUTH_URL || requestOrigin;
  const configuredHost = new URL(configuredOrigin).hostname;
  const origin = localRequest || hostname !== configuredHost ? requestOrigin : configuredOrigin;
  const emailEnabled = Boolean(env.RESEND_API_KEY && env.AUTH_EMAIL_FROM);
  const plugins = [];
  if (env.TURNSTILE_SECRET_KEY) {
    plugins.push(captcha({
      provider: 'cloudflare-turnstile',
      secretKey: env.TURNSTILE_SECRET_KEY,
    }));
  }
  return betterAuth({
    database: env.DB,
    secret: env.BETTER_AUTH_SECRET,
    baseURL: origin,
    basePath: '/api/auth',
    trustedOrigins: [...new Set([origin, configuredOrigin])],
    appName: 'Brittain',
    emailAndPassword: {
      enabled: true,
      minPasswordLength: 10,
      maxPasswordLength: 128,
      requireEmailVerification: emailEnabled,
      revokeSessionsOnPasswordReset: true,
      ...(emailEnabled ? {
        sendResetPassword: ({ user, url }) => sendEmail(env, {
          to: user.email,
          subject: 'Reset your Brittain password',
          intro: 'Use this link to choose a new password. The link expires in one hour.',
          action: 'Reset password',
          url,
        }),
      } : {}),
    },
    emailVerification: emailEnabled ? {
      sendOnSignUp: true,
      sendOnSignIn: true,
      autoSignInAfterVerification: true,
      sendVerificationEmail: ({ user, url }) => sendEmail(env, {
        to: user.email,
        subject: 'Verify your Brittain account',
        intro: 'Verify your email address to start using Brittain 4.',
        action: 'Verify email',
        url,
      }),
    } : undefined,
    session: {
      expiresIn: 60 * 60 * 24 * 30,
      updateAge: 60 * 60 * 24,
      cookieCache: { enabled: true, maxAge: 60 * 5 },
    },
    user: {
      deleteUser: {
        enabled: true,
        beforeDelete: async user => {
          await env.DB.prepare('DELETE FROM chats WHERE user_id = ?').bind(user.id).run();
          try {
            await env.DB.prepare('DELETE FROM chat_exchanges WHERE user_hash = ?').bind(await userHash(user.id)).run();
          } catch { /* Account deletion must still work if the optional transcript table is absent. */ }
        },
      },
    },
    advanced: { cookiePrefix: 'brittain' },
    plugins,
  });
}

export async function accountSession(request, env) {
  return createAuth(env, request.url).api.getSession({ headers: request.headers });
}
