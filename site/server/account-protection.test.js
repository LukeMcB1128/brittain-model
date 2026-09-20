import test from 'node:test';
import assert from 'node:assert/strict';
import { accountFeatures, protectAccountRequest } from './account-protection.js';

function request(path = '/sign-in/email', headers = {}, method = 'POST') {
  return new Request(`https://brittain.app/api/auth${path}`, { method, headers });
}

test('account requests use the Cloudflare client IP, not forwarded input', async () => {
  const keys = [];
  const env = { AUTH_RATE_LIMITER: { limit: async ({ key }) => { keys.push(key); return { success: true }; } } };
  assert.equal(await protectAccountRequest(request('/sign-in/email', { 'CF-Connecting-IP': '203.0.113.4', 'X-Forwarded-For': '1.2.3.4' }), env), null);
  assert.deepEqual(keys, ['203.0.113.4']);
  assert.equal((await protectAccountRequest(request('/sign-in/email', { 'X-Forwarded-For': '1.2.3.4' }), env)).status, 403);
});

test('signup and email endpoints share the stricter bucket, including trailing slash', async () => {
  const env = {
    AUTH_RATE_LIMITER: { limit: async () => ({ success: true }) },
    AUTH_EMAIL_RATE_LIMITER: { limit: async () => ({ success: false }) },
  };
  for (const path of ['/sign-up/email', '/request-password-reset', '/send-verification-email/']) {
    const response = await protectAccountRequest(request(path, { 'CF-Connecting-IP': '203.0.113.4' }), env);
    assert.equal(response.status, 429);
    assert.equal(response.headers.get('Retry-After'), '60');
  }
  assert.equal(await protectAccountRequest(request('/sign-in/email', { 'CF-Connecting-IP': '203.0.113.4' }), env), null);
});

test('login throttling leaves session reads and verification links available', async () => {
  const env = { AUTH_RATE_LIMITER: { limit: async () => ({ success: false }) } };
  assert.equal((await protectAccountRequest(request('/sign-in/email', { 'CF-Connecting-IP': '203.0.113.4' }), env)).status, 429);
  assert.equal(await protectAccountRequest(request('/get-session', {}, 'GET'), env), null);
  assert.equal(await protectAccountRequest(request('/verify-email', {}, 'GET'), env), null);
});

test('account config exposes the public key only when both Turnstile keys exist', () => {
  assert.equal(accountFeatures({ TURNSTILE_SITE_KEY: 'public' }).turnstileSiteKey, '');
  assert.deepEqual(accountFeatures({ TURNSTILE_SITE_KEY: 'public', TURNSTILE_SECRET_KEY: 'secret', RESEND_API_KEY: 'secret', AUTH_EMAIL_FROM: 'accounts@brittain.app' }), { emailVerification: true, turnstileSiteKey: 'public' });
});
