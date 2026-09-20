export function safeNextPath(value) {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || value.includes('\\')) return '/chat';
  if ([...value].some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127)) return '/chat';
  return value;
}

async function request(path, body, captchaToken) {
  const response = await fetch(`/api/auth${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(captchaToken ? { 'x-captcha-response': captchaToken } : {}),
    },
    body: JSON.stringify(body),
  });
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    throw new Error(data.message || data.error?.message || data.error || 'The account request failed.');
  }
  return data;
}

export async function getAccountSession() {
  const response = await fetch('/api/auth/get-session', { headers: { Accept: 'application/json' } });
  if (!response.ok) return null;
  return response.json();
}

export function signUp({ name, email, password, captchaToken }) {
  return request('/sign-up/email', { name, email, password, callbackURL: '/chat' }, captchaToken);
}

export function signIn({ email, password, captchaToken }) {
  return request('/sign-in/email', { email, password, rememberMe: true, callbackURL: '/chat' }, captchaToken);
}

export function signOut() {
  return request('/sign-out', {});
}

export function requestPasswordReset(email, captchaToken) {
  return request('/request-password-reset', { email, redirectTo: '/reset-password' }, captchaToken);
}

export function resetPassword(newPassword, token) {
  return request('/reset-password', { newPassword, token });
}

export function deleteAccount(password) {
  return request('/delete-user', { password, callbackURL: '/' });
}

export async function accountConfig() {
  const response = await fetch('/api/account/config', { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error('Account settings could not be loaded.');
  return response.json();
}
