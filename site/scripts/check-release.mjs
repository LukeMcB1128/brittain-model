// Read-only: no login attempts, email, chat generation, or database changes.
const origin = new URL(process.argv[2] || 'https://brittain.app').origin;
const staging = process.argv.includes('--staging');
let failures = 0;
function check(ok, message) { console.log(`${ok ? 'PASS' : 'FAIL'} ${message}`); if (!ok) failures++; }
async function get(path) {
  return fetch(`${origin}${path}`, { signal: AbortSignal.timeout(15000), redirect: 'manual' });
}
try {
  for (const path of ['/', '/chat', '/models', '/privacy', '/chat/11111111-1111-4111-8111-111111111111']) {
    const response = await get(path);
    const html = await response.text();
    check(response.status === 200 && html.includes('id="root"') && !html.includes('signin-with-chatgpt'), `${path} serves the standalone application`);
    check(Boolean(response.headers.get('Content-Security-Policy')) && Boolean(response.headers.get('Strict-Transport-Security')), `${path} includes security headers`);
    if (staging || path.startsWith('/chat')) check(response.headers.get('X-Robots-Tag')?.includes('noindex'), `${path} is excluded from indexing`);
  }
  const health = await get('/api/health');
  const status = await health.json();
  check(health.ok && status.status === 'ok', 'Worker configuration and database are available (does not test model inference)');
  check(status.chatPaused === staging, staging ? 'Staging chat generation is paused' : 'Chat maintenance mode is off');
  const response = await get('/api/account/config');
  const config = await response.json();
  check(response.ok && Boolean(config.turnstileSiteKey), 'Turnstile is configured');
  if (staging) console.log('SKIP email delivery is not configured on staging; required for production release.');
  else check(response.ok && config.emailVerification === true, 'Email verification and password recovery are configured');
  check((await get('/api/chats')).status === 401, 'Anonymous users cannot read chat history');
  for (const path of ['/robots.txt', '/sitemap.xml', '/brittain-favicon.svg']) {
    const response = await get(path);
    check(response.ok && !response.headers.get('content-type')?.includes('text/html'), `${path} is served as a static file`);
  }
} catch (error) { check(false, `Check interrupted: ${error.message}`); }
console.log(`${failures} failed check(s) for ${origin}.`);
process.exitCode = failures ? 1 : 0;
