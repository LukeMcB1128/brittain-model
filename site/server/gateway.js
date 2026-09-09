const UPSTREAM = 'https://fragility-devoutly-dazzling.ngrok-free.dev/v1/chat/completions';
const MAX_BYTES = 180000;
function json(body, status = 200) {
  return Response.json(body, { status, headers: { 'Cache-Control': 'no-store' } });
}
export async function handleApi(request, env, fetchUpstream = fetch) {
  const path = new URL(request.url).pathname;
  const user = request.headers.get('oai-authenticated-user-id');
  if (!user) return json({ error: 'Sign in to use chat.' }, 401);
  if (path === '/api/session' && request.method === 'GET') {
    if (!env.BRITTAIN4_API_KEY) return json({ authenticated: true, configured: false, ready: false });
    try {
      const response = await fetchUpstream(UPSTREAM.replace('/chat/completions', '/models'), {
        headers: { Authorization: `Bearer ${env.BRITTAIN4_API_KEY}`, 'ngrok-skip-browser-warning': '1' },
        signal: AbortSignal.any([request.signal, AbortSignal.timeout(8000)]),
      });
      if (!response.ok) throw new Error();
      const data = await response.json();
      const model = data.data?.find(m => m.id === 'brittain4');
      if (!model) throw new Error();
      return json({ authenticated: true, configured: true, ready: true, model: 'brittain4', context: model.max_model_len || 32768 });
    } catch { return json({ authenticated: true, configured: true, ready: false }); }
  }
  if (path !== '/api/chat' || request.method !== 'POST') return json({ error: 'Not found.' }, 404);
  const origin = request.headers.get('origin');
  if (origin && origin !== new URL(request.url).origin) return json({ error: 'Request origin is not allowed.' }, 403);
  if (!env.BRITTAIN4_API_KEY) return json({ error: 'Chat is waiting for its server connection. Please try again later.' }, 503);
  if (!request.headers.get('content-type')?.includes('application/json')) return json({ error: 'Expected JSON.' }, 415);
  if (Number(request.headers.get('content-length')) > MAX_BYTES) return json({ error: 'This conversation is too large. Start a new chat.' }, 413);
  let body;
  try {
    const reader = request.body.getReader();
    let bytes = 0;
    const chunks = [];
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      bytes += value.length;
      if (bytes > MAX_BYTES) { await reader.cancel(); return json({ error: 'This conversation is too large. Start a new chat.' }, 413); }
      chunks.push(value);
    }
    const buffer = new Uint8Array(bytes);
    let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
    body = JSON.parse(new TextDecoder().decode(buffer));
  } catch { return json({ error: 'Invalid request.' }, 400); }
  if (!Array.isArray(body.messages) || !body.messages.length || body.messages.length > 200 || body.messages.some(m => !m || !['user', 'assistant'].includes(m.role) || typeof m.content !== 'string' || !m.content.trim()) || body.messages.at(-1).role !== 'user') {
    return json({ error: 'Send a non-empty conversation ending with a user message.' }, 400);
  }
  // Reject very large input early. Exact token accounting belongs to the model server.
  // A context error is shown intact to the user; history is never silently trimmed.
  const payload = {
    model: 'brittain4', messages: body.messages.map(({ role, content }) => ({ role, content })),
    max_tokens: 2048, temperature: 0.7, stream: true,
    stream_options: { include_usage: true }, chat_template_kwargs: { enable_thinking: false },
  };
  let upstream;
  try {
    upstream = await fetchUpstream(UPSTREAM, {
      method: 'POST', headers: { Authorization: `Bearer ${env.BRITTAIN4_API_KEY}`, 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify(payload), signal: AbortSignal.any([request.signal, AbortSignal.timeout(180000)]),
    });
  } catch (error) {
    return json({ error: error.name === 'TimeoutError' ? 'The model took too long to respond. Please retry.' : 'Cannot reach the model server. Please retry.' }, 502);
  }
  if (!upstream.ok) {
    let detail;
    try { const data = await upstream.json(); detail = typeof data.error === 'string' ? data.error : data.error?.message; } catch { /* Never return HTML tunnel errors. */ }
    const status = upstream.status;
    const message = status === 401 || status === 403 ? 'The model server rejected its access key. The site owner must update the connection.'
      : status === 429 ? 'The model server is busy. Please wait and retry.'
      : status === 400 ? `The model could not accept this request. ${detail || 'The conversation may exceed the 32k context limit. Start a new chat or shorten the message.'}`
      : 'The model server is unavailable. Please retry.';
    const headers = new Headers({ 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    if (upstream.headers.has('retry-after')) headers.set('Retry-After', upstream.headers.get('retry-after'));
    return new Response(JSON.stringify({ error: message }), { status: status === 401 || status === 403 ? 502 : status, headers });
  }
  if (!upstream.headers.get('content-type')?.includes('text/event-stream') || !upstream.body) return json({ error: 'The server returned an unexpected response. Please retry.' }, 502);
  return new Response(upstream.body, { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no' } });
}
