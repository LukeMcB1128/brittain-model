export async function readCompletion(response, onChunk) {
  if (!response.ok) {
    let data;
    try { data = await response.json(); } catch { /* Handle proxy HTML or empty errors. */ }
    throw new Error(typeof data?.error === 'string' ? data.error : data?.error?.message || `Chat request failed (${response.status}).`);
  }
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) throw new Error('Chat returned an unexpected response. Please retry.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  let done = false;
  function consume(line) {
    if (!line.startsWith('data:')) return;
    const value = line.slice(5).trim();
    if (!value) return;
    if (value === '[DONE]') { done = true; return; }
    const data = JSON.parse(value);
    if (data.error) throw new Error(typeof data.error === 'string' ? data.error : data.error.message || 'The response failed.');
    if (data.type) {
      if (data.type === 'error') throw new Error(data.error || 'The response failed.');
      if (data.type === 'done') done = true;
      onChunk(data);
      return;
    }
    const choice = data.choices?.find(c => c.index === 0);
    onChunk({ type: 'content', text: choice?.delta?.content || '', finishReason: choice?.finish_reason, usage: data.usage });
  }
  try {
    while (!done) {
      const next = await reader.read();
      pending += decoder.decode(next.value, { stream: !next.done });
      const lines = pending.split(/\r?\n/);
      pending = lines.pop();
      for (const line of lines) { consume(line); if (done) break; }
      if (next.done) { if (pending && !done) consume(pending); break; }
    }
    if (!done) throw new Error('The connection ended before the reply finished. You can retry.');
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
