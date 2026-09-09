export function contextUsage(messages, limit = 32768) {
  const validLimit = Number.isInteger(limit) && limit > 0 ? limit : 32768;
  const latest = messages.at(-1);
  const reported = messages.findLast(m => Number.isInteger(m.usage?.total_tokens) && m.usage.total_tokens >= 0);
  const tokens = reported?.usage.total_tokens ?? (messages.length ? null : 0);
  return {
    tokens, limit: validLimit,
    percent: tokens === null ? null : Math.min(100, Math.round(tokens / validLimit * 1000) / 10),
    stale: Boolean(latest && reported !== latest),
  };
}
