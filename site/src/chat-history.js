export function assistantToolHistory(tools) {
  if (!Array.isArray(tools)) return [];
  return tools
    .filter(tool => tool?.name && tool.status !== 'running')
    .slice(0, 10)
    .map(tool => ({
      name: String(tool.name).slice(0, 40),
      detail: String(tool.detail || '').replace(/[\r\n]+/g, ' ').slice(0, 80),
    }));
}
